/**
 * @file KMeansMPI.h - implementation of k-means clustering via MPI
 * @author Justin Thoreson
 */

#pragma once
#include <vector>
#include <random>
#include <algorithm>
#include <array>
#include <iostream>
#include "mpi.h"

/**
 * @class Abstract k-means MPI class
 * @tparam k the number of clusters for k-means
 * @tparam d the dimensionality of a data element
 */ 
template <int k, int d>
class KMeansMPI {
public:
    // helpful definitions
    using Element = std::array<u_char, d>;
    class Cluster;
    using Clusters = std::array<Cluster, k>;
    const int MAX_FIT_STEPS = 300;

    // debugging
    const bool VERBOSE = false;  // set to true for debugging output
#define V(stuff) if(VERBOSE) {using namespace std; stuff}

    /**
     * Expose the clusters to the client readonly.
     * @return clusters from latest call to fit()
     */
    virtual const Clusters& getClusters() {
        return clusters;
    }

    /**
     * Main k-means clustering algorithm
     * Called by ROOT process, others call fitWork directly
     * @param data The data elements for k-means
     * @param nData The number of data elements
     */
    virtual void fit(const Element* data, int nData) {
        elements = data;
        n = nData;
        fitWork(ROOT);
    }

    /**
     * Per-process work for fitting
     * @param rank Process rank within MPI_COMM_WORLD
     * @pre n and elements are set in ROOT process; all p processes call fitWork simultaneously
     * @post clusters are now stable (or we gave up after MAX_FIT_STEPS)
     */
    virtual void fitWork(int rank) {
        bcastSize();
        partitionElements(rank);
        if (rank == ROOT)
            reseedClusters();
        bcastCentroids(rank);
        Clusters prior = clusters;
        prior[0].centroid[0]++;  // just to make it different the first time
        for (int generation = 0; generation < MAX_FIT_STEPS && prior != clusters; generation++) {
            V(cout<<rank<<" working on generation "<<generation<<endl;)
            updateDistances();
            prior = clusters;
            updateClusters();
            mergeClusters(rank);
            bcastCentroids(rank);
        }
        consolidateElementsByCluster(rank);
        delete[] partition;
        delete[] elementIds;
        partition = nullptr;
        elementIds = nullptr;
    }

    /**
     * The algorithm constructs k clusters and attempts to populate them with like neighbors.
     * This inner class, Cluster, holds each cluster's centroid (mean) and the index of the objects
     * belonging to this cluster.
     */
    struct Cluster {
        Element centroid;  // the current center (mean) of the elements in the cluster
        std::vector<int> elements;

        /**
         * Equality is just the centroids, regarless of elements
         */
        friend bool operator==(const Cluster& left, const Cluster& right) {
            return left.centroid == right.centroid;  // equality means the same centroid, regardless of elements
        }
    };

protected:
    const int ROOT = 0;                      // root process in MPI communicator
    const Element* elements = nullptr;       // set of elements to classify into k categories (supplied to latest call to fit())
    Element* partition = nullptr;            // parition of elements for the current process
    int* elementIds = nullptr;               // locally track indices in this->elements
    int n = 0;                               // number of elements in this->elements
    int m = 0;                               // max number of elements in this->partition
    int p = 0;                               // number of processes in MPI_COMM_WORLD
    Clusters clusters;                       // k clusters resulting from latest call to fit()
    std::vector<std::array<double,k>> dist;  // dist[i][j] is the distance from elements[i] to clusters[j].centroid

    /**
     * Send the number of elements to all other proecesses
     */
    virtual void bcastSize() {
        MPI_Bcast(&n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    }

    /**
     * Scatter elements amongs all processes
     * @param rank The ID of the current process
     */
    virtual void partitionElements(int rank) {
        MPI_Comm_size(MPI_COMM_WORLD, &p);
        u_char* sendbuf = nullptr, *recvbuf = nullptr;
        int* sendcounts = nullptr, *displs = nullptr;
        int elemsPerProc = n / p;

        // marshall
        if (rank == ROOT) {
            sendbuf = new u_char[n * (d + 1)];
            sendcounts = new int[p];
            displs = new int[p];
            int bufIndex = 0;
            for (int elemIndex = 0; elemIndex < n; elemIndex++) {
                for (int dimIndex = 0; dimIndex < d; dimIndex++)
                    sendbuf[bufIndex++] = elements[elemIndex][dimIndex];
                sendbuf[bufIndex++] = (u_char)elemIndex;
            }
            for (int procIndex = 0; procIndex < p; procIndex++) {
                displs[procIndex] = procIndex * elemsPerProc * (d + 1);
                sendcounts[procIndex] = elemsPerProc * (d + 1);
                if (procIndex == p - 1)
                    sendcounts[procIndex] = bufIndex - ((p - 1) * elemsPerProc * (d + 1));
            }
        }

        // set this->m for current process
        m = elemsPerProc;
        if (rank == p - 1)
            m = n - (elemsPerProc * (p - 1));
        dist.resize(m);

        // set up receiving side of message (everyone)
        int recvcount = m * (d + 1);
        recvbuf = new u_char[recvcount];

        // scatter
        MPI_Scatterv(
            sendbuf, sendcounts, displs, MPI_UNSIGNED_CHAR,
            recvbuf, recvcount, MPI_UNSIGNED_CHAR,
            ROOT, MPI_COMM_WORLD
        );

        // unmarshal
        partition = new Element[m];
        elementIds = new int[m];
        int bufIndex = 0;
        for (int elemIndex = 0; elemIndex < m; elemIndex++) {
            for (int dimIndex = 0; dimIndex < d; dimIndex++)
                partition[elemIndex][dimIndex] = recvbuf[bufIndex++];
            elementIds[elemIndex] = (int)recvbuf[bufIndex++];
        }
        delete[] sendbuf;
        delete[] recvbuf;
        delete[] sendcounts;
        delete[] displs;
    }

    /**
     * Reduce all processes' clusters
     * @param rank The ID of the current process
     */
    virtual void mergeClusters(int rank) {
        int sendCount = k * (d + 1), recvCount = p * sendCount;
        u_char* sendbuf = new u_char[sendCount], *recvbuf = nullptr;

        // marshall
        int bufIndex = 0;
        for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
            for (int dimIndex = 0; dimIndex < d; dimIndex++)
                sendbuf[bufIndex++] = clusters[clusterIndex].centroid[dimIndex];
            sendbuf[bufIndex++] = (u_char)clusters[clusterIndex].elements.size();
        }

        // gather
        if (rank == ROOT)
            recvbuf = new u_char[recvCount];
        MPI_Gather(
            sendbuf, sendCount, MPI_UNSIGNED_CHAR,
            recvbuf, sendCount, MPI_UNSIGNED_CHAR,
            ROOT, MPI_COMM_WORLD
        );

        // unmarshal
        if (rank == ROOT) {
            // track accumulation of cluster sizes for proper averaging
            std::array<int, k> clusterSizes;
            for (int clusterIndex = 0; clusterIndex < k; clusterIndex++)
                clusterSizes[clusterIndex] = clusters[clusterIndex].elements.size();
            
            // average out all the centroids
            bufIndex = 0;
            for (int procIndex = 0; procIndex < p; procIndex++)
                for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
                    Element centroid = Element{};
                    for (int dimIndex = 0; dimIndex < d; dimIndex++)
                        centroid[dimIndex] = recvbuf[bufIndex++];
                    int size = (int)recvbuf[bufIndex++];
                    accum(
                        clusters[clusterIndex].centroid,
                        clusterSizes[clusterIndex],
                        centroid, size
                    );
                    clusterSizes[clusterIndex] += size;
                }
        }
        delete[] recvbuf;
        delete[] sendbuf;
    }

    /**
     * Gather all element IDs for each cluster across processes
     * @param rank The ID of the current process
     */
    virtual void consolidateElementsByCluster(int rank) {
        int sendcount = m + k;
        u_char* sendbuf = new u_char[sendcount], *recvbuf = nullptr;
        int* recvcounts = nullptr, *displs = nullptr;
        int bufIndex = 0;

        // marshal
        for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
            sendbuf[bufIndex++] = (u_char)clusters[clusterIndex].elements.size();
            for (int& elemIndex : clusters[clusterIndex].elements)
                sendbuf[bufIndex++] = (u_char)elementIds[elemIndex];
        }

        // gather
        if (rank == ROOT) {
            recvbuf = new u_char[n + k * p];
            recvcounts = new int[p];
            displs = new int[p];
            int elemsPerProc = n / p;
            for (int procIndex = 0; procIndex < p; procIndex++) {
                recvcounts[procIndex] = elemsPerProc + k;
                if (procIndex == p - 1)
                    recvcounts[procIndex] = (n - (elemsPerProc * (p - 1))) + k;
                displs[procIndex] = procIndex * (elemsPerProc + k);
            }
        }
        MPI_Gatherv(
            sendbuf, sendcount, MPI_UNSIGNED_CHAR,
            recvbuf, recvcounts, displs, MPI_UNSIGNED_CHAR,
            ROOT, MPI_COMM_WORLD
        );

        // unmarshal
        if (rank == ROOT) {
            bufIndex = 0;
            for (Cluster& cluster : clusters)
                cluster.elements.clear();
            for (int procIndex = 0; procIndex < p; procIndex++)
                for (int clusterIndex = 0; clusterIndex < k; clusterIndex++) {
                    int size = (int)recvbuf[bufIndex++];
                    for (int e = 0; e < size; e++)
                        clusters[clusterIndex].elements.push_back((u_char)recvbuf[bufIndex++]);
                }
        }
        delete[] sendbuf;
        delete[] recvbuf;
        delete[] recvcounts;
        delete[] displs;
    }

    /**
     * Broadcast cluster centroids to all processes
     * @param rank The ID of the current process
     */
    virtual void bcastCentroids(int rank) {
        V(cout<<" "<<rank<<" bcastCentroids"<<endl;)
        int count = k * d;
        u_char* buffer = new u_char[count];
        if (rank == ROOT) {
            int bufIndex = 0;
            for (int clusterIndex = 0; clusterIndex < k; clusterIndex++)
                for (int dimIndex = 0; dimIndex < d; dimIndex++)
                    buffer[bufIndex++] = clusters[clusterIndex].centroid[dimIndex];
            V(cout<<" "<<rank<<" sending centroids ";for(int x=0;x<count;x++)printf("%03x ",buffer[x]);cout<<endl;)
        }
        MPI_Bcast(buffer, count, MPI_UNSIGNED_CHAR, ROOT, MPI_COMM_WORLD);
        if (rank != ROOT) {
            int bufIndex = 0;
            for (int clusterIndex = 0; clusterIndex < k; clusterIndex++)
                for (int dimIndex = 0; dimIndex < d; dimIndex++)
                    clusters[clusterIndex].centroid[dimIndex] = buffer[bufIndex++];
            V(cout<<" "<<rank<<" receiving centroids ";for(int x=0;x<count;x++)printf("%03x ",buffer[x]);cout<<endl;)
        }
        delete[] buffer;
    }

    /**
     * Get the initial cluster centroids.
     * Default implementation here is to just pick k elements at random from the element
     * set
     * @return list of clusters made by using k random elements as the initial centroids
     */
    virtual void reseedClusters() {
        std::vector<int> seeds;
        std::vector<int> candidates(n);
        std::iota(candidates.begin(), candidates.end(), 0);
        auto random = std::mt19937{std::random_device{}()};
        // Note that we need C++20 for std::sample
        std::sample(candidates.begin(), candidates.end(), back_inserter(seeds), k, random);
        for (int i = 0; i < k; i++) {
            clusters[i].centroid = elements[seeds[i]];
            clusters[i].elements.clear();
        }
    }

    /**
     * Calculate the distance from each element to each centroid.
     * Place into this->dist which is a k-vector of distances from each element to the kth centroid.
     */
    virtual void updateDistances() {
        for (int i = 0; i < m; i++) {
            V(cout<<"distances for "<<i<<"(";for(int x=0;x<d;x++)printf("%02x ",partition[i][x]);)
            for (int j = 0; j < k; j++) {
                dist[i][j] = distance(clusters[j].centroid, partition[i]);
                V(cout<<" " << dist[i][j];)
            }
            V(cout<<endl;)
        }
    }

    /**
     * Recalculate the current clusters based on the new distances shown in this->dist.
     */
    virtual void updateClusters() {
        // reinitialize all the clusters
        for (int j = 0; j < k; j++) {
            clusters[j].centroid = Element{};
            clusters[j].elements.clear();
        }
        // for each element, put it in its closest cluster (updating the cluster's centroid as we go)
        for (int i = 0; i < m; i++) {
            int min = 0;
            for (int j = 1; j < k; j++)
                if (dist[i][j] < dist[i][min])
                    min = j;
            accum(clusters[min].centroid, clusters[min].elements.size(), partition[i], 1);
            clusters[min].elements.push_back(i);
        }
    }

    /**
     * Method to update a centroid with additional element(s)
     * @param centroid   accumulating mean of the elements in a cluster so far
     * @param centroid_n number of elements in the cluster so far
     * @param addend     another element(s) to be added; if multiple, addend is their mean
     * @param addend_n   number of addends represented in the addend argument
     */
    virtual void accum(Element& centroid, int centroid_n, const Element& addend, int addend_n) const {
        int new_n = centroid_n + addend_n;
        for (int i = 0; i < d; i++) {
            double new_total = (double)centroid[i] * centroid_n + (double)addend[i] * addend_n;
            centroid[i] = (u_char)(new_total / new_n);
        }
    }

    /**
     * Subclass-supplied method to calculate the distance between two elements
     * @param a one element
     * @param b another element
     * @return distance from a to b (or more abstract metric); distance(a,b) >= 0.0 always
     */
    virtual double distance(const Element& a, const Element& b) const = 0;
};