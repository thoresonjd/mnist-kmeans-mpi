/**
 * @file MNISTKMeansSequential.cpp - sequential k-means clustering on MNIST
 * @author Justin Thoreson
 * @see http://yann.lecun.com/exdb/mnist/
 */

#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <random>
#include "MNISTKMeans.h"
#include "mpi.h"

const int K = 10;
const int ROOT = 0;
const int IMAGE_LIMIT = 1000;
const std::string MNIST_IMAGES_FILEPATH = "./t10k-images-idx3-ubyte";
const std::string MNIST_LABELS_FILEPATH = "./t10k-labels-idx1-ubyte";

/**
 * Reads and unmarshals the MNIST images data set
 * @param images double pointer to all the image data
 * @param n pointer to the number of images
 */
void readMNISTImages(MNISTImage**, int*);

/**
 * Reads the MNIST labels data set
 * @param labels double pointer to all the label data
 * @param n pointer to the number of labels
 */
void readMNISTLabels(u_char**, int*);

/**
 * Reverses the byte ordering of a 32-bit integer
 * @param i the integer to reverse byte ordering of
 * @return the new byte-reversed integer
 */
uint32_t swapEndian(uint32_t);

/**
 * Outputs a report to the console showing the resulting k-means clusters
 * @param clusters the final clusters after convergence
 * @param labels the MNIST labels data
 */
void printClusters(
    const MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters&,
    const u_char*
);

/**
 * Generates an HTML file visualizing the k-means clustering results
 * @param clusters the final clusters after convergence
 * @param images the MNIST images data
 * @param filename the HTML file name
 */
void toHTML(
    const MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters&,
    const MNISTImage*,
    const std::string&
);

/**
 * Generates an HTML table cell for a given MNIST image
 * @param f an output file stream
 * @param image an MNIST image
 */
void htmlCell(std::ofstream&, const MNISTImage&);

/**
 * Generates a random hex background color (just for funzies)
 */
std::string htmlRandomBackground();

int main(void) {
    MNISTImage* images = nullptr;
    u_char* labels = nullptr;

    // set up k-means
    MNISTKMeans<K, MNISTImage::getNumPixels()> kMeans;

    // run k-means
    int images_n;
    int labels_n;
    readMNISTImages(&images, &images_n);
    readMNISTLabels(&labels, &labels_n);
    kMeans.fit(images, images_n);

    // get results
    MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters clusters = kMeans.getClusters();
    
    // report and visualize results
    printClusters(clusters, labels);
    std::string filename = "kmeans_mnist_seq.html";
    toHTML(clusters, images, filename);
    std::cout << "\nTry displaying visualization file, " << filename << ", in a web browser!\n\n";

    delete[] images;
    delete[] labels;
    return 0;
}

void readMNISTImages(MNISTImage** images, int* n) {
    std::ifstream file(MNIST_IMAGES_FILEPATH);
    if (file.is_open()) {
        uint32_t magicNumber = 0;
        uint32_t images_n = 0;
        uint32_t rows_n = 0;
        uint32_t cols_n = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        file.read((char*)&images_n, sizeof(images_n));
        file.read((char*)&rows_n, sizeof(rows_n));
        file.read((char*)&cols_n, sizeof(cols_n));

        // only needed if we don't already know these values
        magicNumber = swapEndian(magicNumber);
        images_n = swapEndian(images_n);
        rows_n = swapEndian(rows_n);
        cols_n = swapEndian(cols_n);

        MNISTImage* imagesData = new MNISTImage[IMAGE_LIMIT];
        for (int i = 0; i < IMAGE_LIMIT; i++) {
            std::array<u_char, MNISTImage::getNumPixels()> imageData;
            file.read(reinterpret_cast<char*>(imageData.data()), MNISTImage::getNumPixels());
            imagesData[i] = MNISTImage(imageData);
        }
        *images = imagesData;
        *n = IMAGE_LIMIT;
    }
}

void readMNISTLabels(u_char** labels, int* n) {
    std::ifstream file(MNIST_LABELS_FILEPATH);
    if (file.is_open()) {
        uint32_t magicNumber = 0;
        uint32_t labels_n = 0;

        file.read((char*)&magicNumber, sizeof(magicNumber));
        file.read((char*)&labels_n, sizeof(labels_n));

        // only needed if we don't already know these values
        magicNumber = swapEndian(magicNumber);
        labels_n = swapEndian(labels_n);

        u_char* labelsData = new u_char[IMAGE_LIMIT];
        for (int i = 0; i < IMAGE_LIMIT; i++)
            file.read((char*)&labelsData[i], 1);
        *labels = labelsData;
        *n = IMAGE_LIMIT;
    }
}

uint32_t swapEndian(uint32_t i) {
    uint32_t result = 0;
    result |= (i & 0x000000FF) << 24; // leftmost byte
    result |= (i & 0x0000FF00) << 8;  // left middle byte
    result |= (i & 0x00FF0000) >> 8;  // right middle byte
    result |= (i & 0xFF000000) >> 24; // rightmost byte
    return result;
}

void printClusters(
    const MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters& clusters,
    const u_char* labels
) {
    std::cout << "\nMNIST labels report: showing clusters...\n";
    for (size_t i = 0; i < clusters.size(); i++) {
        std::cout << "\ncluster #" << i + 1 << ":\n";
        for (int j: clusters[i].elements)
            std::cout << (int)labels[j] << " ";
        std::cout << std::endl;
    }
}

void toHTML(
    const MNISTKMeans<K, MNISTImage::getNumPixels()>::Clusters& clusters,
    const MNISTImage* images,
    const std::string& filename
) {
    std::ofstream f(filename);
    f << "<body style=\"background:#" << htmlRandomBackground() << ";\">";
    f << "<table><tbody><tr style=\"vertical-align:top;\">\n";
    for (const auto& cluster : clusters) {
        f << "\t<td><table><tbody>\n";
        htmlCell(f, cluster.centroid);
        for (const auto& i: cluster.elements)
            htmlCell(f, images[i]);
        f << "</tbody></table></td>\n";
    }
    f << "</tr></tbody></table></body>\n";
}

void htmlCell(std::ofstream& f, const MNISTImage& image) {
    f << "\t\t<tr><td><table style=\"border-collapse:collapse\"><tbody>\n";
    for (int row = 0; row < MNISTImage::getNumRows(); row++) {
        f << "\t\t\t<tr>\n";
        for (int col = 0; col < MNISTImage::getNumCols(); col++) {
            f << "\t\t\t\t<td style=\"background:#" << image.pixelToHex(row, col) << ";";
            f << "width:5px;height:5px;\"></td>\n";
        }
        f << "\t\t\t</tr>\n";
    }
    f << "\t\t</tbody></table></td></tr>\n";
}

std::string htmlRandomBackground() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> distrib(0,255);
    char buffer[7];
    snprintf(buffer, sizeof(buffer), "%.6x", distrib(rng) << 16 | distrib(rng) << 8 | distrib(rng));
    return {buffer};
}