/**
 * @file MNISTKMeans.h - a subclass of KMeans to cluster MNIST images
 * @author Justin Thoreson
 * @see http://yann.lecun.com/exdb/mnist/
 */

#pragma once
#include "KMeans.h"
#include "MNISTImage.h"

/**
 * @class Concrete MNIST k-means class 
 * @tparam k the number of clusters for k-means
 * @tparam d the dimensionality of an MNIST image
 */
template<int k, int d>
class MNISTKMeans : public KMeans<k, d> {
public:
    /**
     * Run k-means clustering on MNIST images
     * @param images pointer to the MNIST image data
     * @param n the number of images  
     */
    void fit(MNISTImage* images, int n) {
        KMeans<k, d>::fit(reinterpret_cast<std::array<u_char, d>*>(images), n);
    }

protected:
    using Element = std::array<u_char, d>;
    
    /**
     * Overload distance method with euclidean distance for MNIST images
     * @param a one MNIST image
     * @param b another MNIST image
     * @return distance between a and b
     */
    double distance(const Element& a, const Element& b) const {
        return MNISTImage(a).euclideanDistance(MNISTImage(b));
    }
};