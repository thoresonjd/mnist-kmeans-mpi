/**
 * @file MNISTImage.h - a class that encapsulates an MNIST image
 * @author Justin Thoreson
 * @see http://yann.lecun.com/exdb/mnist/
 */

#pragma once
#include <string>
#include <array>
#include <cmath>

/**
 * @class Wrapper class for an MNIST image
 */
class MNISTImage {
private:
    static const int ROWS_N = 28;
    static const int COLS_N = 28;
    static const int PIXELS_N = 784;
    using Pixels = std::array<u_char, PIXELS_N>;
    Pixels pixels;

public:
    /**
     * Constructors
     * @param pixels an array of pixels representing an MNIST image
     */
    MNISTImage() {}
    MNISTImage(const Pixels pixels);

    /**
     * Converts a given pixel value to a hexadecimal string label
     * @param row the row of the pixel 
     * @param col the column of the pixel
     * @return the hex label for a single pixel
     */
    std::string pixelToHex(int row, int col) const;

    /**
     * Calculates the euclidean distance between two MNIST images
     * @param other the MNIST image to measure the current against
     * @return the distance between the current MNIST image and the other 
     */
    double euclideanDistance(const MNISTImage& other) const;

    /**
     * Retrieves a pixel at any given row or column in the image
     * @param row the row of the pixel 
     * @param col the column of the pixel
     * @return the pixel value represented as an unsigned byte
     */
    u_char getPixel(int row, int col) const;

    /**
     * Static accessors for MNIST image dimensionality
     */
    constexpr static int getNumRows() { return ROWS_N; }
    constexpr static int getNumCols() { return COLS_N; }
    constexpr static int getNumPixels() { return PIXELS_N; }
};