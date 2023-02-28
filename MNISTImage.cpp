/**
 * @file MNISTImage.cpp - implementation of MNISTImage class methods
 * @author Justin Thoreson
 * @see http://yann.lecun.com/exdb/mnist/
 */

#include "MNISTImage.h"

MNISTImage::MNISTImage(const Pixels pixels) : pixels(pixels) {}

std::string MNISTImage::pixelToHex(int row, int col) const {
    u_char pixel = pixels[ROWS_N * row + col];
    char buffer[7];
    snprintf(buffer, sizeof(buffer), "%.6x", pixel << 16 | pixel << 8 | pixel);
    return {buffer};
}

double MNISTImage::euclideanDistance(const MNISTImage& other) const {
    double sum = 0;
    for (int i = 0; i < PIXELS_N; i++) {
        double difference = (double)pixels[i] - (double)other.pixels[i];
        sum += difference * difference;
    }
    return sqrt(sum);
}

u_char MNISTImage::getPixel(int row, int col) const {
    return pixels[ROWS_N * row + col];
}
