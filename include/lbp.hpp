#ifndef __LBP_HPP__
#define __LBP_HPP__

#include "opencv2/core/core.hpp"

using namespace cv;

namespace cv {
// TODO Add Uniform Patterns (or other histogram dimensionality reduction)
// TODO Test & Quantization of Variance-based LBP.

// Calculates the Original Local Binary Patterns.
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
void olbp(InputArray src, OutputArray dst);

// Calculates the Extended Local Binary Patterns.
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
void elbp(InputArray src, OutputArray dst, int radius=1, int neighbors=8);

// Calculates the Variance-based Local Binary Patterns (without Quantization).
//
//  Pietikäinen, M., Hadid, A., Zhao, G. and Ahonen, T. (2011), "Computer
//  Vision Using Local Binary Patterns", Springer.
//
void varlbp(InputArray src, OutputArray dst, int radius=1, int neighbors=8);

// Calculates the Spatial Histogram for a given LBP image.
//
// TODO Test, Test, Test!
// TODO Optimize, Optimize, Optimize!
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
Mat spatial_histogram(InputArray src, int numPatterns, int grid_x=8, int grid_y=8, bool normed=true);

// see cv::olbp(InputArray, OutputArray)
Mat olbp(InputArray src);

// see cv::elbp(InputArray, OutputArray, int, int)
Mat elbp(InputArray src, int radius=1, int neighbors=8);

// see cv::varlbp(InputArray, OutputArray, int, int)
Mat varlbp(InputArray src, int radius=1, int neighbors=8);

}

#endif
