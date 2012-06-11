/*
 * Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include <opencv2/opencv.hpp>
#include "helper.hpp"

using namespace cv;

//------------------------------------------------------------------------------
// cv::isSymmetric
//------------------------------------------------------------------------------
namespace cv {

template<typename _Tp> static bool
isSymmetric_(InputArray src) {
    Mat _src = src.getMat();
    if(_src.cols != _src.rows)
        return false;
    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            _Tp a = _src.at<_Tp> (i, j);
            _Tp b = _src.at<_Tp> (j, i);
            if (a != b) {
                return false;
            }
        }
    }
    return true;
}

template<typename _Tp> static bool
isSymmetric_(InputArray src, double eps) {
    Mat _src = src.getMat();
    if(_src.cols != _src.rows)
        return false;
    for (int i = 0; i < _src.rows; i++) {
        for (int j = 0; j < _src.cols; j++) {
            _Tp a = _src.at<_Tp> (i, j);
            _Tp b = _src.at<_Tp> (j, i);
            if (std::abs(a - b) > eps) {
                return false;
            }
        }
    }
    return true;
}

}

bool cv::isSymmetric(InputArray src, double eps) {
    Mat m = src.getMat();
    switch (m.type()) {
    case CV_8SC1: return isSymmetric_<char>(m); break;
    case CV_8UC1:
        return isSymmetric_<unsigned char>(m); break;
    case CV_16SC1:
        return isSymmetric_<short>(m); break;
    case CV_16UC1:
        return isSymmetric_<unsigned short>(m); break;
    case CV_32SC1:
        return isSymmetric_<int>(m); break;
    case CV_32FC1:
        return isSymmetric_<float>(m, eps); break;
    case CV_64FC1:
        return isSymmetric_<double>(m, eps); break;
    default:
        break;
    }
    return false;
}

//------------------------------------------------------------------------------
// cv::argsort
//------------------------------------------------------------------------------
Mat cv::argsort(InputArray _src, bool ascending) {
    Mat src = _src.getMat();
    if (src.rows != 1 && src.cols != 1) {
        CV_Error(CV_StsBadArg, "cv::argsort only sorts 1D matrices.");
    }
    int flags = CV_SORT_EVERY_ROW+(ascending ? CV_SORT_ASCENDING : CV_SORT_DESCENDING);
    Mat sorted_indices;
    cv::sortIdx(src.reshape(1,1),sorted_indices,flags);
    return sorted_indices;
}
//------------------------------------------------------------------------------
// cv::histc
//------------------------------------------------------------------------------
namespace cv {

static Mat
histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false) {
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal-minVal+1;
    // Set the ranges.
    float range[] = { minVal, maxVal+1 } ;
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if(normed) {
        result /= src.total();
    }
    return result.reshape(1,1);
}

}

Mat cv::histc(InputArray _src, int minVal, int maxVal, bool normed) {
    Mat src = _src.getMat();
    switch (src.type()) {
    case CV_8SC1:
        return histc_(Mat_<float>(src), minVal, maxVal, normed);
        break;
    case CV_8UC1:
        return histc_(src, minVal, maxVal, normed);
        break;
    case CV_16SC1:
        return histc_(Mat_<float>(src), minVal, maxVal, normed);
        break;
    case CV_16UC1:
        return histc_(src, minVal, maxVal, normed);
        break;
    case CV_32SC1:
        return histc_(Mat_<float>(src), minVal, maxVal, normed);
        break;
    case CV_32FC1:
        return histc_(src, minVal, maxVal, normed);
        break;
    default:
        CV_Error(CV_StsUnmatchedFormats, "This type is not implemented yet."); break;
    }
    return Mat();
}

//------------------------------------------------------------------------------
// cv::sortMatrixColumnsByIndices
//------------------------------------------------------------------------------

void cv::sortMatrixColumnsByIndices(InputArray _src, InputArray _indices, OutputArray _dst) {
    if(_indices.getMat().type() != CV_32SC1) {
    	string error_message = format("cv::sortRowsByIndices only works on integer indices! Expected: %d. Given: %d.", CV_32SC1, _indices.getMat().type());
    	CV_Error(CV_StsBadArg, error_message);
    }
    Mat src = _src.getMat();
    vector<int> indices = _indices.getMat();
    _dst.create(src.rows, src.cols, src.type());
    Mat dst = _dst.getMat();
    for(int idx = 0; idx < indices.size(); idx++) {
        Mat originalCol = src.col(indices[idx]);
        Mat sortedCol = dst.col(idx);
        originalCol.copyTo(sortedCol);
    }
}

Mat cv::sortMatrixColumnsByIndices(InputArray src, InputArray indices) {
    Mat dst;
    sortMatrixColumnsByIndices(src, indices, dst);
    return dst;
}

//------------------------------------------------------------------------------
// cv::sortMatrixRowsByIndices
//------------------------------------------------------------------------------
void cv::sortMatrixRowsByIndices(InputArray _src, InputArray _indices, OutputArray _dst) {
    if(_indices.getMat().type() != CV_32SC1) {
    	string error_message = format("cv::sortRowsByIndices only works on integer indices! Expected: %d. Given: %d.", CV_32SC1, _indices.getMat().type());
    	CV_Error(CV_StsBadArg, error_message);
    }
    Mat src = _src.getMat();
    vector<int> indices = _indices.getMat();
    _dst.create(src.rows, src.cols, src.type());
    Mat dst = _dst.getMat();
    for(int idx = 0; idx < indices.size(); idx++) {
        Mat originalRow = src.row(indices[idx]);
        Mat sortedRow = dst.row(idx);
        originalRow.copyTo(sortedRow);
    }
}

Mat cv::sortMatrixRowsByIndices(InputArray src, InputArray indices) {
   Mat dst;
   sortMatrixRowsByIndices(src, indices, dst);
   return dst;
}

//------------------------------------------------------------------------------
// cv::asRowMatrix
//------------------------------------------------------------------------------
Mat cv::asRowMatrix(InputArrayOfArrays src, int rtype, double alpha, double beta) {
    // make sure the input data is a vector of matrices or vector of vector
    if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        CV_Error(CV_StsBadArg, "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).");
    }
    // number of samples
    size_t n = src.total();
    // return empty matrix if no matrices given
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src.getMat(0).total();
    // create data matrix
    Mat data(n, d, rtype);
    // now copy data
    for(int i = 0; i < n; i++) {
        // make sure data can be reshaped, throw exception if not!
        if(src.getMat(i).total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src.getMat(i).total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // get a hold of the current row
        Mat xi = data.row(i);
        // make reshape happy by cloning for non-continuous matrices
        if(src.getMat(i).isContinuous()) {
            src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

//------------------------------------------------------------------------------
// cv::asColumnMatrix
//------------------------------------------------------------------------------
Mat cv::asColumnMatrix(InputArrayOfArrays src, int rtype, double alpha, double beta) {
    // make sure the input data is a vector of matrices or vector of vector
    if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        CV_Error(CV_StsBadArg, "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).");
    }
    int n = (int) src.total();
    // return empty matrix if no data given
    if(n == 0)
        return Mat();
    // dimensionality of samples
    int d = src.getMat(0).total();
    // create data matrix
    Mat data(d, n, rtype);
    // now copy data
    for(int i = 0; i < n; i++) {
        // make sure data can be reshaped, throw exception if not!
        if(src.getMat(i).total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src.getMat(i).total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // get a hold of the current row
        Mat yi = data.col(i);
        // make reshape happy by cloning for non-continuous matrices
        if(src.getMat(i).isContinuous()) {
            src.getMat(i).reshape(1, d).convertTo(yi, rtype, alpha, beta);
        } else {
            src.getMat(i).clone().reshape(1, d).convertTo(yi, rtype, alpha, beta);
        }
    }
    return data;
}

//------------------------------------------------------------------------------
// cv::interp1
//------------------------------------------------------------------------------
namespace cv {

template <typename _Tp> static
Mat interp1_(const Mat& X_, const Mat& Y_, const Mat& XI) {
    int n = XI.rows;
    // sort input table
    vector<int> sort_indices = argsort(X_);

    Mat X = sortMatrixRowsByIndices(X_,sort_indices);
    Mat Y = sortMatrixRowsByIndices(Y_,sort_indices);
    // interpolated values
    Mat yi = Mat::zeros(XI.size(), XI.type());
    for(int i = 0; i < n; i++) {
        int c = 0;
        int low = 0;
        int high = X.rows - 1;
        // set bounds
        if(XI.at<_Tp>(i,0) < X.at<_Tp>(low, 0))
            high = 1;
        if(XI.at<_Tp>(i,0) > X.at<_Tp>(high, 0))
            low = high - 1;
        // binary search
        while((high-low)>1) {
            c = low + ((high - low) >> 1);
            if(XI.at<_Tp>(i,0) > X.at<_Tp>(c,0)) {
                low = c;
            } else {
                high = c;
            }
        }
        // linear interpolation
        yi.at<_Tp>(i,0) += Y.at<_Tp>(low,0)
                + (XI.at<_Tp>(i,0) - X.at<_Tp>(low,0))
                * (Y.at<_Tp>(high,0) - Y.at<_Tp>(low,0))
                    / (X.at<_Tp>(high,0) - X.at<_Tp>(low,0));
    }
    return yi;
}

}

Mat cv::interp1(InputArray _x, InputArray _Y, InputArray _xi) {
    // get matrices
    Mat x = _x.getMat();
    Mat Y = _Y.getMat();
    Mat xi = _xi.getMat();
    // check type
    if((x.type() != Y.type()) || (Y.type() != xi.type())) {
        string error_message = format("Interpolation matrices must have the same type (was %d, %d, %d).", x.type(), Y.type(), xi.type());
        CV_Error(CV_StsBadArg, error_message);
    }
    // check alignment
    if((x.cols != 1) || (x.rows != Y.rows) || (x.cols != Y.cols)) {
        string error_message = format("Wrong alignment of given interpolation matrices, was x.rows=%d, x.cols=%d, Y.rows=%d, Y.cols=%d",x.rows, x.cols, Y.rows, Y.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    // call templated interp1
    switch(x.type()) {
        case CV_8SC1: return interp1_<char>(x,Y,xi); break;
        case CV_8UC1: return interp1_<unsigned char>(x,Y,xi); break;
        case CV_16SC1: return interp1_<short>(x,Y,xi); break;
        case CV_16UC1: return interp1_<unsigned short>(x,Y,xi); break;
        case CV_32SC1: return interp1_<int>(x,Y,xi); break;
        case CV_32FC1: return interp1_<float>(x,Y,xi); break;
        case CV_64FC1: return interp1_<double>(x,Y,xi); break;
    }
    return Mat();
}

//------------------------------------------------------------------------------
// cv::linspace
//------------------------------------------------------------------------------
Mat cv::linspace(float x0, float x1, int n) {
    Mat pts(n, 1, CV_32FC1);
    float step = (x1-x0)/static_cast<float>(n-1);
    for(int i = 0; i < n; i++)
        pts.at<float>(i,0) = x0+i*step;
    return pts;
}

//------------------------------------------------------------------------------
// cv::toGrayscale
//------------------------------------------------------------------------------
Mat cv::toGrayscale(InputArray _src, int dtype) {
    Mat src = _src.getMat();
    // only allow one channel
    if(src.channels() != 1) {
        string error_message = format("Only Matrices with one channel are supported. Expected 1, but was %d.", src.channels());
        CV_Error(CV_StsBadArg, error_message);
    }
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

//------------------------------------------------------------------------------
// cv::transpose
//------------------------------------------------------------------------------
Mat cv::transpose(InputArray _src) {
    Mat src = _src.getMat();
    Mat dst;
    transpose(src, dst);
    return dst;
}
