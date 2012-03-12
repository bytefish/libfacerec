#ifndef __LBP_HPP__
#define __LBP_HPP__

#include "opencv2/opencv.hpp"

using namespace cv;

// TODO Quantization of Variance-based LBP.
// TODO Add Literature References.
// TODO Add Tests.

namespace cv {
namespace impl {

template <typename _Tp>
inline void olbp(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) >= center) << 7;
            code |= (src.at<_Tp>(i-1,j) >= center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) >= center) << 5;
            code |= (src.at<_Tp>(i,j+1) >= center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) >= center) << 3;
            code |= (src.at<_Tp>(i+1,j) >= center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) >= center) << 1;
            code |= (src.at<_Tp>(i,j-1) >= center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

template <typename _Tp>
inline void elbp(const Mat& src, Mat& dst, int radius, int neighbors) {
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                // calculate interpolated value
                float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}


template <typename _Tp>
inline void varlbp(const Mat& src, Mat& dst, int radius, int neighbors) {
    dst = Mat::zeros(src.rows-2*radius, src.cols-2*radius, CV_32FC1); //! result
    // allocate some memory for temporary on-line variance calculations
    Mat _mean = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat _delta = Mat::zeros(src.rows, src.cols, CV_32FC1);
    Mat _m2 = Mat::zeros(src.rows, src.cols, CV_32FC1);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius) * cos(2.0*M_PI*n/static_cast<float>(neighbors));
        float y = static_cast<float>(radius) * -sin(2.0*M_PI*n/static_cast<float>(neighbors));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
                _delta.at<float>(i,j) = t - _mean.at<float>(i,j);
                _mean.at<float>(i,j) = (_mean.at<float>(i,j) + (_delta.at<float>(i,j) / (1.0*(n+1)))); // i am a bit paranoid
                _m2.at<float>(i,j) = _m2.at<float>(i,j) + _delta.at<float>(i,j) * (t - _mean.at<float>(i,j));
            }
        }
    }
    // calculate result
    for(int i = radius; i < src.rows-radius; i++) {
        for(int j = radius; j < src.cols-radius; j++) {
            dst.at<float>(i-radius, j-radius) = _m2.at<float>(i,j) / (1.0*(neighbors-1));
        }
    }
}

} // namespace impl

// Calculates the Original Local Binary Patterns.
inline void olbp(const Mat& src, Mat& dst) {
    switch (src.type()) {
    case CV_8SC1:   impl::olbp<char>(src,dst); break;
    case CV_8UC1:   impl::olbp<unsigned char>(src,dst); break;
    case CV_16SC1:  impl::olbp<short>(src,dst); break;
    case CV_16UC1:  impl::olbp<unsigned short>(src,dst); break;
    case CV_32SC1:  impl::olbp<int>(src,dst); break;
    case CV_32FC1:  impl::olbp<float>(src,dst); break;
    case CV_64FC1:  impl::olbp<double>(src,dst); break;
    default: break;
    }
}

// Calculates the Extended Local Binary Patterns.
inline void elbp(const Mat& src, Mat& dst, int radius=1, int neighbors=8) {
    switch (src.type()) {
    case CV_8SC1:   impl::elbp<char>(src,dst, radius, neighbors); break;
    case CV_8UC1:   impl::elbp<unsigned char>(src, dst, radius, neighbors); break;
    case CV_16SC1:  impl::elbp<short>(src,dst, radius, neighbors); break;
    case CV_16UC1:  impl::elbp<unsigned short>(src,dst, radius, neighbors); break;
    case CV_32SC1:  impl::elbp<int>(src,dst, radius, neighbors); break;
    case CV_32FC1:  impl::elbp<float>(src,dst, radius, neighbors); break;
    case CV_64FC1:  impl::elbp<double>(src,dst, radius, neighbors); break;
    default: break;
    }
}

// Calculates the Variance-based Local Binary Patterns (without Quantization).
inline void varlbp(const Mat& src, Mat& dst, int radius=1, int neighbors=8) {
    switch (src.type()) {
    case CV_8SC1:   impl::varlbp<char>(src,dst, radius, neighbors); break;
    case CV_8UC1:   impl::varlbp<unsigned char>(src,dst, radius, neighbors); break;
    case CV_16SC1:  impl::varlbp<short>(src,dst, radius, neighbors); break;
    case CV_16UC1:  impl::varlbp<unsigned short>(src,dst, radius, neighbors); break;
    case CV_32SC1:  impl::varlbp<int>(src,dst, radius, neighbors); break;
    case CV_32FC1:  impl::varlbp<float>(src,dst, radius, neighbors); break;
    case CV_64FC1:  impl::varlbp<double>(src,dst, radius, neighbors); break;
    default: break;
    }
}

// Wrapper functions for convenience.
Mat olbp(const Mat& src) {
    Mat dst;
    olbp(src, dst);
    return dst;
}

Mat elbp(const Mat& src, int radius=1, int neighbors=8) {
    Mat dst;
    elbp(src, dst, radius, neighbors);
    return dst;
}

Mat varlbp(const Mat& src, int radius=1, int neighbors=8) {
    Mat dst;
    varlbp(src, dst, radius, neighbors);
    return dst;
}

}



#endif
