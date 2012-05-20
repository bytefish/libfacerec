#ifndef __OPENCV_TEST_TEST_FUNS_HPP__
#define __OPENCV_TEST_TEST_FUNS_HPP__

#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

// This file contains helper function to make testing OpenCV algorithms easier.

namespace impl {

template<typename _Tp>
inline bool isEqual(InputArray src0, InputArray src1) {
    Mat _src0 = src0.getMat();
    Mat _src1 = src1.getMat();
    if(_src0.rows != _src1.rows)
        return false;
    if(_src0.cols != _src1.cols)
        return false;
    if(_src0.channels() != _src1.channels())
        return false;
    for(int i = 0; i < _src0.rows; i++) {
        for(int j = 0; j < _src0.cols; j++) {
            _Tp a = _src0.at<_Tp>(i,j);
            _Tp b = _src1.at<_Tp>(i,j);
            if(a != b)
                return false;
        }
    }
    return true;
}

template<typename _Tp>
inline bool isEqual(InputArray src0, InputArray src1, double eps) {
    Mat _src0 = src0.getMat();
    Mat _src1 = src1.getMat();
    if(_src0.rows != _src1.rows)
        return false;
    if(_src0.cols != _src1.cols)
        return false;
    if(_src0.channels() != _src1.channels())
        return false;
    for(int i = 0; i < _src0.rows; i++) {
        for(int j = 0; j < _src0.cols; j++) {
            _Tp a = _src0.at<_Tp>(i,j);
            _Tp b = _src1.at<_Tp>(i,j);
            if(std::abs(a - b) > eps) {
                return false;
            }
        }
    }
    return true;
}

}

template<typename _Tp>
inline Mat getMatrixAsType(InputArray src) {
    return Mat_<_Tp>(src.getMat());
}

inline bool isEqual(InputArray src0, InputArray src1, double eps = 1E-16) {
    Mat _src0 = src0.getMat();
    Mat _src1 = src1.getMat();
    switch (_src0.type()) {
    case CV_8SC1: return impl::isEqual<char>(_src0, _src1); break;
    case CV_8UC1: return impl::isEqual<unsigned char>(_src0, _src1); break;
    case CV_16SC1: return impl::isEqual<short>(_src0, _src1); break;
    case CV_16UC1: return impl::isEqual<unsigned short>(_src0, _src1); break;
    case CV_32SC1: return impl::isEqual<int>(_src0, _src1); break;
    case CV_32FC1: return impl::isEqual<float>(_src0, _src1, eps); break;
    case CV_64FC1: return impl::isEqual<double>(_src0, _src1, eps); break;
    default:break;
    }
    return false;
}


#endif
