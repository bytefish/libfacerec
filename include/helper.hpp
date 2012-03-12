/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
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

#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include "opencv2/opencv.hpp"
#include <vector>
#include <set>
#ifdef HAVE_EIGEN
#include <eigen3/Eigen/Dense>
#endif

using namespace std;

// Removes duplicate elements in a given vector.
template<typename _Tp>
inline vector<_Tp> remove_dups(const vector<_Tp>& src) {
    typedef typename set<_Tp>::const_iterator constSetIterator;
    typedef typename vector<_Tp>::const_iterator constVecIterator;
    set<_Tp> set_elems;
    for (constVecIterator it = src.begin(); it != src.end(); ++it)
        set_elems.insert(*it);
    vector<_Tp> elems;
    for (constSetIterator it = set_elems.begin(); it != set_elems.end(); ++it)
        elems.push_back(*it);
    return elems;
}

// The namespace cv provides opencv related helper functions.
namespace cv {

// The namespace impl provides opencv related helper functions.
// This interface is not guaranteed to be stable, the convention
// is to not program against namespace impl functions!
namespace impl {

template<typename _Tp>
inline bool isSymmetric(InputArray src) {
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

template<typename _Tp>
inline bool isSymmetric(InputArray src, double eps) {
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


//! ascending sort operator
template<typename _Tp>
class SortByFirstAscending_ {
public:
    bool operator()(const std::pair<_Tp, int>& left,
            const std::pair<_Tp, int>& right) {
        return left.first < right.first;
    }
};

//! descending sort operator
template<typename _Tp>
class SortByFirstDescending_ {
public:
    bool operator()(const std::pair<_Tp, int>& left,
            const std::pair<_Tp, int>& right) {
        return left.first > right.first;
    }
};

template<typename _Tp>
vector<int> argsort_(const Mat& src, bool asc) {
    if (src.rows != 1 && src.cols != 1)
        CV_Error(CV_StsBadArg, "Argsort only sorts 1D Vectors");
    vector<pair<_Tp, int> > val_indices;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            val_indices.push_back(make_pair(src.at<_Tp> (i, j), val_indices.size()));
        }
    }

    if (asc) {
        std::sort(val_indices.begin(), val_indices.end(), SortByFirstAscending_<_Tp> ());
    } else {
        std::sort(val_indices.begin(), val_indices.end(), SortByFirstDescending_<_Tp> ());
    }

    vector<int> indices;
    for (int i = 0; i < val_indices.size(); i++)
        indices.push_back(val_indices[i].second);
    return indices;
}

}

// Checks if a given matrix is symmetric.
inline bool isSymmetric(InputArray src, double eps = 1E-16) {
    Mat m = src.getMat();
    switch (m.type()) {
    case CV_8SC1: return impl::isSymmetric<char>(m);
        break;
    case CV_8UC1:
        return impl::isSymmetric<unsigned char>(m);
        break;
    case CV_16SC1:
        return impl::isSymmetric<short>(m);
        break;
    case CV_16UC1:
        return impl::isSymmetric<unsigned short>(m);
        break;
    case CV_32SC1:
        return impl::isSymmetric<int>(m);
        break;
    case CV_32FC1:
        return impl::isSymmetric<float>(m, eps);
        break;
    case CV_64FC1:
        return impl::isSymmetric<double>(m, eps);
        break;
    default:
        break;
    }
    return false;
}

// Sorts a 1D Matrix by given sort order and returns the sorted indices.
inline vector<int> argsort(const Mat& src, bool sortAscending = true) {
    switch (src.type()) {
    case CV_8SC1:
        return impl::argsort_<char>(src, sortAscending);
        break;
    case CV_8UC1:
        return impl::argsort_<unsigned char>(src, sortAscending);
        break;
    case CV_16SC1:
        return impl::argsort_<short>(src, sortAscending);
        break;
    case CV_16UC1:
        return impl::argsort_<unsigned short>(src, sortAscending);
        break;
    case CV_32SC1:
        return impl::argsort_<int>(src, sortAscending);
        break;
    case CV_32FC1:
        return impl::argsort_<float>(src, sortAscending);
        break;
    case CV_64FC1:
        return impl::argsort_<double>(src, sortAscending);
        break;
    }
}

// Reads a sequence from a FileNode::SEQ with type _Tp into a result vector.
template<typename _Tp>
inline void readFileNodeList(const FileNode& fn, vector<_Tp>& result) {
    if (fn.type() == FileNode::SEQ) {
        for (FileNodeIterator it = fn.begin(); it != fn.end();) {
            _Tp item;
            it >> item;
            result.push_back(item);
        }
    }
}

// Writes the a list of given items to a cv::FileStorage.
template<typename _Tp>
inline void writeFileNodeList(FileStorage& fs, const string& name,
        const vector<_Tp>& items) {
    // typedefs
    typedef typename vector<_Tp>::const_iterator constVecIterator;
    // write the elements in item to fs
    fs << name << "[";
    for (constVecIterator it = items.begin(); it != items.end(); ++it) {
        cout << "wrote:" << *it << endl;
        fs << *it;
    }
    fs << "]";
}


// Sorts a given matrix src by column for given indices.
//
// Note: create is called on dst.
inline void sortMatrixByColumn(const Mat& src, Mat& dst, vector<int> indices) {
    dst.create(src.rows, src.cols, src.type());
    for(int idx = 0; idx < indices.size(); idx++) {
        Mat originalCol = src.col(indices[idx]);
        Mat sortedCol = dst.col(idx);
        originalCol.copyTo(sortedCol);
    }
}


// Sorts a given matrix src by row for given indices.
inline Mat sortMatrixByColumn(const Mat& src, vector<int> indices) {
    Mat dst;
    sortMatrixByColumn(src, dst, indices);
    return dst;
}

// Sorts a given matrix src by row for given indices.
//
// Note: create is called on dst.
inline void sortMatrixByRow(const Mat& src, Mat& dst, vector<int> indices) {
    dst.create(src.rows, src.cols, src.type());
    for(int idx = 0; idx < indices.size(); idx++) {
        Mat originalRow = src.row(indices[idx]);
        Mat sortedRow = dst.row(idx);
        originalRow.copyTo(sortedRow);
    }
}

// Sorts a given matrix src by row for given indices.
inline Mat sortMatrixByRow(const Mat& src, vector<int> indices) {
    Mat dst;
    sortMatrixByRow(src, dst, indices);
    return dst;
}

// Turns a vector of matrices into a row matrix.
inline Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha=1, double beta=0) {
    // number of samples
    int n = src.size();
    // return empty matrix if no data given
    if(n == 0)
        return Mat();
    // dimensionality of samples
    int d = src[0].total();
    // create data matrix
    Mat data(n, d, rtype);
    // copy data
    for(int i = 0; i < src.size(); i++) {
        Mat xi = data.row(i);
        src[i].reshape(1, 1).convertTo(xi, rtype);
    }
    return data;
}

// Turns a vector of matrices into a column matrix.
inline Mat asColumnMatrix(const vector<Mat>& src, int rtype, double alpha=1, double beta=0) {
    int n = src.size();
    // return empty matrix if no data given
    if(n == 0)
        return Mat();
    // dimensionality of samples
    int d = src[0].total();
    // create data matrix
    Mat data(d, n, rtype);
    // copy data
    for(int i = 0; i < src.size(); i++) {
        Mat yi = data.col(i);
        src[i].reshape(1, d).convertTo(yi, rtype);
    }
    return data;
}

// Turns a given matrix into its grayscale representation.
inline Mat toGrayscale(InputArray src, int dtype = CV_8UC1) {
    Mat _src = src.getMat();
    // only allow one channel
    if(_src.channels() != 1)
        CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
    // create and return normalized image
    Mat dst;
    cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// Transposes a matrix.
inline Mat transpose(const Mat& src) {
    Mat dst;
    transpose(src, dst);
    return dst;
}

// Converts an integer number to a string.
//
// Equivalent to GNU Octave/MATLAB function "num2str".
inline string num2str(int num) {
    stringstream ss;
    ss << num;
    return ss.str();
}

#ifdef HAVE_EIGEN
template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst )
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), DataType<_Tp>::type,
                (void*)src.data(), src.stride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), DataType<_Tp>::type,
                (void*)src.data(), src.stride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void cv2eigen( const Mat& src,
        Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& dst )
{
    CV_DbgAssert(src.rows == _rows && src.cols == _cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
        transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
        Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
    else
    {
        Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}

template<typename _Tp>
void cv2eigen( const Mat& src,
        Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
        transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
        Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
    else
    {
        Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}

template<typename _Tp>
void cv2eigen( const Mat& src,
        Eigen::Matrix<_Tp, Eigen::Dynamic, 1>& dst )
{
    CV_Assert(src.cols == 1);
    dst.resize(src.rows);

    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
        transpose(src, _dst);
        else
        Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
    else
    {
        Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}

template<typename _Tp>
void cv2eigen( const Mat& src,
        Eigen::Matrix<_Tp, 1, Eigen::Dynamic>& dst )
{
    CV_Assert(src.rows == 1);
    dst.resize(src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
        transpose(src, _dst);
        else
        Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
    else
    {
        Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}
#endif
}

#endif
