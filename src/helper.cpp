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

#include "helper.hpp"
#include <iostream>
#include <set>

using namespace cv;

void cv::sortMatrixByColumn(const Mat& src, Mat& dst, vector<int> indices) {
	dst.create(src.rows, src.cols, src.type());
	for(int idx = 0; idx < indices.size(); idx++) {
		Mat originalCol = src.col(indices[idx]);
		Mat sortedCol = dst.col(idx);
		originalCol.copyTo(sortedCol);
	}
}

Mat cv::sortMatrixByColumn(const Mat& src, vector<int> indices) {
	Mat dst;
	sortMatrixByColumn(src, dst, indices);
	return dst;
}

void cv::sortMatrixByRow(const Mat& src, Mat& dst, vector<int> indices) {
	dst.create(src.rows, src.cols, src.type());
	for(int idx = 0; idx < indices.size(); idx++) {
		Mat originalRow = src.row(indices[idx]);
		Mat sortedRow = dst.row(idx);
		originalRow.copyTo(sortedRow);
	}
}

Mat cv::sortMatrixByRow(const Mat& src, vector<int> indices) {
	Mat dst;
	sortMatrixByRow(src, dst, indices);
	return dst;
}

Mat cv::asColumnMatrix(const vector<Mat>& src, int dtype) {
	int n = src.size();
	// return empty matrix if no data given
	if(n == 0)
		return Mat();
	// dimensionality of samples
	int d = src[0].total();
	// create data matrix
	Mat data(d, n, dtype);
	// copy data
	for(int i = 0; i < src.size(); i++) {
		Mat yi = data.col(i);
		src[i].reshape(1, d).convertTo(yi, dtype);
	}
	return data;
}

Mat cv::asRowMatrix(const vector<Mat>& src, int dtype) {
	// number of samples
	int n = src.size();
	// return empty matrix if no data given
	if(n == 0)
		return Mat();
	// dimensionality of samples
	int d = src[0].total();
	// create data matrix
	Mat data(n, d, dtype);
	// copy data
	for(int i = 0; i < src.size(); i++) {
		Mat xi = data.row(i);
		src[i].reshape(1, 1).convertTo(xi, dtype);
	}
	return data;
}

Mat cv::transpose(const Mat& src) {
		Mat dst;
		transpose(src, dst);
		return dst;
}

Mat cv::toGrayscale(InputArray _src, int dtype) {
	Mat src = _src.getMat();
	// only allow one channel
	if(src.channels() != 1)
		CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
	// create and return normalized image
	Mat dst;
	cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	return dst;
}

string cv::num2str(int i) {
	stringstream ss;
	ss << i;
	return ss.str();
}
