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

#include "facerec.hpp"
#include "subspace.hpp"
#include "helper.hpp"
#include <limits>
#include <cmath>

using namespace util;

void Fisherfaces::train(const vector<Mat>& src, const vector<int>& labels) {
	assert(src.size() == labels.size());
	Mat data = asRowMatrix(src,CV_64FC1);
	int N = data.rows; // number of samples
	int D = data.cols; // dimension of samples
	// compute the Fisherfaces
	int C = remove_dups(labels).size(); // number of unique classes
	// clip number of components to be a valid number
	if((_num_components <= 0) || (_num_components > (C-1)))
		_num_components = (C-1);
	// perform a PCA and keep (N-C) components
	PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (N-C));
	// project the data and perform a LDA on it
	subspace::LinearDiscriminantAnalysis lda(pca.project(data),labels, _num_components);
	// store the total mean vector
	_mean = pca.mean.reshape(1,1);
	// store labels
	_labels = labels;
	// store the eigenvalues of the discriminants
	lda.eigenvalues().convertTo(_eigenvalues, CV_64FC1);
	// Now calculate the projection matrix as pca.eigenvectors * lda.eigenvectors.
	// Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
	gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, CV_GEMM_A_T);
	// store the projections of the original data
	for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
		Mat p = subspace::project(_eigenvectors, _mean, data.row(sampleIdx));
		_projections.push_back(p);
	}
}

int Fisherfaces::predict(const Mat& src) {
	Mat q = subspace::project(_eigenvectors, _mean, src.reshape(1,1));
	// find 1-nearest neighbor
	double minDist = numeric_limits<double>::max();
	int minClass = -1;
	for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
		double dist = norm(_projections[sampleIdx], q, NORM_L2);
		if(dist < minDist) {
			minDist = dist;
			minClass = _labels[sampleIdx];
		}
	}
	return minClass;
}
