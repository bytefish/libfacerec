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

#ifndef __SUBSPACE_HPP__
#define __SUBSPACE_HPP__

#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

namespace cv { namespace subspace {

// Calculates the projection Y = (X - mean) * W.
Mat project(InputArray W, InputArray mean, InputArray X);

// Calculates the reconstruction X = Y*W + mean.
Mat reconstruct(InputArray W, InputArray mean, InputArray Y);

// This class performs a Linear Discriminant Analysis with the classic
// Fisher's Optimization Criterion.
//
// TODO Use InputArray instead of Mat and vector<Mat> for data input.
class LDA {

private:
    bool _dataAsRow;
    int _num_components;
    Mat _eigenvectors;
    Mat _eigenvalues;

    void lda(InputArray src, InputArray labels);

public:

    // Initializes a LDA with num_components (default 0) and specifies how
    // samples are aligned (default dataAsRow=true).
    LDA(int num_components = 0) :
        _num_components(num_components) {};

    // Initializes and performs a Discriminant Analysis with Fisher's
    // Optimization Criterion on given data in src and corresponding labels
    // in labels. If 0 (or less) number of components are given, they are
    // automatically determined for given data in computation.
    LDA(const Mat& src, vector<int> labels,
            int num_components = 0) :
                _num_components(num_components)
    {
        this->compute(src, labels); //! compute eigenvectors and eigenvalues
    }

    // Initializes and performs a Discriminant Analysis with Fisher's
    // Optimization Criterion on given data in src and corresponding labels
    // in labels. If 0 (or less) number of components are given, they are
    // automatically determined for given data in computation.
    LDA(InputArray src, InputArray labels,
            int num_components = 0) :
                _num_components(num_components)
    {
        this->compute(src, labels); //! compute eigenvectors and eigenvalues
    }

    // Serializes this object to a given filename.
    void save(const string& filename) const;

    // Deserializes this object from a given filename.
    void load(const string& filename);

    // Serializes this object to a given cv::FileStorage.
    void save(FileStorage& fs) const;

        // Deserializes this object from a given cv::FileStorage.
    void load(const FileStorage& node);

    // Destructor.
    ~LDA() {}

    //! Compute the discriminants for data in src and labels.
    void compute(InputArray src, InputArray labels);

    // Projects samples into the LDA subspace.
    Mat project(InputArray src);

    // Reconstructs projections from the LDA subspace.
    Mat reconstruct(InputArray src);

    // Returns the eigenvectors of this LDA.
    Mat eigenvectors() const { return _eigenvectors; };

    // Returns the eigenvalues of this LDA.
    Mat eigenvalues() const { return _eigenvalues; }
};

}} // namespace
#endif
