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
#include "facerec.hpp"
#include "helper.hpp"
#include "decomposition.hpp"

#include "opencv2/imgproc/imgproc.hpp"

//------------------------------------------------------------------------------
// cv::FaceRecognizer
//------------------------------------------------------------------------------
void cv::FaceRecognizer::save(const string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
        CV_Error(CV_StsError, "File can't be opened for writing!");
    this->save(fs);
    fs.release();
}

void cv::FaceRecognizer::load(const string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        CV_Error(CV_StsError, "File can't be opened for writing!");
    this->load(fs);
    fs.release();
}


//------------------------------------------------------------------------------
// cv::Eigenfaces
//------------------------------------------------------------------------------
void cv::Eigenfaces::train(InputArray src, InputArray _lbls) {
    if(src.total() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(CV_StsUnsupportedFormat, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(CV_StsUnsupportedFormat, error_message);
    }
    // make sure data has correct size
    if(src.total() > 1) {
        for(int i = 1; i < src.total(); i++) {
            if(src.getMat(i-1).total() != src.getMat(i).total()) {
                string error_message = format("In the Eigenfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", src.getMat(i-1).total(), src.getMat(i).total());
                CV_Error(CV_StsUnsupportedFormat, error_message);
            }
        }
    }
    // get labels
    vector<int> labels = _lbls.getMat();
    // observations in row
    Mat data = asRowMatrix(src, CV_64FC1);
    // number of samples
    int n = data.rows;
    // dimensionality of data
    int d = data.cols;
    // assert there are as much samples as labels
    if(n != labels.size()) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", n, labels.size());
        CV_Error(CV_StsBadArg, error_message);
    }
    // clip number of components to be valid
    if((_num_components <= 0) || (_num_components > n))
        _num_components = n;
    // perform the PCA
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, _num_components);
    // copy the PCA results
    _mean = pca.mean.reshape(1,1); // store the mean vector
    _eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
    _eigenvectors = transpose(pca.eigenvectors); // eigenvectors by column
    _labels = labels; // store labels for prediction
    // save projections
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = subspace::project(_eigenvectors, _mean, data.row(sampleIdx).clone());
        this->_projections.push_back(p);
    }
}

void cv::Eigenfaces::predict(InputArray _src, int &minClass, double &minDist) const {
    Mat src = _src.getMat();
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This cv::Eigenfaces model is not computed yet. Did you call cv::Eigenfaces::train?";
        CV_Error(CV_StsError, error_message);
    } else if(_eigenvectors.rows != src.total()) {
        // check data alignment just for clearer exception messages
        string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(CV_StsError, error_message);
    }
    // project into PCA subspace
    Mat q = subspace::project(_eigenvectors, _mean, src.reshape(1,1));
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels[sampleIdx];
        }
    }
}

int cv::Eigenfaces::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

void cv::Eigenfaces::load(const FileStorage& fs) {
    //read matrices
    fs["num_components"] >> _num_components;
    fs["mean"] >> _mean;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
    // read sequences
    readFileNodeList(fs["projections"], _projections);
    readFileNodeList(fs["labels"], _labels);
}

void cv::Eigenfaces::save(FileStorage& fs) const {
    // write matrices
    fs << "num_components" << _num_components;
    fs << "mean" << _mean;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
    // write sequences
    writeFileNodeList(fs, "projections", _projections);
    writeFileNodeList(fs, "labels", _labels);
}

//------------------------------------------------------------------------------
// cv::Fisherfaces
//------------------------------------------------------------------------------
void cv::Fisherfaces::train(InputArray src, InputArray _lbls) {
    if(src.total() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(CV_StsUnsupportedFormat, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(CV_StsUnsupportedFormat, error_message);
    }
    // make sure data has correct size
    if(src.total() > 1) {
        for(int i = 1; i < src.total(); i++) {
            if(src.getMat(i-1).total() != src.getMat(i).total()) {
                string error_message = format("In the Fisherfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", src.getMat(i-1).total(), src.getMat(i).total());
                CV_Error(CV_StsUnsupportedFormat, error_message);
            }
        }
    }
    // get data
    vector<int> labels = _lbls.getMat();
    // wrap asRowMatrix in a try/catch, as people tend to pass wrong data here
    Mat data = asRowMatrix(src, CV_64FC1);
    // number of samples (N) and dimensions (D)
    int N = data.rows;
    int D = data.cols;
    // assert data is correctly given
    if(labels.size() != N) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", N, labels.size());
        CV_Error(CV_StsBadArg, error_message);
    }
    // the following equals len(unique(C))
    int C = remove_dups(labels).size();
    // clip number of components to be a valid number
    if((_num_components <= 0) || (_num_components > (C-1)))
        _num_components = (C-1);
    // perform a PCA and keep (N-C) components
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, (N-C));
    // project the data and perform a LDA on it
    subspace::LDA lda(pca.project(data), labels, _num_components);
    // store the total mean vector
    _mean = pca.mean.reshape(1,1);
    // store labels
    _labels = labels;
    // store the eigenvalues of the discriminants (and make sure they are doubles!)
    lda.eigenvalues().convertTo(_eigenvalues, CV_64FC1);
    // Now calculate the projection matrix as pca.eigenvectors * lda.eigenvectors.
    // Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
    gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);
    // store the projections of the original data
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = subspace::project(_eigenvectors, _mean, data.row(sampleIdx).clone());
        _projections.push_back(p);
    }
}

int cv::Fisherfaces::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}


void cv::Fisherfaces::predict(InputArray _src, int &minClass, double &minDist) const {
    Mat src = _src.getMat();
    // check data alignment just for clearer exception messages
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This cv::Fisherfaces model is not computed yet. Did you call cv::Fisherfaces::train?";
        CV_Error(CV_StsError, error_message);
    } else if(_eigenvectors.rows != src.total()) {
        string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(CV_StsError, error_message);
    }
    // project into LDA subspace
    Mat q = subspace::project(_eigenvectors, _mean, src.reshape(1,1));
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(int sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels[sampleIdx];
        }
    }
}

// See cv::FaceRecognizer::load.
void cv::Fisherfaces::load(const FileStorage& fs) {
    //read matrices
    fs["num_components"] >> _num_components;
    fs["mean"] >> _mean;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
    // read sequences
    readFileNodeList(fs["projections"], _projections);
    readFileNodeList(fs["labels"], _labels);
}

// See cv::FaceRecognizer::save.
void cv::Fisherfaces::save(FileStorage& fs) const {
    // write matrices
    fs << "num_components" << _num_components;
    fs << "mean" << _mean;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
    // write sequences
    writeFileNodeList(fs, "projections", _projections);
    writeFileNodeList(fs, "labels", _labels);
}

//------------------------------------------------------------------------------
// cv::LBPH
//------------------------------------------------------------------------------
void cv::LBPH::load(const FileStorage& fs) {
    fs["radius"] >> _radius;
    fs["neighbors"] >> _neighbors;
    fs["grid_x"] >> _grid_x;
    fs["grid_y"] >> _grid_y;
    //read matrices
    readFileNodeList(fs["histograms"], _histograms);
    readFileNodeList(fs["labels"], _labels);
}

// See cv::FaceRecognizer::save.
void cv::LBPH::save(FileStorage& fs) const {
    fs << "radius" << _radius;
    fs << "neighbors" << _neighbors;
    fs << "grid_x" << _grid_x;
    fs << "grid_y" << _grid_y;
    // write matrices
    writeFileNodeList(fs, "histograms", _histograms);
    writeFileNodeList(fs, "labels", _labels);
}

void cv::LBPH::train(InputArray _src, InputArray _lbls) {
    if(_src.kind() != _InputArray::STD_VECTOR_MAT && _src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        string error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
        CV_Error(CV_StsBadArg, error_message);
    }
    if(_src.total() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(CV_StsUnsupportedFormat, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(CV_StsUnsupportedFormat, error_message);
    }
    // get the vector of matrices
    vector<Mat> src;
    _src.getMatVector(src);
    // turn the label matrix into a vector
    vector<int> labels = _lbls.getMat();
    if(labels.size() != src.size()) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), labels.size());
        CV_Error(CV_StsBadArg, error_message);
    }
    // store given labels
    _labels = labels;
    // store the spatial histograms of the original data
    for(int sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        // calculate lbp image
        Mat lbp_image = elbp(src[sampleIdx], _radius, _neighbors);
        // get spatial histogram from this lbp image
        Mat p = spatial_histogram(
                lbp_image, /* lbp_image */
                static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
                _grid_x, /* grid size x */
                _grid_y, /* grid size y */
                true);
        // add to templates
        _histograms.push_back(p);
    }
}

int cv::LBPH::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

void cv::LBPH::predict(InputArray _src, int &minClass, double &minDist) const {
    Mat src = _src.getMat();
    // get the spatial histogram from input image
    Mat lbp_image = elbp(src, _radius, _neighbors);
    Mat query = spatial_histogram(
            lbp_image, /* lbp_image */
            static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
            _grid_x, /* grid size x */
            _grid_y, /* grid size y */
            true /* normed histograms */);
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(int sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
        double dist = compareHist(_histograms[sampleIdx], query, CV_COMP_CHISQR);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels[sampleIdx];
        }
    }
}
