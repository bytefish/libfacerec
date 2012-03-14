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
#ifndef __FACEREC_HPP__
#define __FACEREC_HPP__

#include "opencv2/opencv.hpp"

#include "colormap.hpp"
#include "subspace.hpp"
#include "helper.hpp"
#include "lbp.hpp"

using namespace std;

namespace cv {

class FaceRecognizer {
public:

    //! virtual destructor
    virtual ~FaceRecognizer() {}

    // Trains a FaceRecognizer.
    virtual void train(const vector<Mat>& src, const vector<int>& labels) = 0;

    // Gets a prediction from a FaceRecognizer.
    virtual int predict(const Mat& src) = 0;

    // Serializes this object to a given filename.
    virtual void save(const string& filename) const {
        FileStorage fs(filename, FileStorage::WRITE);
        if (!fs.isOpened())
            CV_Error(CV_StsError, "File can't be opened for writing!");
        this->save(fs);
        fs.release();
    }

    // Deserializes this object from a given filename.
    virtual void load(const string& filename) {
        FileStorage fs(filename, FileStorage::READ);
        if (!fs.isOpened())
            CV_Error(CV_StsError, "File can't be opened for writing!");
        this->load(fs);
        fs.release();
    }

    // Serializes this object to a given cv::FileStorage.
    virtual void save(FileStorage& fs) const = 0;

    // Deserializes this object from a given cv::FileStorage.
    virtual void load(const FileStorage& node) = 0;
};

// Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of
// Cognitive Neuroscience 3 (1991), 71–86.
class Eigenfaces: public FaceRecognizer {

private:
    int _num_components;
    vector<Mat> _projections;
    vector<int> _labels;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes an empty Eigenfaces model.
    Eigenfaces(int num_components = 0) :
        _num_components(num_components) { }

    // Initializes and computes an Eigenfaces model with images in src and
    // corresponding labels in labels. num_components will be kept for
    // classification.
    Eigenfaces(const vector<Mat>& src, const vector<int>& labels,
            int num_components = 0) :
        _num_components(num_components) {
        train(src, labels);
    }

    // Computes an Eigenfaces model with images in src and corresponding labels
    // in labels.
    void train(const vector<Mat>& src, const vector<int>& labels) {
        // observations in row
        Mat data = asRowMatrix(src, CV_64FC1);
        // number of samples
        int n = data.rows;
        // dimensionality of data
        int d = data.cols;
        // assert there are as much samples as labels
        if(n != labels.size())
            CV_Error(CV_StsBadArg, "The number of samples must equal the number of labels!");
        // clip number of components to be valid
        if((_num_components <= 0) || (_num_components > n))
            _num_components = n;
        // perform the PCA
        PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, _num_components);
        // copy the PCA results
        _mean = pca.mean.reshape(1,1); // store the mean vector
        _eigenvalues = pca.eigenvalues.clone(); // store the eigenvectors
        _eigenvectors = transpose(pca.eigenvectors); // OpenCV stores the Eigenvectors by row (??)
        _labels = vector<int>(labels); // store labels for projections
        // save projections
        for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
            Mat p = subspace::project(_eigenvectors, _mean, data.row(sampleIdx));
            this->_projections.push_back(p);
        }
    }

    // Predicts the label of a query image in src.
    int predict(const Mat& src) {
        Mat q = subspace::project(_eigenvectors, _mean, src.reshape(1,1));
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

    // See cv::FaceRecognizer::load.
    void load(const FileStorage& fs) {
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
    void save(FileStorage& fs) const {
        // write matrices
        fs << "num_components" << _num_components;
        fs << "mean" << _mean;
        fs << "eigenvalues" << _eigenvalues;
        fs << "eigenvectors" << _eigenvectors;
        // write sequences
        writeFileNodeList(fs, "projections", _projections);
        writeFileNodeList(fs, "labels", _labels);
    }

    // Returns the eigenvectors of this PCA.
    Mat eigenvectors() const { return _eigenvectors; }

    // Returns the eigenvalues of this PCA.
    Mat eigenvalues() const { return _eigenvalues; }

    // Returns the sample mean of this PCA.
    Mat mean() const { return _mean; }

    // Returns the number of components used in this PCA.
    int num_components() const { return _num_components; }
};

// Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisher-
// faces: Recognition using class specific linear projection.". IEEE
// Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997),
// 711–720.
class Fisherfaces: public FaceRecognizer {

private:
    int _num_components;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;
    vector<Mat> _projections;
    vector<int> _labels;

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes an empty Fisherfaces model.
    Fisherfaces(int num_components = 0) :
        _num_components(num_components) {}

    // Initializes and computes a Fisherfaces model with images in src and
    // corresponding labels in labels. num_components will be kept for
    // classification.
    Fisherfaces(const vector<Mat>& src,
            const vector<int>& labels,
            int num_components = 0) :
        _num_components(num_components) {
        train(src, labels);
    }

    ~Fisherfaces() { }

    // Computes a Fisherfaces model with images in src and corresponding labels
    // in labels.
    void train(const vector<Mat>& src, const vector<int>& labels) {
        assert(src.size() == labels.size());
        Mat data = asRowMatrix(src, CV_64FC1);
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
        subspace::LDA lda(pca.project(data),labels, _num_components);
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

    // Predicts the label of a query image in src.
    int predict(const Mat& src) {
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

    // See cv::FaceRecognizer::load.
    void load(const FileStorage& fs) {
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
    void save(FileStorage& fs) const {
        // write matrices
        fs << "num_components" << _num_components;
        fs << "mean" << _mean;
        fs << "eigenvalues" << _eigenvalues;
        fs << "eigenvectors" << _eigenvectors;
        // write sequences
        writeFileNodeList(fs, "projections", _projections);
        writeFileNodeList(fs, "labels", _labels);
    }

    // Returns the eigenvectors of this Fisherfaces model.
    Mat eigenvectors() const { return _eigenvectors; }

    // Returns the eigenvalues of this Fisherfaces model.
    Mat eigenvalues() const { return _eigenvalues; }

    // Returns the sample mean of this Fisherfaces model.
    Mat mean() const { return _eigenvalues; }

    // Returns the number of components used in this Fisherfaces model.
    int num_components() const { return _num_components; }
};

// Face Recognition based on Local Binary Patterns.
//
// TODO Allow to change the distance metric.
// TODO Allow to change LBP computation (Extended LBP used right now).
// TODO Optimize, Optimize, Optimize!
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
class LBPH : public FaceRecognizer {

private:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;

    vector<Mat> _histograms;
    vector<int> _labels;

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes this LBPH Model. The current implementation is rather fixed
    // as it uses the Extended Local Binary Patterns per default.
    //
    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    LBPH(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8) :
        _grid_x(grid_x),
        _grid_y(grid_y),
        _radius(radius),
        _neighbors(neighbors) {}

    // Initializes and computes this LBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    LBPH(const vector<Mat>& src,
            const vector<int>& labels,
            int radius=1, int neighbors=8,
            int grid_x=8, int grid_y=8) :
                _grid_x(grid_x),
                _grid_y(grid_y),
                _radius(radius),
                _neighbors(neighbors) {
        train(src, labels);
    }

    ~LBPH() { }

    // Computes a LBPH model with images in src and
    // corresponding labels in labels.
    void train(const vector<Mat>& src, const vector<int>& labels) {
        assert(src.size() == labels.size());
        // store given labels
        _labels = labels;
        // store the spatial histograms of the original data
        for(int sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
            // calculate lbp image
            Mat lbp_image = elbp(src[sampleIdx], _radius, _neighbors);
            // get spatial histogram from this lbp image
            Mat p = spatial_histogram(
                    lbp_image, /* lbp_image */
                    std::pow(2, _neighbors), /* number of possible patterns */
                    _grid_x, /* grid size x */
                    _grid_y, /* grid size y */
                    true);
            // add to templates
            _histograms.push_back(p);
        }
    }

    // Predicts the label of a query image in src.
    int predict(const Mat& src) {
        // get the spatial histogram from input image
        Mat lbp_image = elbp(src, _radius, _neighbors);
        Mat query = spatial_histogram(
                lbp_image, /* lbp_image */
                std::pow(2, _neighbors), /* number of possible patterns */
                _grid_x, /* grid size x */
                _grid_y, /* grid size y */
                true /* normed histograms */);
        // find 1-nearest neighbor
        double minDist = numeric_limits<double>::max();
        int minClass = -1;
        for(int sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
            double dist = compareHist(_histograms[sampleIdx], query, CV_COMP_CHISQR);
            if(dist < minDist) {
                minDist = dist;
                minClass = _labels[sampleIdx];
            }
        }
        return minClass;
    }

    // See cv::FaceRecognizer::load.
    void load(const FileStorage& fs) {
        fs["radius"] >> _radius;
        fs["neighbors"] >> _neighbors;
        fs["grid_x"] >> _grid_x;
        fs["grid_y"] >> _grid_y;
        //read matrices
        readFileNodeList(fs["histograms"], _histograms);
        readFileNodeList(fs["labels"], _labels);
    }

    // See cv::FaceRecognizer::save.
    void save(FileStorage& fs) const {
        fs << "radius" << _radius;
        fs << "neighbors" << _neighbors;
        fs << "grid_x" << _grid_x;
        fs << "grid_y" << _grid_y;
        // write matrices
        writeFileNodeList(fs, "histograms", _histograms);
        writeFileNodeList(fs, "labels", _labels);
    }

    // Getter functions.
    int neighbors() { return _neighbors; }
    int radius() { return _radius; }
    int grid_x() { return _grid_x; }
    int grid_y() { return _grid_y; }

};

}

#endif
