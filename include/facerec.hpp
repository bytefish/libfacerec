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

#include "opencv2/core/core.hpp"


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
    virtual void train(InputArray src, InputArray labels) = 0;

    // Gets a prediction from a FaceRecognizer.
    virtual int predict(InputArray src) const = 0;

    // Gets a prediction from a FaceRecognizer.
    virtual void predict(InputArray src, int &label, double &confidence) const = 0;

    // Serializes this object to a given filename.
    virtual void save(const string& filename) const;

    // Deserializes this object from a given filename.
    virtual void load(const string& filename);

    // Serializes this object to a given cv::FileStorage.
    virtual void save(FileStorage& fs) const = 0;

    // Deserializes this object from a given cv::FileStorage.
    virtual void load(const FileStorage& fs) = 0;
};

// Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of
// Cognitive Neuroscience 3 (1991), 71–86.
class Eigenfaces : public FaceRecognizer {

private:
    int _num_components;
    double _threshold;

    vector<Mat> _projections;
    vector<int> _labels;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;


public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes an empty Eigenfaces model.
    Eigenfaces(int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) { }

    // Initializes and computes an Eigenfaces model with images in src and
    // corresponding labels in labels. num_components will be kept for
    // classification.
    Eigenfaces(InputArray src, InputArray labels,
            int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {
        train(src, labels);
    }

    // Computes an Eigenfaces model with images in src and corresponding labels
    // in labels.
    void train(InputArray src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(const InputArray src) const;

    // Returns the predicted label and confidence for the prediction.
    void predict(InputArray src, int &label, double &confidence) const;

    // See cv::FaceRecognizer::load.
    void load(const FileStorage& fs);

    // See cv::FaceRecognizer::save.
    void save(FileStorage& fs) const;

    // Returns the eigenvectors of this PCA.
    Mat eigenvectors() const { return _eigenvectors; }

    // Returns the eigenvalues of this PCA.
    Mat eigenvalues() const { return _eigenvalues; }

    // Returns the sample mean of this PCA.
    Mat mean() const { return _mean; }

    // Returns the number of components used in this PCA.
    int num_components() const { return _num_components; }

    // Returns the threshold used in this cv::Eigenfaces.
    double getThreshold() const { return _threshold; }

    // Sets the threshold used in this cv::Eigenfaces.
    void setThreshold(double threshold) { _threshold = threshold; }

};

// Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisher-
// faces: Recognition using class specific linear projection.". IEEE
// Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997),
// 711–720.
class Fisherfaces: public FaceRecognizer {

private:
    int _num_components;
    double _threshold;

    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;
    vector<Mat> _projections;
    vector<int> _labels;


public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes an empty Fisherfaces model.
    Fisherfaces(int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {}

    // Initializes and computes a Fisherfaces model with images in src and
    // corresponding labels in labels. num_components will be kept for
    // classification.
    Fisherfaces(InputArray src, InputArray labels,
            int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {
        train(src, labels);
    }

    ~Fisherfaces() { }

    // Computes a Fisherfaces model with images in src and corresponding labels
    // in labels.
    void train(InputArray src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Returns the predicted label and confidence for the prediction.
    void predict(InputArray src, int &label, double &confidence) const;

    // See cv::FaceRecognizer::load.
    virtual void load(const FileStorage& fs);

    // See cv::FaceRecognizer::save.
    virtual void save(FileStorage& fs) const;

    // Returns the eigenvectors of this Fisherfaces model.
    Mat eigenvectors() const { return _eigenvectors; }

    // Returns the eigenvalues of this Fisherfaces model.
    Mat eigenvalues() const { return _eigenvalues; }

    // Returns the sample mean of this Fisherfaces model.
    Mat mean() const { return _eigenvalues; }

    // Returns the number of components used in this Fisherfaces model.
    int num_components() const { return _num_components; }

    // Returns the threshold used in this cv::Fisherfaces.
    double getThreshold() const { return _threshold; }

    // Sets the threshold used in this cv::Fisherfaces.
    void setThreshold(double threshold) { _threshold = threshold; }
};

// Face Recognition based on Local Binary Patterns.
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
    double _threshold;

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
    LBPH(int radius=1, int neighbors=8,
            int grid_x=8, int grid_y=8,
            double threshold = DBL_MAX) :
        _grid_x(grid_x),
        _grid_y(grid_y),
        _radius(radius),
        _neighbors(neighbors),
        _threshold(threshold) {}

    // Initializes and computes this LBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    LBPH(InputArray src, InputArray labels,
            int radius=1, int neighbors=8,
            int grid_x=8, int grid_y=8,
            double threshold = DBL_MAX) :
                _grid_x(grid_x),
                _grid_y(grid_y),
                _radius(radius),
                _neighbors(neighbors),
                _threshold(threshold) {
        train(src, labels);
    }

    ~LBPH() { }

    // Computes a LBPH model with images in src and
    // corresponding labels in labels.
    void train(InputArray src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Returns the predicted label and confidence for the prediction.
    void predict(InputArray src, int &label, double &confidence) const;

    // See cv::FaceRecognizer::load.
    void load(const FileStorage& fs);

    // See cv::FaceRecognizer::save.
    void save(FileStorage& fs) const;

    // Getter functions.
    int neighbors() const { return _neighbors; }
    int radius() const { return _radius; }
    int grid_x() const { return _grid_x; }
    int grid_y() const { return _grid_y; }

    // Returns the threshold used in this cv::LBPH.
    double getThreshold() const { return _threshold; }

    // Sets the threshold used in this cv::LBPH.
    void setThreshold(double threshold) { _threshold = threshold; }


};

}

#endif
