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

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "facerec.hpp"
#include "helper.hpp"

#include <set>
#include <limits>

using std::set;

namespace cv
{

// Define the CV_INIT_ALGORITHM macro for OpenCV 2.4.0:
#ifndef CV_INIT_ALGORITHM
#define CV_INIT_ALGORITHM(classname, algname, memberinit) \
    static Algorithm* create##classname() \
    { \
        return new classname; \
    } \
    \
    static AlgorithmInfo& classname##_info() \
    { \
        static AlgorithmInfo classname##_info_var(algname, create##classname); \
        return classname##_info_var; \
    } \
    \
    static AlgorithmInfo& classname##_info_auto = classname##_info(); \
    \
    AlgorithmInfo* classname::info() const \
    { \
        static volatile bool initialized = false; \
        \
        if( !initialized ) \
        { \
            initialized = true; \
            classname obj; \
            memberinit; \
        } \
        return &classname##_info(); \
    }
#endif
  
// Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of
// Cognitive Neuroscience 3 (1991), 71–86.
class Eigenfaces : public FaceRecognizer
{
private:
    int _num_components;
    double _threshold;
    vector<Mat> _projections;
    Mat _labels;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes an empty Eigenfaces model.
    Eigenfaces(int num_components = 0, double threshold = DBL_MAX) :
        _num_components(num_components),
        _threshold(threshold) {}

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

    // Adds new images to this
    // in labels.
    void add(InputArray src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(InputArray _src, int &label, double &dist) const;

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;

    AlgorithmInfo* info() const;
};

// Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisher-
// faces: Recognition using class specific linear projection.". IEEE
// Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997),
// 711–720.
class Fisherfaces: public FaceRecognizer
{
private:
    int _num_components;
    double _threshold;
    Mat _eigenvectors;
    Mat _eigenvalues;
    Mat _mean;
    vector<Mat> _projections;
    Mat _labels;

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

    ~Fisherfaces() {}

    // Computes a Fisherfaces model with images in src and corresponding labels
    // in labels.
    void train(InputArray src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(InputArray _src, int &label, double &dist) const;

    // See FaceRecognizer::load.
    virtual void load(const FileStorage& fs);

    // See FaceRecognizer::save.
    virtual void save(FileStorage& fs) const;

    AlgorithmInfo* info() const;
};

// Face Recognition based on Local Binary Patterns.
//
//  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
//  patterns: Application to face recognition." IEEE Transactions on Pattern
//  Analysis and Machine Intelligence, 28(12):2037-2041.
//
class LBPH : public FaceRecognizer
{
private:
    int _grid_x;
    int _grid_y;
    int _radius;
    int _neighbors;
    double _threshold;

    vector<Mat> _histograms;
    Mat _labels;

    bool _uniform;
    int _num_uniforms;
    std::vector<int> _uniform_lookup;

    void initUniformLookup();
    
    // Computes a LBPH model with images in src and
    // corresponding labels in labels, possibly preserving
    // old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    // Initializes this LBPH Model. The current implementation is rather fixed
    // as it uses the Extended Local Binary Patterns per default.
    //
    // radius, neighbors are used in the local binary patterns creation.
    // grid_x, grid_y control the grid size of the spatial histograms.
    LBPH(int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            bool uniform = false) :
        _grid_x(gridx),
        _grid_y(gridy),
        _radius(radius_),
        _neighbors(neighbors_),
        _threshold(threshold),
        _num_uniforms(0),
        _uniform(uniform) {
        if (uniform) {
            initUniformLookup();
        }
    }

    // Initializes and computes this LBPH Model. The current implementation is
    // rather fixed as it uses the Extended Local Binary Patterns per default.
    //
    // (radius=1), (neighbors=8) are used in the local binary patterns creation.
    // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
    LBPH(InputArrayOfArrays src,
            InputArray labels,
            int radius_=1, int neighbors_=8,
            int gridx=8, int gridy=8,
            double threshold = DBL_MAX,
            bool uniform=false) :
                _grid_x(gridx),
                _grid_y(gridy),
                _radius(radius_),
                _neighbors(neighbors_),
                _threshold(threshold),
                _num_uniforms(0),
                _uniform(uniform) {
        if (uniform) {
            initUniformLookup();
        }
        train(src, labels);
    }

    ~LBPH() { }

    // Computes a LBPH model with images in src and
    // corresponding labels in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Updates this LBPH model with images in src and
    // corresponding labels in labels.
    void update(InputArrayOfArrays src, InputArray labels);

    // Predicts the label of a query image in src.
    int predict(InputArray src) const;

    // Predicts the label and confidence for a given sample.
    void predict(InputArray _src, int &label, double &dist) const;

    // See FaceRecognizer::load.
    void load(const FileStorage& fs);

    // See FaceRecognizer::save.
    void save(FileStorage& fs) const;

    // Getter functions.
    int neighbors() const { return _neighbors; }
    int radius() const { return _radius; }
    int grid_x() const { return _grid_x; }
    int grid_y() const { return _grid_y; }
    bool is_uniform() const { return _uniform; }

    AlgorithmInfo* info() const;
};


//------------------------------------------------------------------------------
// FaceRecognizer
//------------------------------------------------------------------------------
void FaceRecognizer::update(InputArrayOfArrays, InputArray) {
    string error_msg = format("This FaceRecognizer (%s) does not support updating, you have to use FaceRecognizer::train to update it.", this->name().c_str());
    CV_Error(Error::StsNotImplemented, error_msg);
}

void FaceRecognizer::save(const string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");
    this->save(fs);
    fs.release();
}

void FaceRecognizer::load(const string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");
    this->load(fs);
    fs.release();
}


//------------------------------------------------------------------------------
// Eigenfaces
//------------------------------------------------------------------------------
void Eigenfaces::train(InputArray _src, InputArray _local_labels) {
    if(_src.total() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsBadArg, error_message);
    } else if(_local_labels.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _local_labels.type());
        CV_Error(Error::StsBadArg, error_message);
    }
    // make sure data has correct size
    if(_src.total() > 1) {
        for(int i = 1; i < static_cast<int>(_src.total()); i++) {
            if(_src.getMat(i-1).total() != _src.getMat(i).total()) {
                string error_message = format("In the Eigenfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", _src.getMat(i-1).total(), _src.getMat(i).total());
                CV_Error(Error::StsUnsupportedFormat, error_message);
            }
        }
    }
    // get labels
    Mat labels = _local_labels.getMat();
    // observations in row
    Mat data = asRowMatrix(_src, CV_64FC1);
    // number of samples
   int n = data.rows;
    // assert there are as much samples as labels
    if(static_cast<int>(labels.total()) != n) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", n, labels.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // clear existing model data
    _labels.release();
    _projections.clear();
    // clip number of components to be valid
    if((_num_components <= 0) || (_num_components > n))
        _num_components = n;
    // perform the PCA
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, _num_components);
    // copy the PCA results
    _mean = pca.mean.reshape(1,1); // store the mean vector
    _eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
    transpose(pca.eigenvectors, _eigenvectors); // eigenvectors by column
    labels.copyTo(_labels); // store labels for prediction
    // save projections
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
        _projections.push_back(p);
    }
}

void Eigenfaces::predict(InputArray _src, int &minClass, double &minDist) const {
    // get data
    Mat src = _src.getMat();
    // make sure the user is passing correct data
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This Eigenfaces model is not computed yet. Did you call Eigenfaces::train?";
        CV_Error(Error::StsError, error_message);
    } else if(_eigenvectors.rows != static_cast<int>(src.total())) {
        // check data alignment just for clearer exception messages
        string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // project into PCA subspace
    Mat q = subspaceProject(_eigenvectors, _mean, src.reshape(1,1));
    minDist = DBL_MAX;
    minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels.at<int>((int)sampleIdx);
        }
    }
}

int Eigenfaces::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

void Eigenfaces::load(const FileStorage& fs) {
    //read matrices
    fs["num_components"] >> _num_components;
    fs["mean"] >> _mean;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
    // read sequences
    readFileNodeList(fs["projections"], _projections);
    fs["labels"] >> _labels;
}

void Eigenfaces::save(FileStorage& fs) const {
    // write matrices
    fs << "num_components" << _num_components;
    fs << "mean" << _mean;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
    // write sequences
    writeFileNodeList(fs, "projections", _projections);
    fs << "labels" << _labels;
}

//------------------------------------------------------------------------------
// Fisherfaces
//------------------------------------------------------------------------------

// See FaceRecognizer::load.
void Fisherfaces::load(const FileStorage& fs) {
    //read matrices
    fs["num_components"] >> _num_components;
    fs["mean"] >> _mean;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
    // read sequences
    readFileNodeList(fs["projections"], _projections);
    fs["labels"] >> _labels;
}

// See FaceRecognizer::save.
void Fisherfaces::save(FileStorage& fs) const {
    // write matrices
    fs << "num_components" << _num_components;
    fs << "mean" << _mean;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
    // write sequences
    writeFileNodeList(fs, "projections", _projections);
    fs << "labels" << _labels;
}

void Fisherfaces::train(InputArray src, InputArray _lbls) {
    if(src.total() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsBadArg, error_message);
    } else if(_lbls.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _lbls.type());
        CV_Error(Error::StsBadArg, error_message);
    }
    // make sure data has correct size
    if(src.total() > 1) {
        for(int i = 1; i < static_cast<int>(src.total()); i++) {
            if(src.getMat(i-1).total() != src.getMat(i).total()) {
                string error_message = format("In the Fisherfaces method all input samples (training images) must be of equal size! Expected %d pixels, but was %d pixels.", src.getMat(i-1).total(), src.getMat(i).total());
                CV_Error(Error::StsUnsupportedFormat, error_message);
            }
        }
    }
    // get data
    Mat labels = _lbls.getMat();
    Mat data = asRowMatrix(src, CV_64FC1);
    // number of samples
    int N = data.rows;
    // make sure labels are passed in correct shape
    if(labels.total() != (size_t) N) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", N, labels.total());
        CV_Error(Error::StsBadArg, error_message);
    } else if(labels.rows != 1 && labels.cols != 1) {
        string error_message = format("Expected the labels in a matrix with one row or column! Given dimensions are rows=%s, cols=%d.", labels.rows, labels.cols);
       CV_Error(Error::StsBadArg, error_message);
    }
    // clear existing model data
    _labels.release();
    _projections.clear();
    // safely copy from cv::Mat to std::vector
    vector<int> ll;
    for(size_t i = 0; i < labels.total(); i++) {
        ll.push_back(labels.at<int>(i));
    }
    // get the number of unique classes
    int C = (int) remove_dups(ll).size();
    // clip number of components to be a valid number
    if((_num_components <= 0) || (_num_components > (C-1)))
        _num_components = (C-1);
    // perform a PCA and keep (N-C) components
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, (N-C));
    // project the data and perform a LDA on it
    LDA lda(pca.project(data),labels, _num_components);
    // store the total mean vector
    _mean = pca.mean.reshape(1,1);
    // store labels
    labels.copyTo(_labels);
    // store the eigenvalues of the discriminants
    lda.eigenvalues().convertTo(_eigenvalues, CV_64FC1);
    // Now calculate the projection matrix as pca.eigenvectors * lda.eigenvectors.
    // Note: OpenCV stores the eigenvectors by row, so we need to transpose it!
    gemm(pca.eigenvectors, lda.eigenvectors(), 1.0, Mat(), 0.0, _eigenvectors, GEMM_1_T);
    // store the projections of the original data
    for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
        Mat p = subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
        _projections.push_back(p);
    }
}

void Fisherfaces::predict(InputArray _src, int &minClass, double &minDist) const {
    Mat src = _src.getMat();
    // check data alignment just for clearer exception messages
    if(_projections.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This Fisherfaces model is not computed yet. Did you call Fisherfaces::train?";
        CV_Error(Error::StsBadArg, error_message);
    } else if(src.total() != (size_t) _eigenvectors.rows) {
        string error_message = format("Wrong input image size. Reason: Training and Test images must be of equal size! Expected an image with %d elements, but got %d.", _eigenvectors.rows, src.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // project into LDA subspace
    Mat q = subspaceProject(_eigenvectors, _mean, src.reshape(1,1));
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
        double dist = norm(_projections[sampleIdx], q, NORM_L2);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels.at<int>((int)sampleIdx);
        }
    }
}

int Fisherfaces::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

//------------------------------------------------------------------------------
// cv::LBPH
//------------------------------------------------------------------------------

//
// a bitmask is 'uniform' if the number of transitions <= 2.
//
// we precompute the possible values to a lookup(index) table for
// all possible lbp combinations of n bits(neighbours).
//
// check, if the 1st bit eq 2nd, 2nd eq 3rd, ..., last eq 1st,
// else add a transition for each bit.
//
// if there's no transition, it's solid
// 1 transition: we've found a solid edge.
// 2 transitions: we've found a line.
//
// since the radius of the lbp operator is quite small,
// we consider any larger number of transitions as noise,
// and 'discard' them from our histogram, by assinging all of them
// to a single noise bin
//
// this way, using uniform lbp features boils down to indexing into the lut
// instead of the original value, and adjusting the sizes for the histograms.
//
bool bit(unsigned b, unsigned i) {
    return ((b & (1 << i)) != 0);
}
void LBPH::initUniformLookup() {
    int numSlots = 1 << _neighbors; // 2 ** _neighbours
    _num_uniforms = 0;
    _uniform_lookup = std::vector<int>(numSlots);
    for ( int i=0; i<numSlots; i++ ) {
        int transitions = 0;
        for ( int j=0; j<_neighbors-1; j++ ) {
            transitions += (bit(i,j) != bit(i,j+1));
        }
        transitions += (bit(i,_neighbors-1) != bit(i,0));

        if ( transitions <= 2 ) {
            _uniform_lookup[i] = _num_uniforms++;
        } else {
            _uniform_lookup[i] = -1; // mark all non-uniforms as noise channel
        }
    }

    // now run again through the lut, replace -1 with the 'noise' slot (numUniforms)
    for ( int i=0; i<numSlots; i++ ) {
        if ( _uniform_lookup[i] == -1 ) {
            _uniform_lookup[i] = _num_uniforms;
        }
    }
}

void LBPH::load(const FileStorage& fs) {
    fs["radius"] >> _radius;
    fs["neighbors"] >> _neighbors;
    fs["grid_x"] >> _grid_x;
    fs["grid_y"] >> _grid_y;
    //read matrices
    readFileNodeList(fs["histograms"], _histograms);
    fs["labels"] >> _labels;
    fs["uniform"] >> _uniform;
    if (_uniform) {
        initUniformLookup();
    }
}

// See cv::FaceRecognizer::save.
void LBPH::save(FileStorage& fs) const {
    fs << "radius" << _radius;
    fs << "neighbors" << _neighbors;
    fs << "grid_x" << _grid_x;
    fs << "grid_y" << _grid_y;
    // write matrices
    writeFileNodeList(fs, "histograms", _histograms);
    fs << "labels" << _labels;
    fs << "uniform" << _uniform;
}

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels) {
    this->train(_in_src, _in_labels, false);
}

void LBPH::update(InputArrayOfArrays _in_src, InputArray _in_labels) {
    // got no data, just return
    if(_in_src.total() == 0)
        return;
    // train
    this->train(_in_src, _in_labels, true);
}

void LBPH::train(InputArrayOfArrays _in_src, InputArray _in_labels, bool preserveData) {
    if(_in_src.kind() != _InputArray::STD_VECTOR_MAT && _in_src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        string error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< vector<...> >).";
        CV_Error(Error::StsBadArg, error_message);
    }
    if(_in_src.total() == 0) {
        string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(Error::StsUnsupportedFormat, error_message);
    } else if(_in_labels.getMat().type() != CV_32SC1) {
        string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
        CV_Error(Error::StsUnsupportedFormat, error_message);
    }
    // get the vector of matrices
    vector<Mat> src;
    _in_src.getMatVector(src);
    // get the label matrix
    Mat labels = _in_labels.getMat();
    // check if data is well- aligned
    if(labels.total() != src.size()) {
        string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), _labels.total());
        CV_Error(Error::StsBadArg, error_message);
    }
    // if this model should be trained without preserving old data, delete old model data
    if(!preserveData) {
        _labels.release();
        _histograms.clear();
    }
    // append labels to _labels matrix
    for(size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++) {
        _labels.push_back(labels.at<int>((int)labelIdx));
    }
    // store the spatial histograms of the original data
    for(size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        // calculate lbp image
        Mat lbp_image = elbp(src[sampleIdx], _radius, _neighbors, _uniform, _uniform_lookup);

        // check number of possible patterns ( this is different in the uniform case )
        int numPatterns = static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors)));
        if ( _uniform ) {
            numPatterns = _num_uniforms;
        }
        // get spatial histogram from this lbp image
        Mat p = spatial_histogram(
                lbp_image, /* lbp_image */
                numPatterns, /* number of possible patterns */
                _grid_x, /* grid size x */
                _grid_y, /* grid size y */
                true);
        // add to templates
        _histograms.push_back(p);
    }
}

void LBPH::predict(InputArray _src, int &minClass, double &minDist) const {
    if(_histograms.empty()) {
        // throw error if no data (or simply return -1?)
        string error_message = "This LBPH model is not computed yet. Did you call the train method?";
        CV_Error(Error::StsBadArg, error_message);
    }
    Mat src = _src.getMat();

    int numPatterns = 1 << _neighbors ; 
    if ( _uniform ) {
        numPatterns = _num_uniforms;
    }

    // get the spatial histogram from input image
    Mat lbp_image = elbp(src, _radius, _neighbors,_uniform,_uniform_lookup);
    Mat query = spatial_histogram(
            lbp_image, /* lbp_image */
            numPatterns, /* number of possible patterns */
            _grid_x, /* grid size x */
            _grid_y, /* grid size y */
            true /* normed histograms */);
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
        double dist = compareHist(_histograms[sampleIdx], query, HISTCMP_CHISQR);
        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels.at<int>(sampleIdx);
        }
    }
}

int LBPH::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components, double threshold)
{
    return new Eigenfaces(num_components, threshold);
}

Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components, double threshold)
{
    return new Fisherfaces(num_components, threshold);
}

Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius, int neighbors,
                                             int grid_x, int grid_y, double threshold, bool uniform)
{
    return new LBPH(radius, neighbors, grid_x, grid_y, threshold, uniform);
}

CV_INIT_ALGORITHM(Eigenfaces, "FaceRecognizer.Eigenfaces",
                  obj.info()->addParam(obj, "ncomponents", obj._num_components);
                  obj.info()->addParam(obj, "threshold", obj._threshold);
                  obj.info()->addParam(obj, "projections", obj._projections, true);
                  obj.info()->addParam(obj, "labels", obj._labels, true);
                  obj.info()->addParam(obj, "eigenvectors", obj._eigenvectors, true);
                  obj.info()->addParam(obj, "eigenvalues", obj._eigenvalues, true);
                  obj.info()->addParam(obj, "mean", obj._mean, true));

CV_INIT_ALGORITHM(Fisherfaces, "FaceRecognizer.Fisherfaces",
                  obj.info()->addParam(obj, "ncomponents", obj._num_components);
                  obj.info()->addParam(obj, "threshold", obj._threshold);
                  obj.info()->addParam(obj, "projections", obj._projections, true);
                  obj.info()->addParam(obj, "labels", obj._labels, true);
                  obj.info()->addParam(obj, "eigenvectors", obj._eigenvectors, true);
                  obj.info()->addParam(obj, "eigenvalues", obj._eigenvalues, true);
                  obj.info()->addParam(obj, "mean", obj._mean, true));

CV_INIT_ALGORITHM(LBPH, "FaceRecognizer.LBPH",
                  obj.info()->addParam(obj, "radius", obj._radius);
                  obj.info()->addParam(obj, "neighbors", obj._neighbors);
                  obj.info()->addParam(obj, "grid_x", obj._grid_x);
                  obj.info()->addParam(obj, "grid_y", obj._grid_y);
                  obj.info()->addParam(obj, "threshold", obj._threshold);
                  obj.info()->addParam(obj, "uniform", obj._uniform);
                  obj.info()->addParam(obj, "histograms", obj._histograms, true);
                  obj.info()->addParam(obj, "labels", obj._labels, true));

bool initModule_contrib()
{
    Ptr<Algorithm> efaces = createEigenfaces(), ffaces = createFisherfaces(), lbph = createLBPH();
    return efaces->info() != 0 && ffaces->info() != 0 && lbph->info() != 0;
}

}
