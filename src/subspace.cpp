#include "subspace.hpp"
#include "decomposition.hpp"
#include "helper.hpp"

#include <iostream>

//------------------------------------------------------------------------------
// cv::subspace::project
//------------------------------------------------------------------------------
Mat cv::subspace::project(InputArray _W, InputArray _mean, InputArray _src) {
    // get data matrices
    Mat W = _W.getMat();
    Mat mean = _mean.getMat();
    Mat src = _src.getMat();
    // get number of samples and dimension
    int n = src.rows;
    int d = src.cols;
    // make sure the data has the correct shape
    if(W.rows != d) {
        string error_message = format("Wrong shapes for given matrices. Was size(src) = (%d,%d), size(W) = (%d,%d).", src.rows, src.cols, W.rows, W.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    // make sure mean is correct if not empty
    if(!mean.empty() && (mean.total() != d)) {
        string error_message = format("Wrong mean shape for the given data matrix. Expected %d, but was %d.", d, mean.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    // create temporary matrices
    Mat X, Y;
    // make sure you operate on correct type
    src.convertTo(X, W.type());
    // safe to do, because of above assertion
    // safe to do, because of above assertion
    if(!mean.empty()) {
        for(int i=0; i<n; i++) {
            Mat r_i = X.row(i);
            subtract(r_i, mean.reshape(1,1), r_i);
        }
    }
    // finally calculate projection as Y = (X-mean)*W
    gemm(X, W, 1.0, Mat(), 0.0, Y);
    return Y;
}

//------------------------------------------------------------------------------
// cv::subspace::reconstruct
//------------------------------------------------------------------------------
Mat cv::subspace::reconstruct(InputArray _W, InputArray _mean, InputArray _src) {
    // get data matrices
    Mat W = _W.getMat();
    Mat mean = _mean.getMat();
    Mat src = _src.getMat();
    // get number of samples and dimension
    int n = src.rows;
    int d = src.cols;
    // make sure the data has the correct shape
    if(W.cols != d) {
        string error_message = format("Wrong shapes for given matrices. Was size(src) = (%d,%d), size(W) = (%d,%d).", src.rows, src.cols, W.rows, W.cols);
        CV_Error(CV_StsBadArg, error_message);
    }
    // make sure mean is correct if not empty
    if(!mean.empty() && (mean.total() != W.rows)) {
        string error_message = format("Wrong mean shape for the given eigenvector matrix. Expected %d, but was %d.", W.cols, mean.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    // initalize temporary matrices
    Mat X, Y;
    // copy data & make sure we are using the correct type
    src.convertTo(Y, W.type());
    // calculate the reconstruction
    gemm(Y, W, 1.0, Mat(), 0.0, X, GEMM_2_T);
    // safe to do because of above assertion
    if(!mean.empty()) {
        for(int i=0; i<n; i++) {
            Mat r_i = X.row(i);
            add(r_i, mean.reshape(1,1), r_i);
        }
    }
    return X;
}

//------------------------------------------------------------------------------
// Linear Discriminant Analysis implementation
//------------------------------------------------------------------------------
void cv::subspace::LDA::save(const string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened()) {
        CV_Error(CV_StsError, "File can't be opened for writing!");
    }
    this->save(fs);
    fs.release();
}

// Deserializes this object from a given filename.
void cv::subspace::LDA::load(const string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
       CV_Error(CV_StsError, "File can't be opened for writing!");
    }
    this->load(fs);
    fs.release();
}

// Serializes this object to a given cv::FileStorage.
void cv::subspace::LDA::save(FileStorage& fs) const {
    // write matrices
    fs << "num_components" << _num_components;
    fs << "eigenvalues" << _eigenvalues;
    fs << "eigenvectors" << _eigenvectors;
}

// Deserializes this object from a given cv::FileStorage.
void cv::subspace::LDA::load(const FileStorage& fs) {
    //read matrices
    fs["num_components"] >> _num_components;
    fs["eigenvalues"] >> _eigenvalues;
    fs["eigenvectors"] >> _eigenvectors;
}

void cv::subspace::LDA::lda(InputArray _src, InputArray _lbls) {
    // get data
    Mat src = _src.getMat();
    vector<int> labels = _lbls.getMat();
    // turn into row sampled matrix
    Mat data;
    // ensure working matrix is double precision
    src.convertTo(data, CV_64FC1);
    // maps the labels, so they're ascending: [0,1,...,C]
    vector<int> mapped_labels(labels.size());
    vector<int> num2label = remove_dups(labels);
    map<int, int> label2num;
    for (int i = 0; i < num2label.size(); i++)
        label2num[num2label[i]] = i;
    for (int i = 0; i < labels.size(); i++)
        mapped_labels[i] = label2num[labels[i]];
    // get sample size, dimension
    int N = data.rows;
    int D = data.cols;
    // number of unique labels
    int C = num2label.size();
    // we can't do a LDA on one class, what do you
    // want to separate from each other then?
    if(C == 1) {
        string error_message = "At least two classes are needed to perform a LDA. Reason: Only one class was given!";
        CV_Error(CV_StsBadArg, error_message);
    }
    // throw error if less labels, than samples
    if (labels.size() != N) {
        string error_message = format("The number of samples must equal the number of labels. Given %d labels, %d samples. ",labels.size(), N);
        CV_Error(CV_StsBadArg, error_message);
    }
    // warn if within-classes scatter matrix becomes singular
    if (N < D)
        cout << "Warning: Less observations than feature dimension given!"
        << "Computation will probably fail."
        << endl;
    // clip number of components to be a valid number
    if ((_num_components <= 0) || (_num_components > (C - 1)))
        _num_components = (C - 1);
    // holds the mean over all classes
    Mat meanTotal = Mat::zeros(1, D, data.type());
    // holds the mean for each class
    vector<Mat> meanClass(C);
    vector<int> numClass(C);
    // initialize
    for (int i = 0; i < C; i++) {
        numClass[i] = 0;
        meanClass[i] = Mat::zeros(1, D, data.type()); //! Dx1 image vector
    }
    // calculate sums
    for (int i = 0; i < N; i++) {
        Mat instance = data.row(i);
        int classIdx = mapped_labels[i];
        add(meanTotal, instance, meanTotal);
        add(meanClass[classIdx], instance, meanClass[classIdx]);
        numClass[classIdx]++;
    }
    // calculate total mean
    meanTotal.convertTo(meanTotal, meanTotal.type(), 1.0 / static_cast<double> (N));
    // calculate class means
    for (int i = 0; i < C; i++) {
        meanClass[i].convertTo(meanClass[i], meanClass[i].type(), 1.0 / static_cast<double> (numClass[i]));
    }
    // subtract class means
    for (int i = 0; i < N; i++) {
        int classIdx = mapped_labels[i];
        Mat instance = data.row(i);
        subtract(instance, meanClass[classIdx], instance);
    }
    // calculate within-classes scatter
    Mat Sw = Mat::zeros(D, D, data.type());
    mulTransposed(data, Sw, true);
    // calculate between-classes scatter
    Mat Sb = Mat::zeros(D, D, data.type());
    for (int i = 0; i < C; i++) {
        Mat tmp;
        subtract(meanClass[i], meanTotal, tmp);
        mulTransposed(tmp, tmp, true);
        add(Sb, tmp, Sb);
    }
    // invert Sw
    Mat Swi = Sw.inv();
    // M = inv(Sw)*Sb
    Mat M;
    gemm(Swi, Sb, 1.0, Mat(), 0.0, M);
    EigenvalueDecomposition es(M);
    _eigenvalues = es.eigenvalues();
    _eigenvectors = es.eigenvectors();
    // reshape eigenvalues, so they are stored by column
    _eigenvalues = _eigenvalues.reshape(1, 1);
    // get sorted indices descending by their eigenvalue
    vector<int> sorted_indices = argsort(_eigenvalues, false);
    // now sort eigenvalues and eigenvectors accordingly
    _eigenvalues = sortMatrixColumnsByIndices(_eigenvalues, sorted_indices);
    _eigenvectors = sortMatrixColumnsByIndices(_eigenvectors, sorted_indices);
    // and now take only the num_components and we're out!
    _eigenvalues = Mat(_eigenvalues, Range::all(), Range(0, _num_components));
    _eigenvectors = Mat(_eigenvectors, Range::all(), Range(0, _num_components));
}

void cv::subspace::LDA::compute(InputArray _src, InputArray _lbls) {
    switch(_src.kind()) {
    case _InputArray::STD_VECTOR_MAT:
        lda(asRowMatrix(_src, CV_64FC1), _lbls);
        break;
    case _InputArray::MAT:
        lda(_src.getMat(), _lbls);
        break;
    default:
        CV_Error(CV_StsNotImplemented, "This data type is not supported by cv::subspace::LDA::compute.");
        break;
    }
}

// Projects samples into the LDA subspace.
Mat cv::subspace::LDA::project(InputArray src) {
   return cv::subspace::project(_eigenvectors, Mat(), _dataAsRow ? src : transpose(src));
}

// Reconstructs projections from the LDA subspace.
Mat cv::subspace::LDA::reconstruct(InputArray src) {
   return cv::subspace::reconstruct(_eigenvectors, Mat(), _dataAsRow ? src : transpose(src));
}
