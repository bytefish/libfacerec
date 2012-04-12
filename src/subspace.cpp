#include "subspace.hpp"
#include "decomposition.hpp"
#include "helper.hpp"

//------------------------------------------------------------------------------
// cv::subspace::project
//------------------------------------------------------------------------------
Mat cv::subspace::project(InputArray _W, InputArray _mean, InputArray _src) {
    // get data matrices
    Mat W = _W.getMat();
    Mat mean = _mean.getMat();
    Mat src = _src.getMat();
    // create temporary matrices
    Mat X, Y;
    // copy data & make sure we are using the correct type
    src.convertTo(X, W.type());
    // get number of samples and dimension
    int n = X.rows;
    int d = X.cols;
    // center the data if correct aligned sample mean is given
    if(mean.total() == d)
        subtract(X, repeat(mean.reshape(1,1), n, 1), X);
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
    // initalize temporary matrices
    Mat X, Y;
    // copy data & make sure we are using the correct type
    src.convertTo(Y, W.type());
    // calculate the reconstruction
    gemm(Y,
            W,
            1.0,
            (d == mean.total()) ? repeat(mean.reshape(1,1), n, 1) : Mat(),
            (d == mean.total()) ? 1.0 : 0.0,
            X,
            GEMM_2_T);
    return X;
}


//------------------------------------------------------------------------------
// Linear Discriminant Analysis implementation
//------------------------------------------------------------------------------
void cv::subspace::LDA::save(const string& filename) const {
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
        CV_Error(CV_StsError, "File can't be opened for writing!");
    this->save(fs);
    fs.release();
}

// Deserializes this object from a given filename.
void cv::subspace::LDA::load(const string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
       CV_Error(CV_StsError, "File can't be opened for writing!");
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
    // throw error if less labels, than samples
    if (labels.size() != N)
        CV_Error(CV_StsBadArg, "Error: The number of samples must equal the number of labels.");
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
    // calculate means
    meanTotal.convertTo(meanTotal, meanTotal.type(),
            1.0 / static_cast<double> (N));
    for (int i = 0; i < C; i++)
        meanClass[i].convertTo(meanClass[i], meanClass[i].type(),
                1.0 / static_cast<double> (numClass[i]));
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
