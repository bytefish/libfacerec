#include "helper.hpp"
#include "facerec.hpp"
#include "subspace.hpp"

using namespace cv;

void Eigenfaces::train(const vector<Mat>& src, const vector<int>& labels) {
	// observations in row
	Mat data = asRowMatrix(src);
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
	PCA pca(data,
			Mat(),
			CV_PCA_DATA_AS_ROW,
			_num_components);
	// set the data
	_mean = _dataAsRow ? pca.mean.reshape(1,1) : pca.mean.reshape(1, pca.mean.total()); // store the mean vector
	_eigenvalues = pca.eigenvalues.clone(); // store the eigenvectors
	_eigenvectors = transpose(pca.eigenvectors); // OpenCV stores the Eigenvectors by row (??)
	_labels = vector<int>(labels); // store labels for projections
	// projections
	for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
		Mat p = subspace::project(_eigenvectors, _mean, data.row(sampleIdx));
		this->_projections.push_back(p);
	}
}

int Eigenfaces::predict(const Mat& src) {
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

