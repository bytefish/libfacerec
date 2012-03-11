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
#include "helper.hpp"

using namespace std;

namespace cv {

class FaceRecognizer {
public:

	//! virtual destructor
	virtual ~FaceRecognizer() {};

	// Trains a FaceRecognizer.
	virtual void train(const vector<Mat>& src, const vector<int>& labels) = 0;

	// Gets a prediction from a FaceRecognizer.
	virtual int predict(const Mat& src) = 0;
};


class Serializable {
public:
	//! virtual destructor
	virtual ~Serializable() {};

	// Serializes this object to a given filename.
	virtual void save(const string& filename) const {
		FileStorage fs(filename, FileStorage::WRITE);
		if(!fs.isOpened())
			CV_Error(CV_StsError, "File can't be opened for writing!");
		this->save(fs);
		fs.release();
	}

	// Deserializes this object from a given filename.
	virtual void load(const string& filename) {
		FileStorage fs(filename, FileStorage::READ);
		if(!fs.isOpened())
			CV_Error(CV_StsError, "File can't be opened for writing!");
		this->load(fs);
		fs.release();
	}

	//
	virtual void save(FileStorage& fs) const = 0;

	//
	virtual void load(const FileStorage& node) = 0;

};

//
class Eigenfaces : public FaceRecognizer, public Serializable {

private:
	int _num_components;
	vector<Mat> _projections;
	vector<int> _labels;
	Mat _eigenvectors;
	Mat _eigenvalues;
	Mat _mean;

public:
	using Serializable::save;
	using Serializable::load;

	//
	Eigenfaces(int num_components = 0) :
		_num_components(num_components) {};

	//
	Eigenfaces(const vector<Mat>& src,
			const vector<int>& labels,
			int num_components=0) : _num_components(num_components) {
		train(src, labels);
	}

	//
	void train(const vector<Mat>& src, const vector<int>& labels);

	//
	int predict(const Mat& src);

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

	// returns the eigenvectors of this PCA
	Mat eigenvectors() const { return _eigenvectors; }
	// returns the eigenvalues of this PCA
	Mat eigenvalues() const { return _eigenvalues; }
	// returns the mean of this PCA
	Mat mean() const { return _mean; }
};


class Fisherfaces : public FaceRecognizer, public Serializable {

private:
	int _num_components;
	Mat _eigenvectors;
	Mat _eigenvalues;
	Mat _mean;
	vector<Mat> _projections;
	vector<int> _labels;

public:
	using Serializable::save;
	using Serializable::load;

	Fisherfaces(int num_components=0) :
		_num_components(num_components) {};

	Fisherfaces(const vector<Mat>& src,
			const vector<int>& labels,
			int num_components = 0) :
				_num_components(num_components) {
		train(src, labels);
	}

	~Fisherfaces() {}

	void train(const vector<Mat>& src, const vector<int>& labels);

	int predict(const Mat& src);

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

	Mat eigenvectors() const { return _eigenvectors; };
	Mat eigenvalues() const { return _eigenvalues; }
	Mat mean() const { return _eigenvalues; }
};


}

#endif
