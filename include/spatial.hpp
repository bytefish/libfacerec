/*
 * Copyright (c) 2013.berak (https://github.com/berak) 
 *                    Philipp Wagner <bytefish[at]gmx[dot]de>
 *                .
 * Released under terms of the BSD Simplified license.
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


#ifndef SPATIAL_HPP_
#define SPATIAL_HPP_

#include "facerec.hpp"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

namespace libfacerec {

// This implementation has been provided by berak (https://github.com/berak)
// and has been integrated into libfacerec. All credit goes to him.
//
// This is the base class for all FaceRecognizers implementing a SpatialHistogram
// algorithm (such as LBP). You have to override the 
// `virtual void oper(const Mat & src, Mat & hist) const`  method for the per pixel
// histogram mapping job.
//
class SpatialHistogramRecognizer : public FaceRecognizer
{
protected:

    //! the bread-and-butter thing, collect a histogram (per patch)
    virtual void oper(const Mat & src, Mat & hist) const = 0;

    //! choose a distance function for your algo
    virtual double distance(const Mat & hist_a, Mat & hist_b) const = 0;


protected:
    int _grid_x;
    int _grid_y;
    double _threshold;

    int step_size;
    int hist_len;
    int hist_type;

    std::vector<Mat> _histograms;
    Mat _labels;

    // Computes a SpatialHistogramRecognizer model with images in src and
    // corresponding labels in labels. Possibly preserving old model data.
    void train(InputArrayOfArrays src, InputArray labels, bool preserveData);

    Mat spatial_histogram(InputArray _src) const ;
    Mat spatial_histogram_overlap(InputArray _src) const ;


public:
    using FaceRecognizer::save;
    using FaceRecognizer::load;

    
    SpatialHistogramRecognizer( int gridx=8, int gridy=8,
            double threshold = DBL_MAX, int h_len=255,int h_type=CV_8U, int step_size=0) :
        _grid_x(gridx),
        _grid_y(gridy),
        _threshold(threshold),
        hist_len(h_len),
        hist_type(h_type),
        step_size(step_size) {}


    virtual ~SpatialHistogramRecognizer() {}

    // Computes a model with images in src and
    // corresponding labels in labels.
    void train(InputArrayOfArrays src, InputArray labels);

    // Updates this model with images in src and
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

    AlgorithmInfo* info() const;
};
}

#endif /* SPATIAL_HPP_ */
