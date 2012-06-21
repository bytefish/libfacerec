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

#include "subspace.hpp"
#include "helper.hpp"
#include "lbp.hpp"

using namespace std;

namespace cv {
enum
{
    COLORMAP_AUTUMN = 0,
    COLORMAP_BONE = 1,
    COLORMAP_JET = 2,
    COLORMAP_WINTER = 3,
    COLORMAP_RAINBOW = 4,
    COLORMAP_OCEAN = 5,
    COLORMAP_SUMMER = 6,
    COLORMAP_SPRING = 7,
    COLORMAP_COOL = 8,
    COLORMAP_HSV = 9,
    COLORMAP_PINK = 10,
    COLORMAP_HOT = 11
};

/**
 * Applies a colormap to the image given in src and returns the
 * colormap version in dst. Note: Mat::create is called on dst!
 *
 * The available colormaps are:
 *
 *   COLORMAP_AUTUMN = 0,
 *   COLORMAP_BONE = 1,
 *   COLORMAP_JET = 2,
 *   COLORMAP_WINTER = 3,
 *   COLORMAP_RAINBOW = 4,
 *   COLORMAP_OCEAN = 5,
 *   COLORMAP_SUMMER = 6,
 *   COLORMAP_SPRING = 7,
 *   COLORMAP_COOL = 8,
 *   COLORMAP_HSV = 9,
 *   COLORMAP_PINK = 10,
 *   COLORMAP_HOT = 11
 *
 */
CV_EXPORTS void applyColorMap(InputArray src, OutputArray dst, int colormap);


/**
 * cv::FaceRecognizer is the abstract base class for all
 * face recognition algorithms in OpenCV.
 */
class FaceRecognizer : public Algorithm {

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

CV_EXPORTS Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);

CV_EXPORTS Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);

CV_EXPORTS Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8,
                                                            int grid_x=8, int grid_y=8,
                                                            double threshold = DBL_MAX);

CV_EXPORTS bool initModule_contrib();


}

#endif
