/*
 * Copyright (c) 2013 Kevin Hughes <kevinhughes27[at]gmail[dot]com> 
 * Modified from code by Philipp Wagner <bytefish[at]gmx[dot]de>
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
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <output_file> " << endl;
        exit(1);
    }
    string output_file;
    if (argc == 3) {
        output_file = string(argv[2]);
    }
    // Get the path to your CSV.
    string fn_csv = string(argv[1]);
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 3) {
        string error_message = "This demo needs at least 4 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }
    // The following code shuffles the input data, this 
    // is done so that when we create the test set
    // that is separate from the training set
    // there is a good variety of faces    
    vector<int> idx;
    for(int i = 0; i < (int)images.size(); i++)
        idx.push_back(i);
    std::random_shuffle(idx.begin(), idx.end());
    for(int i = 0; i < (int)idx.size(); i++)
    {
        std::swap( images[i], images[idx[i]] );
        std::swap( labels[i], labels[idx[i]] );
    }
    // The following lines split the data into two
    // data sets, one for training and one for testing
    // This is done, so that the training data (which 
    // we learn the cv::FaceRecognizer on) and the test 
    // data we test the model with, do not overlap.
    vector<Mat> training_images;
    vector<int> training_labels;
    vector<Mat> testing_images;
    vector<int> testing_labels;
    // default split is 80/20
    for(int i = 0; i < (int)images.size(); i++)
    {
        // training set		
        if( i < images.size() * 0.5 )
        {
            training_images.push_back( images[i] );
            training_labels.push_back( labels[i] );
        }
        // test set
        else
        {
             testing_images.push_back( images[i] );
             testing_labels.push_back( labels[i] );
        }
    }
    // The following lines create an Eigenfaces model
    // with the default object parameters and trains 
    // it with the images and labels read from the 
    // given CSV file.
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(training_images, training_labels);

    // Vary the threshold from low to high to
    // generate the data for an ROC curve and
    // to find the optimal threshold value
    //
    // Proper ROC curvers only apply to 
    // binary classifiers and not multiclass
    // classifiers so the theory here is simply
    // adapted from ROC analysis but the outcome
    // provides data of the same utility.
    //
    // See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    // for more details
    //
    std::ofstream out;
    out.open(output_file.c_str());
    out << "TP,FN,FP,TN,P,R,F1" << std::endl; // csv header
    for(int thresh = 1000; thresh < 4000; thresh = thresh + 100)
    {
        std::cout << "testing thresh: " << thresh << std::endl;

        int TruePositive=0, FalsePositive=0,
            TrueNegative=0, FalseNegative=0;
        
        for(int i = 0; i < (int)testing_images.size(); i++)
        {
            // Predict            
            int predictedLabel = -1;
            double confidence = 0.0;
            model->predict(testing_images[i], predictedLabel, confidence);

            if(confidence < thresh)
                predictedLabel = -1;
            
            // Analysis
            if( predictedLabel == testing_labels[i] )
                TruePositive++;
            else if( predictedLabel == -1 )
                FalseNegative++;
            else // predictedLabel != testing_labels[i]
                FalsePositive++;
        }

        double recall = (float)TruePositive / (float)(TruePositive+FalsePositive);
        double precision = (float)TruePositive / (float)(TruePositive+FalseNegative);
        double F1 = 2.0 *(precision*recall / (precision+recall));

        // save to file        
        out << TruePositive << "," << FalseNegative << "," << FalsePositive << ","
            << TrueNegative << "," << precision << "," << recall << "," << F1 << std::endl;
    }
    
    // close the file
    out.close();
    
    return 0;
}
