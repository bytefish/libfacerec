
#include "opencv2/core/core.hpp"
#include "opencv2/ts.hpp"

#include "facerec.hpp"

#include <vector>

using namespace cv;
using namespace std;

// TODO Port the various fine granular numerical tests from libfacerec at
//  https://github.com/bytefish/libfacerec/tree/master/test to the
//  cv::FaceRecognizer OpenCV (because in OpenCV all this is hidden within
//  the implementation).

//------------------------------------------------------------------------------
// cv::Eigenfaces
//------------------------------------------------------------------------------

TEST(TestEigenfaces, TrainMultiple) {
    // Create the new Eigenfaces model:
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();

    vector<Mat> images;
    vector<int> labels;

    // Add data for two classes:
    Mat img0 = Mat::zeros(100,100, CV_8UC1);
    Mat img1 = Mat::zeros(100,100, CV_8UC1);
    Mat img2 = Mat::ones(100,100, CV_8UC1);
    Mat img3 = Mat::ones(100,100, CV_8UC1);

    images.push_back(img0); labels.push_back(0);
    images.push_back(img1); labels.push_back(0);

    images.push_back(img2); labels.push_back(1);
    images.push_back(img3); labels.push_back(1);

    // Train the model the first time.
    ASSERT_NO_THROW(model->train(images, labels));
    {
        vector<Mat> const &projections = model->getMatVector("projections");
        Mat const &l = model->getMat("labels");

        ASSERT_EQ(projections.size(), 4);
        ASSERT_EQ(l.total(), 4);

        ASSERT_EQ(l.at<int>(0), 0);
        ASSERT_EQ(l.at<int>(1), 0);
        ASSERT_EQ(l.at<int>(2), 1);
        ASSERT_EQ(l.at<int>(3), 1);
    }

    // delete an element
    images.pop_back();
    labels.pop_back();

    // change labels
    labels[1] = 1;
    labels[2] = 2;

    // Train a second time, this should delete the old data:
    ASSERT_NO_THROW(model->train(images, labels));
    {
        vector<Mat> projections = model->getMatVector("projections");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(projections.size(), 3);
        ASSERT_EQ(labels.total(), 3);

        ASSERT_EQ(labels.at<int>(0), 0);
        ASSERT_EQ(labels.at<int>(1), 1);
        ASSERT_EQ(labels.at<int>(2), 2);
    }
}

TEST(TestEigenfaces, UpdateThrows) {
    // Create the new Eigenfaces model:
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    vector<Mat> src;
    vector<int> labels;
    ASSERT_ANY_THROW(model->update(src, labels));
}

TEST(TestEigenfaces, PredictEmpty) {
    // Create the new Eigenfaces model:
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    // Create a sample:
    Mat img0 = Mat::ones(100,100, CV_8UC1);
    // Not trained yet. This should fail!
    ASSERT_ANY_THROW(model->predict(img0));
}

TEST(TestEigenfaces, PredictWrongType) {
    // Create the new Eigenfaces model:
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    // Holds the training data:
    vector<Mat> images;
    vector<int> labels;
    // Create a training sample:
    Mat img0 = Mat::zeros(100,100, CV_8UC1);

    images.push_back(img0);
    labels.push_back(0);
    // Learn the model:
    ASSERT_NO_THROW(model->train(images, labels));
    // Create test samples:
    Mat correct_type = Mat::ones(100, 100, CV_8UC1);
    Mat wrong_type = Mat::ones(100, 100, CV_32FC3);
    // Trained and correct type. This shouldn't fail!
    ASSERT_NO_THROW(model->predict(correct_type));
    // Trained, but wrong type. This should fail!
    ASSERT_ANY_THROW(model->predict(wrong_type));
}

//------------------------------------------------------------------------------
// cv::Fisherfaces
//------------------------------------------------------------------------------

TEST(TestFisherfaces, TrainMultiple) {
    // Create the new LBPH model:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();

    vector<Mat> images;
    vector<int> labels;

    // Add data for two classes:
    Mat img0 = Mat::zeros(100,100, CV_8UC1);
    Mat img1 = Mat::zeros(100,100, CV_8UC1);
    Mat img2 = Mat::ones(100,100, CV_8UC1);
    Mat img3 = Mat::ones(100,100, CV_8UC1);

    images.push_back(img0); labels.push_back(0);
    images.push_back(img1); labels.push_back(0);

    images.push_back(img2); labels.push_back(1);
    images.push_back(img3); labels.push_back(1);

    // Train the model the first time.
    ASSERT_NO_THROW(model->train(images, labels));
    {
        vector<Mat> projections = model->getMatVector("projections");
        Mat l = model->getMat("labels");

        ASSERT_EQ(projections.size(), 4);
        ASSERT_EQ(l.total(), 4);

        ASSERT_EQ(l.at<int>(0), 0);
        ASSERT_EQ(l.at<int>(1), 0);
        ASSERT_EQ(l.at<int>(2), 1);
        ASSERT_EQ(l.at<int>(3), 1);
    }

    // delete an element
    images.pop_back();
    labels.pop_back();

    // change labels
    labels[1] = 1;
    labels[2] = 2;

    // Train a second time, this should delete the old data:
    ASSERT_NO_THROW(model->train(images, labels));
    {
        vector<Mat> projections = model->getMatVector("projections");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(projections.size(), 3);
        ASSERT_EQ(labels.total(), 3);

        ASSERT_EQ(labels.at<int>(0), 0);
        ASSERT_EQ(labels.at<int>(1), 1);
        ASSERT_EQ(labels.at<int>(2), 2);
    }
}

TEST(TestFisherfaces, UpdateThrows) {
    // Create the new Fisherfaces model:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    vector<Mat> src;
    vector<int> labels;
    ASSERT_ANY_THROW(model->update(src, labels));
}


TEST(TestFisherfaces, PredictEmpty) {
    // Create the new Fisherfaces model:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    // Create a sample:
    Mat img0 = Mat::ones(100,100, CV_8UC1);
    // Not trained yet. This should fail!
    ASSERT_ANY_THROW(model->predict(img0));
}

TEST(TestFisherfaces, PredictWrongType) {
    // Create the new Eigenfaces model:
    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    // Holds the training data:
    vector<Mat> images;
    vector<int> labels;
    // Create a training sample:
    Mat img0 = Mat::zeros(100,100, CV_8UC1);
    Mat img1 = Mat::zeros(100,100, CV_8UC1);

    images.push_back(img0); labels.push_back(0);
    images.push_back(img1); labels.push_back(1);

    // Learn the model:
    ASSERT_NO_THROW(model->train(images, labels));
    // Create test samples:
    Mat correct_type = Mat::ones(100, 100, CV_8UC1);
    Mat wrong_type = Mat::ones(100, 100, CV_32FC3);
    // Trained and correct type. This shouldn't fail!
    ASSERT_NO_THROW(model->predict(correct_type));
    // Trained, but wrong type. This should fail!
    ASSERT_ANY_THROW(model->predict(wrong_type));
}
//------------------------------------------------------------------------------
// cv::LBPH
//------------------------------------------------------------------------------
TEST(TestLBPH, TrainMultiple) {
    // Create the new LBPH model:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

    vector<Mat> images;
    vector<int> labels;

    Mat img0 = Mat::zeros(100,100, CV_8UC1);

    images.push_back(img0);
    labels.push_back(0);

    // Train the model the first time.
    ASSERT_NO_THROW(model->train(images, labels));
    {
        vector<Mat> histograms = model->getMatVector("histograms");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(histograms.size(), 1);
        ASSERT_EQ(labels.total(), 1);

        ASSERT_EQ(labels.at<int>(0), 0);
    }

    // add another sample
    Mat img1 = Mat::zeros(100,100, CV_8UC1);
    images.push_back(img1);
    labels.push_back(1);

    // train the model
    ASSERT_NO_THROW(model->train(images, labels));
    {
        vector<Mat> histograms = model->getMatVector("histograms");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(histograms.size(), 2);
        ASSERT_EQ(labels.total(), 2);

        ASSERT_EQ(labels.at<int>(0), 0);
        ASSERT_EQ(labels.at<int>(1), 1);
    }
}

TEST(TestLBPH, UpdateNoThrow) {
    // Create the new LBPH model:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    vector<Mat> src;
    vector<int> labels;
    ASSERT_NO_THROW(model->update(src, labels));
}

TEST(TestLBPH, UpdateAfterTrain) {
    // Create the new LBPH model:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

    vector<Mat> images;
    vector<int> labels;

    Mat img0 = Mat::zeros(100,100, CV_8UC1);

    images.push_back(img0);
    labels.push_back(0);

    // Train the model the first time.
    ASSERT_NO_THROW(model->train(images, labels));
    {
        vector<Mat> histograms = model->getMatVector("histograms");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(histograms.size(), 1);
        ASSERT_EQ(labels.total(), 1);

        ASSERT_EQ(labels.at<int>(0), 0);
    }

    // add another sample
    Mat img1 = Mat::zeros(100,100, CV_8UC1);
    images.push_back(img1);
    labels.push_back(1);

    // train the model
    ASSERT_NO_THROW(model->update(images, labels));
    {
        vector<Mat> histograms = model->getMatVector("histograms");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(histograms.size(), 3);
        ASSERT_EQ(labels.total(), 3);

        ASSERT_EQ(labels.at<int>(0), 0);
        ASSERT_EQ(labels.at<int>(1), 0);
        ASSERT_EQ(labels.at<int>(2), 1);
    }
}

TEST(TestLBPH, InitialUpdateWithoutTrain) {
    // Create the new LBPH model:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();

    vector<Mat> images;
    vector<int> labels;

    Mat img0 = Mat::zeros(100,100, CV_8UC1);

    images.push_back(img0);
    labels.push_back(0);

    // Train the model the first time.
    ASSERT_NO_THROW(model->update(images, labels));
    {
        vector<Mat> histograms = model->getMatVector("histograms");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(histograms.size(), 1);
        ASSERT_EQ(labels.total(), 1);

        ASSERT_EQ(labels.at<int>(0), 0);
    }

    // add another sample
    Mat img1 = Mat::zeros(100,100, CV_8UC1);
    images.push_back(img1);
    labels.push_back(1);

    // train the model
    ASSERT_NO_THROW(model->update(images, labels));
    {
        vector<Mat> histograms = model->getMatVector("histograms");
        Mat labels = model->getMat("labels");

        ASSERT_EQ(histograms.size(), 3);
        ASSERT_EQ(labels.total(), 3);

        ASSERT_EQ(labels.at<int>(0), 0);
        ASSERT_EQ(labels.at<int>(1), 0);
        ASSERT_EQ(labels.at<int>(2), 1);
    }
}

TEST(TestLBPH, PredictEmpty) {
    // Create the new Eigenfaces model:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    // Create a sample:
    Mat img0 = Mat::ones(100, 100, CV_8UC1);
    // Not trained yet. This should fail!
    ASSERT_ANY_THROW(model->predict(img0));
}

TEST(TestLBPH, PredictWrongType) {
    // Create the new Eigenfaces model:
    Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
    // Holds the training data:
    vector<Mat> images;
    vector<int> labels;
    // Create a training sample:
    Mat img0 = Mat::zeros(100,100, CV_8UC1);

    images.push_back(img0);
    labels.push_back(0);
    // Learn the model:
    ASSERT_NO_THROW(model->train(images, labels));
    // Create test samples:
    Mat correct_type = Mat::ones(100, 100, CV_8UC1);
    Mat wrong_type = Mat::ones(100, 100, CV_32FC3);
    // Trained and correct type. This shouldn't fail!
    ASSERT_NO_THROW(model->predict(correct_type));
    // Trained, but wrong type. This should fail!
    ASSERT_NO_THROW(model->predict(wrong_type));
}
