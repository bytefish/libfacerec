#include "test_precomp.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/ts/ts.hpp"

// some helper methods for testing
#include "test_funs.hpp"

// includes objects under test
#include "lbp.hpp"

using namespace cv;
using namespace std;

// The fixture for testing class cv::LDA.
class LBPTest : public ::testing::Test {
 protected:

  // Once setup for all tests.
  LBPTest() {
      // 0,0,0
      // 0,0,0
      // 0,0,0
      mAllZero_ = (Mat_<unsigned char>(3,3) << 0,0,0,0,0,0,0,0,0);
      // 1,1,1
      // 1,0,1
      // 1,1,1
      mAllOneCenterZero_ = (Mat_<unsigned char>(3,3) << 1,1,1,1,0,1,1,1,1);
      // 47,51,65
      // 62,70,70
      // 80,83,70
      mMixed_ = (Mat_<unsigned char>(3,3) << 47, 51, 65, 62, 70, 70, 80, 83, 78);
  }

  virtual ~LBPTest() {}

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  virtual void SetUp() {}

  virtual void TearDown() {}

  // Objects declared here can be used by all tests in the test case.
  Mat mAllZero_;
  Mat mAllOneCenterZero_;
  Mat mMixed_;

};

TEST_F(LBPTest, checkOriginalLBPAllZero) {
    // Calculate Original LBP codes.
    Mat actual = olbp(mAllZero_);
    // 1x1 Matrix expected.
    ASSERT_EQ(1, actual.rows);
    ASSERT_EQ(1, actual.cols);
    // Check LBP Code.
    ASSERT_EQ(255, actual.at<unsigned char>(0,0));
}

TEST_F(LBPTest, checkOriginalLBPAllOneCenterZero) {
    // Calculate Original LBP codes.
    Mat actual = olbp(mAllOneCenterZero_);
    // 1x1 Matrix expected.
    ASSERT_EQ(1, actual.rows);
    ASSERT_EQ(1, actual.cols);
    // Check LBP Code.
    ASSERT_EQ(255, actual.at<unsigned char>(0,0));
}
TEST_F(LBPTest, checkOriginalLBPMixed) {
    // mMixed_ =
    // 47,51,65
    // 62,70,70
    // 80,83,78
    Mat actual = olbp(mMixed_);
    // 1x1 Matrix expected.
    ASSERT_EQ(1, actual.rows);
    ASSERT_EQ(1, actual.cols);
    // LBP: bin2dec(01111000) == 30
    ASSERT_EQ(30, actual.at<unsigned char>(0,0));
}
TEST_F(LBPTest, checkExtendedLBPAllZero) {
    // Calculate Original LBP codes.
    Mat actual = elbp(mAllZero_);
    // 1x1 Matrix expected.
    ASSERT_EQ(1, actual.rows);
    ASSERT_EQ(1, actual.cols);
    // Check LBP Code.
    ASSERT_EQ(255, actual.at<int>(0,0));
}

TEST_F(LBPTest, checkExtendedLBPAllOneCenterZero) {
    // Calculate Original LBP codes.
    Mat actual = Mat::zeros(1,1, CV_8UC1);
    // 1x1 Matrix expected.
    ASSERT_EQ(1, actual.rows);
    ASSERT_EQ(1, actual.cols);
    // Check LBP Code.
    ASSERT_EQ(0, actual.at<int>(0,0));
}

TEST_F(LBPTest, checkExtendedLBPMixed) {
    // mMixed_ =
    // 47,51,65
    // 62,70,70
    // 80,83,78
    //
    // neighbor n at (x,y) =
    // n=0, (0,1)
    // n=1, (-0.707107,0.707107)
    // n=2, (-1,0)
    // n=3, (-0.707107,-0.707107)
    // n=4, (0,-1)
    // n=5, (0.707107,-0.707107)
    // n=6, (1,0)
    // n=7, (0.707107,0.707107)
    //
    // Bilinear interpolation at sample points on mMixed_  =
    // 52.9081, 51, 63.565
    // 62,      70  70
    // 76.0355, 83, 76.6924
    //
    // Extended LBP code:
    // 0,0,0
    // 0,x,1
    // 1,1,1
    //
    // bin2dec(11000011) = 195
    //
    // Calculate Original LBP codes.
    Mat actual = elbp(mMixed_);
    // 1x1 Matrix expected.
    ASSERT_EQ(1, actual.rows);
    ASSERT_EQ(1, actual.cols);
    // Check LBP Code.
    ASSERT_EQ(195, actual.at<int>(0,0));
}
