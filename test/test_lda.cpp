
#include "opencv2/opencv.hpp"
#include "opencv2/ts/ts.hpp"

// some helper methods for testing
#include "test_funs.hpp"

// includes objects under test
#include "subspace.hpp"

using namespace cv;
using namespace std;

// The fixture for testing class cv::LDA.
class LDATest : public ::testing::Test {
 protected:

  // Once setup for all tests.
  LDATest() {
      // Example for a Linear Discriminant Analysis
      // (example taken from: http://www.bytefish.de/wiki/pca_lda_with_gnu_octave)
      double d[11][2] = {{2,3}, {3,4}, {4,5},
              {5,6}, {5,7}, {2,1},
              {3,2}, {4,2}, {4,3},
              {6,4}, {7,6} };

      int c[11] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 };

      X_ = Mat(11, 2, CV_64FC1, d).clone();
      vector<int> _classes(c, c + sizeof(c) / sizeof(int));

      lda1_.compute(X_, _classes);
  }

  virtual ~LDATest() {}

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  virtual void SetUp() {}

  virtual void TearDown() {}

  // Objects declared here can be used by all tests in the test case.
  subspace::LDA lda0_;
  Mat X_;
  subspace::LDA lda1_;

};

TEST_F(LDATest, ISEmptyInitially) {
    ASSERT_TRUE(lda0_.eigenvalues().empty());
    ASSERT_TRUE(lda0_.eigenvectors().empty());
}

TEST_F(LDATest, CheckEigenvalues) {
    // 2 classes, so there's only 1 Eigenvalue
    ASSERT_EQ(1, lda1_.eigenvalues().total());
    // The Eigenvalue should be the same for all solvers.
    ASSERT_NEAR(1.519536390756362, lda1_.eigenvalues().at<double>(0,0), 1e-10);
}

TEST_F(LDATest, CheckEigenvectors) {
    // 2 classes, so 1 Eigenvector (stored by column!)
    ASSERT_EQ(1, lda1_.eigenvectors().cols);
    // 2-dim data
    ASSERT_EQ(2, lda1_.eigenvectors().rows);
    // Eigenvectors found by JAMA
    Mat expected = (Mat_<double>(2,1) << 0.8254890051644113 , -0.8148145783734921);
    // Compare with a floating point precision of 1e-10.
    ASSERT_TRUE(isEqual(expected, lda1_.eigenvectors(), 1e-10));
}

TEST_F(LDATest, CheckProjection) {
    Mat sample0 = X_.row(0).clone();
    Mat actual = lda1_.project(sample0);
    // Projection found by JAMA.
    Mat expected = (Mat_<double>(1,1) << -0.7934657247916539);
    ASSERT_TRUE(isEqual(expected, actual, 1e-10));
}

TEST_F(LDATest, CheckReconstruction) {
    Mat sample0 = X_.row(0).clone();
    Mat actual = lda1_.reconstruct(lda1_.project(sample0));
    // Reconstruction found by JAMA.
    Mat expected = (Mat_<double>(1,2) << -0.6549972317903209, 0.6465274399999288);
    ASSERT_TRUE(isEqual(expected, actual, 1e-10));
}
