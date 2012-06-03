
#include "opencv2/opencv.hpp"
#include "opencv2/ts/ts.hpp"

// some helper methods for testing
#include "test_funs.hpp"
#include "helper.hpp"

using namespace cv;
using namespace std;

//------------------------------------------------------------------------------
// cv::histogram
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// cv::argsort
//------------------------------------------------------------------------------
TEST(HelperTest, checkArgsort) {

    Mat mNotSorted = (Mat_<double>(1,4) << 1.0, 0.0, 3.0, -1.0);

    vector<int> expected_ascending;

    expected_ascending.push_back(3);
    expected_ascending.push_back(1);
    expected_ascending.push_back(0);
    expected_ascending.push_back(2);

    Mat x = cv::argsort(mNotSorted, true);

    vector<int> actual_ascending = cv::argsort(mNotSorted, true);
    vector<int> actual_descending = cv::argsort(mNotSorted, false);

    // check if actual_ascending is sorted correctly
    for(int i=0; i < expected_ascending.size(); i++)
        ASSERT_EQ(expected_ascending[i], actual_ascending[i]);
    // check if actual_descending is sorted correctly
    for(int i=0; i < expected_ascending.size(); i++)
        ASSERT_EQ(expected_ascending[expected_ascending.size()-(i+1)], actual_descending[i]);

}

TEST(HelperTest, checkArgsortOneElement) {
    Mat mNotSorted = (Mat_<double>(1,1) << 0.0);
    // only 0 expected as result
    vector<int> expected_ascending;
    expected_ascending.push_back(0);

    vector<int> actual_ascending = cv::argsort(mNotSorted, true);
    vector<int> actual_descending = cv::argsort(mNotSorted, false);

    ASSERT_EQ(expected_ascending[0], actual_ascending[0]);
    ASSERT_EQ(expected_ascending[0], actual_descending[0]);
}

TEST(HelperTest, checkArgsortThrow) {
    Mat mEmpty = (Mat_<double>());
    Mat m2by2 = (Mat_<double>(2,2));

    ASSERT_ANY_THROW(cv::argsort(mEmpty, true));
    ASSERT_ANY_THROW(cv::argsort(m2by2, true));

    ASSERT_ANY_THROW(cv::argsort(mEmpty, false));
    ASSERT_ANY_THROW(cv::argsort(m2by2, false));
}

//------------------------------------------------------------------------------
// cv::isSymmetric
//------------------------------------------------------------------------------
TEST(HelperTest, checkSymmetryEmpty) {
    Mat mEmptyMat = (Mat_<double>(0,0));
    ASSERT_TRUE(isSymmetric(mEmptyMat));
}

TEST(HelperTest, checkSymmetryOneElement) {
    Mat mOneElement = (Mat_<double>(1,1) << 1);
    ASSERT_TRUE(isSymmetric(mOneElement));
}

TEST(HelperTest, checkSymmetryIntegral) {
    Mat mSymmetric = (Mat_<int>(2,2) << 1, 2, 2, 1);
    Mat mNonSymmetric = (Mat_<int>(2,2) << 1, 2, 3, 4);

    // Check isSymmetric for a floating point matrices.
    ASSERT_TRUE(isSymmetric(mSymmetric));
    ASSERT_FALSE(isSymmetric(mNonSymmetric));
}

TEST(HelperTest, checkSymmetryInexact) {
    Mat mAlmostSymmetric = (Mat_<double>(2,2) << 1.0, 2.0, 2.000001, 1.0);
    ASSERT_TRUE(isSymmetric(mAlmostSymmetric, 1e-3));
    ASSERT_FALSE(isSymmetric(mAlmostSymmetric, 1e-15));
}

TEST(HelperTest, checkSymmetryAllTypes) {
    Mat mSymmetric = (Mat_<double>(2,2) << 1, 2, 2, 1);
    Mat mNonSymmetric = (Mat_<double>(2,2) << 1, 2, 3, 4);

    // Check isSymmetric for all possible OpenCV types.
    ASSERT_TRUE(isSymmetric(getMatrixAsType<char>(mSymmetric)));
    ASSERT_TRUE(isSymmetric(getMatrixAsType<unsigned char>(mSymmetric)));
    ASSERT_TRUE(isSymmetric(getMatrixAsType<short>(mSymmetric)));
    ASSERT_TRUE(isSymmetric(getMatrixAsType<unsigned short>(mSymmetric)));
    ASSERT_TRUE(isSymmetric(getMatrixAsType<int>(mSymmetric)));
    ASSERT_TRUE(isSymmetric(getMatrixAsType<float>(mSymmetric)));
    ASSERT_TRUE(isSymmetric(getMatrixAsType<double>(mSymmetric)));

    ASSERT_FALSE(isSymmetric(getMatrixAsType<char>(mNonSymmetric)));
    ASSERT_FALSE(isSymmetric(getMatrixAsType<unsigned char>(mNonSymmetric)));
    ASSERT_FALSE(isSymmetric(getMatrixAsType<short>(mNonSymmetric)));
    ASSERT_FALSE(isSymmetric(getMatrixAsType<unsigned short>(mNonSymmetric)));
    ASSERT_FALSE(isSymmetric(getMatrixAsType<int>(mNonSymmetric)));
    ASSERT_FALSE(isSymmetric(getMatrixAsType<float>(mNonSymmetric)));
    ASSERT_FALSE(isSymmetric(getMatrixAsType<double>(mNonSymmetric)));
}

//------------------------------------------------------------------------------
// cv::sortMatrixColumnsByIndices
//------------------------------------------------------------------------------
TEST(HelperTest, checkSortByColumn) {
    // {{1.0, 2.0},
    //  {3.0, 4.0}};
    Mat mUnsorted = (Mat_<double>(2,2) << 1.0, 2.0, 3.0, 4.0);
    // [1,0]
    vector<int> indices;
    indices.push_back(1);
    indices.push_back(0);
    // Sorting columns by indices:
    // {{2.0, 1.0},
    //  {4.0, 3.0}};
    Mat expected = (Mat_<double>(2,2) << 2.0, 1.0, 4.0, 3.0);

    Mat actual = cv::sortMatrixColumnsByIndices(mUnsorted, indices);

    ASSERT_TRUE(isEqual(expected, actual));
}

//------------------------------------------------------------------------------
// cv::sortMatrixRowsByIndices
//------------------------------------------------------------------------------
TEST(HelperTest, checkSortByRow) {
    // {{1.0, 2.0},
    //  {3.0, 4.0}};
    Mat mUnsorted = (Mat_<double>(2,2) << 1.0, 2.0, 3.0, 4.0);
    // [1,0]
    vector<int> indices;
    indices.push_back(1);
    indices.push_back(0);
    // {{3.0, 4.0},
    //  {1.0, 2.0}};
    Mat expected = (Mat_<double>(2,2) << 3.0, 4.0, 1.0, 2.0);

    Mat actual = cv::sortMatrixRowsByIndices(mUnsorted, indices);

    ASSERT_TRUE(isEqual(expected, actual));
}

//------------------------------------------------------------------------------
// cv::asRowMatrix
//------------------------------------------------------------------------------
TEST(HelperTest, asRowMatrix_Empty) {
    vector<Mat> matrices;
    Mat expected = Mat();
    Mat actual = asRowMatrix(matrices, CV_8UC1);
    ASSERT_EQ(expected.empty(), actual.empty());
}

TEST(HelperTest, asRowMatrix_SingleMat) {
    vector<Mat> matrices;
    Mat m0 =  (Mat_<double>(3,1) << 1.0, 2.0, 3.0);

    matrices.push_back(m0);

    Mat expected = (Mat_<double>(1,3) << 1.0, 2.0, 3.0, 4.0);
    Mat actual = asRowMatrix(matrices, matrices[0].type());

    ASSERT_TRUE(isEqual(expected, actual));
}

TEST(HelperTest, asRowMatrix_MultipleMat) {
    vector<Mat> matrices;
    Mat m0 =  (Mat_<double>(2,1) << 1.0, 2.0);
    Mat m1 =  (Mat_<double>(2,1) << 3.0, 4.0);
    matrices.push_back(m0);
    matrices.push_back(m1);
    // {{1.0,2.0},
    //  {3.0,4.0}}
    Mat expected = (Mat_<double>(2,2) << 1.0, 2.0, 3.0, 4.0);
    Mat actual = asRowMatrix(matrices, matrices[0].type());

    ASSERT_TRUE(isEqual(expected, actual));
}

TEST(HelperTest, asRowMatrix_DifferentTypesMat) {
    vector<Mat> matrices;
    Mat m0 =  (Mat_<float>(2,1) << 1.0, 2.0);
    Mat m1 =  (Mat_<int>(2,1) << 3, 4);
    Mat m2 =  (Mat_<unsigned char>(2,1) << 5, 6);

    matrices.push_back(m0);
    matrices.push_back(m1);
    matrices.push_back(m2);

    // {{1.0,2.0},
    //  {3.0,4.0},
    //  {5.0,6.0}}
    Mat expected = (Mat_<double>(3,2) << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    Mat actual = asRowMatrix(matrices, CV_64FC1);

    ASSERT_TRUE(isEqual(expected, actual));
}

TEST(HelperTest, asRowMatrix_DifferentShapesMat) {
    vector<Mat> matrices;
    Mat m0 =  (Mat_<double>(3,1) << 1.0, 2.0, 3.0);
    Mat m1 =  (Mat_<double>(2,1) << 3.0, 4.0);
    Mat m2 =  (Mat_<double>(3,1) << 3.0, 4.0, 1.0);

    matrices.push_back(m0);
    matrices.push_back(m1);
    matrices.push_back(m2);

    ASSERT_THROW(asRowMatrix(matrices, matrices[0].type()), cv::Exception);
}

TEST(HelperTest, asRowMatrix_WrongInputArray) {
    Mat matrices;
    ASSERT_THROW(asRowMatrix(matrices, matrices.type()), cv::Exception);
}

//------------------------------------------------------------------------------
// cv::asColumnMatrix
//------------------------------------------------------------------------------
TEST(HelperTest, asColumnMatrix_EmptyMat) {
    vector<Mat> matrices;
    Mat expected = Mat();
    Mat actual = asColumnMatrix(matrices, CV_8UC1);
    ASSERT_EQ(expected.empty(), actual.empty());
}

TEST(HelperTest, asColumnMatrix_SingleMat) {
    vector<Mat> matrices;
    Mat m0 =  (Mat_<double>(2,1) << 1.0, 2.0);
    matrices.push_back(m0);

    Mat expected = (Mat_<double>(2,1) << 1.0, 2.0);
    Mat actual = asColumnMatrix(matrices, matrices[0].type());

    ASSERT_TRUE(isEqual(expected, actual));
}

TEST(HelperTest, asColumnMatrix_MultipleMat) {
    vector<Mat> matrices;
    Mat m0 =  (Mat_<double>(2,1) << 1.0, 2.0);
    Mat m1 =  (Mat_<double>(2,1) << 3.0, 4.0);
    matrices.push_back(m0);
    matrices.push_back(m1);
    // {{1.0,3.0},
    //  {2.0,4.0}}
    Mat expected = (Mat_<double>(2,2) << 1.0, 3.0, 2.0, 4.0);
    // get data matrix with matrices by row
    Mat actual = asColumnMatrix(matrices, matrices[0].type());
    ASSERT_TRUE(isEqual(expected, actual));
}

TEST(HelperTest, asColumnMatrix_DifferentShapesMat) {
    vector<Mat> matrices;

    Mat m0 =  (Mat_<double>(3,1) << 1.0, 2.0, 3.0);
    Mat m1 =  (Mat_<double>(2,1) << 3.0, 4.0);
    Mat m2 =  (Mat_<double>(3,1) << 1.0, 2.0, 3.0);

    matrices.push_back(m0);
    matrices.push_back(m1);
    matrices.push_back(m2);

    ASSERT_THROW(asColumnMatrix(matrices, matrices[0].type()), cv::Exception);
}

TEST(HelperTest, asColumnMatrix_WrongInputArray) {
    Mat matrices;
    ASSERT_THROW(asColumnMatrix(matrices, matrices.type()), cv::Exception);
}

//------------------------------------------------------------------------------
// cv::writeFileNodeList
//------------------------------------------------------------------------------
TEST(HelperTest, checkReadWriteIntegerList) {
    // test data
    vector<int> IntList0;
    IntList0.push_back(0);
    IntList0.push_back(1);
    IntList0.push_back(2);

    vector<Mat> MatList0;
    MatList0.push_back((Mat_<double>(1,1) << 1.0));
    MatList0.push_back((Mat_<double>(2,2) << 1,2,3,4));

    // open for writing
    FileStorage fs("test.yml", FileStorage::WRITE);
    cv::writeFileNodeList(fs, "IntList", IntList0);
    cv::writeFileNodeList(fs, "MatList", MatList0);
    fs.release();

    vector<int> IntList1;
    vector<Mat> MatList1;

    // open for reading
    fs.open("test.yml", FileStorage::READ);
    cv::readFileNodeList(fs["IntList"], IntList1);
    cv::readFileNodeList(fs["MatList"], MatList1);
    fs.release();

    // assert size
    ASSERT_EQ(IntList0.size(), IntList1.size());
    ASSERT_EQ(MatList0.size(), MatList1.size());

    // assert data
    for(int i=0;i < IntList0.size(); i++)
        ASSERT_EQ(IntList0[i], IntList1[i]);
    for(int i=0;i < MatList0.size(); i++)
        ASSERT_TRUE(isEqual(MatList0[i], MatList1[i]));
}


//------------------------------------------------------------------------------
// cv::histc
//------------------------------------------------------------------------------
TEST(HelperTest, checkHistogramEmpty) {
    Mat mEmpty = Mat_<int>();

    Mat expected = (Mat_<float>(1,4) << 0, 0, 0, 0);
    Mat actual = histc(mEmpty, 0, 3, false);

    cout << actual << endl;
    ASSERT_TRUE(isEqual(expected, actual));
}

TEST(HelperTest, checkHistogramIntMat) {
    Mat mData = (Mat_<int>(2,2) << 1, 1, 2, 3);

    Mat expected = (Mat_<float>(1,4) << 0, 2, 1, 1);
    Mat actual = histc(mData, 0, 3, false);

    ASSERT_TRUE(isEqual(expected, actual));
}

TEST(HelperTest, checkHistogramNegativeValues) {
    Mat mData = (Mat_<int>(2,2) << -1, -1, 0, 1);

    Mat expected = (Mat_<float>(1,4) << 2, 1, 1, 0);
    Mat actual = histc(mData, -1, 2, false);

    ASSERT_TRUE(isEqual(expected, actual));

}

TEST(HelperTest, checkHistogramTypes) {
    Mat mData = (Mat_<unsigned char>(2,2) << 1, 1, 2, 3);

    Mat expected = (Mat_<float>(1,4) << 0, 2, 1, 1);

    ASSERT_TRUE(isEqual(expected, histc(Mat_<char>(mData), 0, 3, false)));
    ASSERT_TRUE(isEqual(expected, histc(Mat_<uchar>(mData), 0, 3, false)));
    ASSERT_TRUE(isEqual(expected, histc(Mat_<short>(mData), 0, 3, false)));
    ASSERT_TRUE(isEqual(expected, histc(Mat_<ushort>(mData), 0, 3, false)));
    ASSERT_TRUE(isEqual(expected, histc(Mat_<int>(mData), 0, 3, false)));
    ASSERT_TRUE(isEqual(expected, histc(Mat_<float>(mData), 0, 3, false)));
}

TEST(HelperTest, checkHistogramDoubleMat) {
    Mat mData = (Mat_<double>(2,2) << 1, 1, 2, 3);

    ASSERT_ANY_THROW(histc(mData, 0, 3, false));
}
