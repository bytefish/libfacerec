#include "decomposition.hpp"
#include "helper.hpp"

void EigenvalueDecomposition::compute(InputArray src) {
    if(cv::isSymmetric(src)) {
        // Fall back to OpenCV for a symmetric matrix!
        cv::eigen(src, _eigenvalues, _eigenvectors);
    } else {
        Mat tmp;
        // Convert the given input matrix to double. Is there any way to
        // prevent allocating the temporary memory? Only used for copying
        // into working memory and deallocated after.
        src.getMat().convertTo(tmp, CV_64FC1);
        // Get dimension of the matrix.
        this->n = tmp.cols;
        // Allocate the matrix data to work on.
        this->H = alloc_2d<double> (n, n);
        // Now safely copy the data.
        for (int i = 0; i < tmp.rows; i++) {
            for (int j = 0; j < tmp.cols; j++) {
                this->H[i][j] = tmp.at<double>(i, j);
            }
        }
        // Deallocates the temporary matrix before computing.
        tmp.release();
        // Performs the eigenvalue decomposition of H.
        compute();
    }
}
