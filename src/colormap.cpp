#include "colormap.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

//------------------------------------------------------------------------------
// cv::colormap::ColorMap
//------------------------------------------------------------------------------

Mat cv::colormap::ColorMap::operator()(InputArray _src) const {
    if(_lut.total() != 256) {
        CV_Error(CV_StsAssert, "cv::LUT only supports tables of size 256.");
    }
    Mat src = _src.getMat();
    // Return original matrix if wrong type is given (is fail loud better here?)
    if(src.type() != CV_8UC1 && src.type() != CV_8UC3)
        return src;
    // Turn into a BGR matrix into its grayscale representation.
    if(src.type() == CV_8UC3)
        cvtColor(src.clone(), src, CV_BGR2GRAY);
    cvtColor(src.clone(), src, CV_GRAY2BGR);
    // Apply the ColorMap.
    LUT(src.clone(), _lut, src);
    return src;
}


Mat cv::colormap::ColorMap:: linear_colormap(InputArray X,
        InputArray r, InputArray g, InputArray b,
        InputArray xi) {
    Mat lut;
    Mat planes[] = {
            interp1(X, b, xi),
            interp1(X, g, xi),
            interp1(X, r, xi)};
    merge(planes, 3, lut);
    return lut;
}

//------------------------------------------------------------------------------
// cv::imwrite
//------------------------------------------------------------------------------
void cv::imwrite(const string& filename, InputArray img,
        const colormap::ColorMap& cm,
        const vector<int>& params) {
    Mat tmp = img.getMat().clone();
    tmp = cm(tmp);
    tmp.convertTo(tmp, CV_8UC3, 255.);
    imwrite(filename, tmp, params);
}

//------------------------------------------------------------------------------
// cv::imshow
//------------------------------------------------------------------------------
void cv::imshow(const string& winname, InputArray img,
        const colormap::ColorMap& cm) {
    Mat tmp = img.getMat().clone();
    tmp = cm(tmp);
    tmp.convertTo(tmp, CV_8UC3, 255.);
    imshow(winname, tmp);
}
