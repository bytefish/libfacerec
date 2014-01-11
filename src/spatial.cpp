#include "spatial.hpp"

using namespace cv;

Mat libfacerec::SpatialHistogramRecognizer::spatial_histogram(InputArray _src) const {
    Mat src = _src.getMat();
    if(src.empty())
        return Mat();

    // calculate patch size
    int width = src.cols/_grid_x;
    int height = src.rows/_grid_y;

    Mat result = Mat::zeros(0, 0, hist_type);
    // iterate through grid
    for(int i = 0; i < _grid_y; i++) {
        for(int j = 0; j < _grid_x; j++) {
            Mat src_cell(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
            Mat hist = Mat::zeros(1,hist_len,hist_type);

            oper(src_cell,hist);

            result.push_back(hist);
        }
    }
    return result;
}

// Reads a sequence from a FileNode::SEQ with type _Tp into a result vector.
template<typename _Tp>
inline void readFileNodeList(const FileNode& fn, std::vector<_Tp>& result) {
    if (fn.type() == FileNode::SEQ) {
        for (FileNodeIterator it = fn.begin(); it != fn.end();) {
            _Tp item;
            it >> item;
            result.push_back(item);
        }
    }
}

// Writes the a list of given items to a cv::FileStorage.
template<typename _Tp>
inline void writeFileNodeList(FileStorage& fs, const cv::String& name, const std::vector<_Tp>& items) {
    // typedefs
    typedef typename std::vector<_Tp>::const_iterator constVecIterator;
    // write the elements in item to fs
    fs << name << "[";
    for (constVecIterator it = items.begin(); it != items.end(); ++it) {
        fs << *it;
    }
    fs << "]";
}
void libfacerec::SpatialHistogramRecognizer::load(const FileStorage& fs) {
    fs["grid_x"] >> _grid_x;
    fs["grid_y"] >> _grid_y;
    //read matrices
    readFileNodeList(fs["histograms"], _histograms);
    fs["labels"] >> _labels;

}

// See FaceRecognizer::save.
void libfacerec::SpatialHistogramRecognizer::save(FileStorage& fs) const {
    fs << "grid_x" << _grid_x;
    fs << "grid_y" << _grid_y;
    // write matrices
    writeFileNodeList(fs, "histograms", _histograms);
    fs << "labels" << _labels;
}

void libfacerec::SpatialHistogramRecognizer::train(InputArrayOfArrays _in_src, InputArray _in_labels) {
    this->train(_in_src, _in_labels, false);
}

void libfacerec::SpatialHistogramRecognizer::update(InputArrayOfArrays _in_src, InputArray _in_labels) {
    // got no data, just return
    if(_in_src.total() == 0)
        return;

    this->train(_in_src, _in_labels, true);
}

void libfacerec::SpatialHistogramRecognizer::train(InputArrayOfArrays _in_src, InputArray _in_labels, bool preserveData) {
    if(_in_src.kind() != _InputArray::STD_VECTOR_MAT && _in_src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        std::string error_message = "The images are expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
        CV_Error(CV_StsBadArg, error_message);
    }
    if(_in_src.total() == 0) {
        std::string error_message = format("Empty training data was given. You'll need more than one sample to learn a model.");
        CV_Error(CV_StsUnsupportedFormat, error_message);
    } else if(_in_labels.getMat().type() != CV_32SC1) {
        std::string error_message = format("Labels must be given as integer (CV_32SC1). Expected %d, but was %d.", CV_32SC1, _in_labels.type());
        CV_Error(CV_StsUnsupportedFormat, error_message);
    }
    // get the vector of matrices
    std::vector<Mat> src;
    _in_src.getMatVector(src);
    // get the label matrix
    Mat labels = _in_labels.getMat();
    // check if data is well- aligned
    if(labels.total() != src.size()) {
        std::string error_message = format("The number of samples (src) must equal the number of labels (labels). Was len(samples)=%d, len(labels)=%d.", src.size(), _labels.total());
        CV_Error(CV_StsBadArg, error_message);
    }
    // if this model should be trained without preserving old data, delete old model data
    if(!preserveData) {
        _labels.release();
        _histograms.clear();
    }
    // append labels to _labels matrix
    for(size_t labelIdx = 0; labelIdx < labels.total(); labelIdx++) {
        _labels.push_back(labels.at<int>((int)labelIdx));
    }
    // calculate and store the spatial histograms of the original data
    for(size_t sampleIdx = 0; sampleIdx < src.size(); sampleIdx++) {
        Mat p = spatial_histogram( src[sampleIdx] );
        //normalize(p,p);
        if ( ! p.empty() )
        _histograms.push_back(p);
    }
//    std::cout << _histograms.size() << " * " << _histograms[0].total() << " = " << (_histograms.size() * _histograms[0].total()) << " elems." << std::endl;;
}

void libfacerec::SpatialHistogramRecognizer::predict(InputArray _src, int &minClass, double &minDist) const {
    if(_histograms.empty()) {
        // throw error if no data (or simply return -1?)
        std::string error_message = "This SpatialHistogramReco model is not computed yet. Did you call the train method?";
        CV_Error(CV_StsBadArg, error_message);
    }
    Mat src = _src.getMat();
    // get the spatial histogram from input image
    Mat query = spatial_histogram( src );
    // find 1-nearest neighbor
    minDist = DBL_MAX;
    minClass = -1;
    for(size_t sampleIdx = 0; sampleIdx < _histograms.size(); sampleIdx++) {
        // call vmethod!
        double dist = this->distance(_histograms[sampleIdx], query);

        if((dist < minDist) && (dist < _threshold)) {
            minDist = dist;
            minClass = _labels.at<int>((int) sampleIdx);
        }
    }
}

int libfacerec::SpatialHistogramRecognizer::predict(InputArray _src) const {
    int label;
    double dummy;
    predict(_src, label, dummy);
    return label;
}

AlgorithmInfo * libfacerec::SpatialHistogramRecognizer::info() const { return 0; } // dummy for now, as it's a total pita to reconstruct the resp. macro outside opencv.