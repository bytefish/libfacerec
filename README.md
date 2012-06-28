# Description

[libfacerec](http://www.github.com/bytefish/libfacerec) is a library for face recognition in OpenCV. It has been merged into OpenCV 2.4 (contrib module) and both implementations are synchronized. So if you are in (a recent) OpenCV 2.4: There is no need to compile libfacerec yourself, you have everything to get started. Note: Make sure to work on a recent OpenCV revision, if you want to be compatible with the very latest libfacerec version.

The library comes with an extensive documentation, which can be found at

* [http://docs.opencv.org/trunk/modules/contrib/doc/facerec/index.html](http://docs.opencv.org/trunk/modules/contrib/doc/facerec/index.html)
  
The documentation includes:

* [The API (cv::FaceRecognizer)](http://docs.opencv.org/trunk/modules/contrib/doc/facerec/facerec_api.html)
* [Guide to Face Recognition with OpenCV](http://docs.opencv.org/trunk/modules/contrib/doc/facerec/facerec_tutorial.html)
* [Tutorial on Gender Classification](http://docs.opencv.org/trunk/modules/contrib/doc/facerec/tutorial/facerec_gender_classification.html)
* **[Face Recognition in Videos](http://docs.opencv.org/trunk/modules/contrib/doc/facerec/tutorial/facerec_video_recognition.html)**

There are no additional dependencies to build the library. The Eigenfaces, Fisherfaces method and Local Binary Patterns Histograms (LBPH) are implemented and most parts of the library are covered by unit tests. As of OpenCV 2.4+ this library has been merged into the OpenCV contrib module, so if you are using OpenCV 2.4+ you can [start right away](http://code.opencv.org/projects/opencv/repository/entry/trunk/opencv/samples/cpp/facerec_demo.cpp). 

Again note: This library is included in the contrib module of OpenCV.


# Issues and Feature Requests

This project is now open for bug reports and feature requests.

# Building the library with Microsoft Visual Studio 2008/2010

If you have problems with building libfacerec with Microsoft Visual Studio 2008/2010, then please read my blog post at:

* [http://www.bytefish.de/blog/opencv_visual_studio_and_libfacerec](http://www.bytefish.de/blog/opencv_visual_studio_and_libfacerec)

This is based on version 0.04 of the libfacerec, available here:

* [https://github.com/bytefish/libfacerec/zipball/v0.04](https://github.com/bytefish/libfacerec/zipball/v0.04)
  
# Literature

* Eigenfaces (Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.)
* Fisherfaces (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection.". IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.)
* Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)

