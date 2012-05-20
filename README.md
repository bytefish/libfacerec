# Description

[libfacerec](http://www.github.com/bytefish/libfacerec) is a library for face recognition in OpenCV. There are no additional dependencies to build the library. The Eigenfaces, Fisherfaces method and Local Binary Patterns Histograms (LBPH) are implemented and most parts of the library are covered by unit tests. 

# Issues and Feature Requests

This project is now open for bug reports and feature requests.

# Tutorial

The documentation of the library comes with an extensive API description and carefully designed tutorials. It is available in the `doc/build` folder coming with this project. If you want to compile the documentation yourself, then switch to the folder `doc` and run `make <target>`. For the html version you would `make html` and for the PDF version `make latexpdf`. You'll need [Sphinx](http://sphinx.pocoo.org) for this. 

# Building the library with Microsoft Visual Studio 2008/2010

If you have problems with building libfacerec with Microsoft Visual Studio 2008/2010, then please read my blog post at:

* [http://www.bytefish.de/blog/opencv_visual_studio_and_libfacerecs](http://www.bytefish.de/blog/opencv_visual_studio_and_libfacerec)

# Literature

* Eigenfaces (Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.)
* Fisherfaces (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection.". IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.)
* Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)

