# Description

[libfacerec](http://www.github.com/bytefish/libfacerec) is a library for face recognition in OpenCV. There are no additional dependencies to build the library. The Eigenfaces, Fisherfaces method and Local Binary Patterns Histograms (LBPH) are implemented and most parts of the library are covered by unit tests. 

# Issues and Feature Requests

This project now open for bug reports and feature requests. Please note: the bug tracker is not the right place to ask for implementation details or pose questions on algorithmic theory, such questions be closed.

# Tutorial

The documentation of the library comes with an extensive API description and carefully designed tutorials. It is available in the `doc/build` folder coming with this project. If you want to compile the documentation yourself, then switch to the folder `doc` and run `make <target>`. For the html version you would `make html` and for the PDF version `make latexpdf`. You'll need [Sphinx](http://sphinx.pocoo.org) for this. 

# Windows Installation

Some people had problems getting the library to run on Windows. I've checked the latest revision with the OpenCV Superpack 2.3.1 and here'

Download the OpenCV 2.3.1 superpack

Append the following Path
<code>
;C:\opencv\build\x86\vc10\bin; C:\opencv\build\common\tbb\ia32\vc10
</code>
# Literature

* Eigenfaces (Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.)
* Fisherfaces (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection.". IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.)
* Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)

