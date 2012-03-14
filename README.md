# Warning

This library is still **Work in Progress**. No bug reports in these early development stages please. Expect major interface changes coming.

# Description

[libfacerec](http://www.github.com/bytefish/libfacerec) is a header-only library for face recognition in OpenCV. Using it as simple as adding it to your include path, there are no additional dependencies. The Eigenfaces, Fisherfaces method and Local Binary Patterns Histograms (LBPH) are implemented and most parts of the library are covered by Unit Tests. 

Please see [src/main.cpp](https://github.com/bytefish/libfacerec/blob/master/src/main.cpp) or the [tests](https://github.com/bytefish/libfacerec/tree/master/test) to get a feeling for the API. I am working on documentation & tutorials right now. If you want to run [the example](https://github.com/bytefish/libfacerec/blob/master/src/main.cpp), you'll need a CSV file with lines composed of a _filename_ followed by a _;_ followed by the _label_ (as **integer number**), making up a line like this: `/path/to/image.ext;0` ([read my notes on this here](http://www.bytefish.de/blog/fisherfaces_in_opencv)). 

# Tutorial

Coming soon!

# Literature

* Eigenfaces (Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.)
* Fisherfaces (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection.". IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.)
* Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)

