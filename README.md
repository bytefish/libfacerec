# Description

[libfacerec](http://www.github.com/bytefish/libfacerec) is a library for face recognition in OpenCV. There are no additional dependencies to build the library. The Eigenfaces, Fisherfaces method and Local Binary Patterns Histograms (LBPH) are implemented and most parts of the library are covered by unit tests. As of OpenCV 2.4+ this library has been merged into the OpenCV contrib module, so if you are using OpenCV 2.4+ you can [start right away](http://code.opencv.org/projects/opencv/repository/entry/trunk/opencv/samples/cpp/facerec_demo.cpp). 

Right now I am in the process of synchronizing OpenCV 2.4 and libfacerec, so be careful with the current master branch. This code is only supported from OpenCV2.4 on. Note: if you are using OpenCV 2.3, then use the stable 0.04 release of libfacerec:

* [https://github.com/bytefish/libfacerec/zipball/v0.04](https://github.com/bytefish/libfacerec/zipball/v0.04)

**I can't stress this enough, because with these changes I see myself reading through many, many mails already. If you are using OpenCV 2.3, then please use libfacerec 0.04.** I'll merge the most important bugfixes (such as numerical errors) back into the 0.04 branch, so you are on the safe side. 

Why am I doing all this? Quoting from my website:

```
During the next days I am going to synchronize the libfacerec implementation and 
the face recognition implementation I have contributed to OpenCV. I am going to 
tag the latest libfacerec version in github, because it is going to be the last 
version compatible to OpenCV 2.3. The new implementation is going to make use of 
the new cv::Algorithm base class, to wrap all non-trivial functionality (this class
 is available since OpenCV 2.4).

The big advantadge for you as user is:

   * A simple, but rich interface, with access to all model internals.
   * A thorough documentation, which the project is somewhat lacking right now.

Once OpenCV 2.4+ is shipped with libfacerec - it actually is already! - there is no 
need for you to compile libfacerec anymore. You can directly use OpenCV, as both are 
going to be based on the same implementation. All this is going to clear a lot confusion 
on the user side, which the current implementations may have caused. For me as a 
developer, there are several advantadges of synchronizing both. The most important 
is, that I only need to maintain a single version of the code and I don't force 
myself anymore to support OpenCV versions as early as OpenCV 2.3.

I'll need to make some minor modifications to the libfacerec API (the current OpenCV API 
is not going to change), but all this is only to make the algorithm even easier for you. 
The modifications are going to take some time, as I have to refactor the classes, the 
tests, the documentation and make sure everything's working as expected. 
```

# Issues and Feature Requests

This project is now open for bug reports and feature requests.

# Tutorial

The documentation of the library comes with an extensive API description and carefully designed tutorials. It is available in the `doc/build` folder coming with this project. If you want to compile the documentation yourself, then switch to the folder `doc` and run `make <target>`. For the html version you would `make html` and for the PDF version `make latexpdf`. You'll need [Sphinx](http://sphinx.pocoo.org) for this. 

# Building the library with Microsoft Visual Studio 2008/2010

If you have problems with building libfacerec with Microsoft Visual Studio 2008/2010, then please read my blog post at:

* [http://www.bytefish.de/blog/opencv_visual_studio_and_libfacerec](http://www.bytefish.de/blog/opencv_visual_studio_and_libfacerec)

This is based on version 0.04 of the libfacerec, available here:

* [https://github.com/bytefish/libfacerec/zipball/v0.04](https://github.com/bytefish/libfacerec/zipball/v0.04)
  
# Literature

* Eigenfaces (Turk, M., and Pentland, A. "Eigenfaces for recognition.". Journal of Cognitive Neuroscience 3 (1991), 71–86.)
* Fisherfaces (Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisherfaces: Recognition using class specific linear projection.". IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.)
* Local Binary Patterns Histograms (Ahonen, T., Hadid, A., and Pietikainen, M. "Face Recognition with Local Binary Patterns.". Computer Vision - ECCV 2004 (2004), 469–481.)

