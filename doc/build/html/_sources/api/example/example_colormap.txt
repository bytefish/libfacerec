Working with ColorMap
=====================

Introduction
------------

Applying a colormap on an image is very easy with 
:ocv:class:`colormap::ColorMap`. This page will show you how to use 
colormaps in OpenCV for your images.

Imagine we want to recolor `Lena <http://en.wikipedia.org/wiki/Lenna>`_:

.. image::  /img/tutorial/colormap/lena.jpg

Then recoloring can be done with :ocv:func:`ColorMap::operator()` or by using 
the ColorMap version of :ocv:func:`imwrite` or :ocv:func:`imshow`. Here's a 
sample program, that loads `Lena <http://en.wikipedia.org/wiki/Lenna>`_ and 
applies various colormaps on the image. The results are shown below the code.

Source Code
-----------

.. code-block:: cpp

  #include "opencv2/opencv.hpp"
  #include "opencv2/highgui/highgui.hpp"

  #include "facerec.hpp"

  using namespace cv;
  using namespace std;

  int main(int argc, const char *argv[]) {
      // Read the image. It doesn't matter if you load as grayscale or not.
      Mat img = imread("/home/philipp/lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
      // Use imwrite wrapper that works with ColorMaps.
      string prefix("lena_");
      imwrite(prefix + string("autumn.jpg"), img, colormap::Autumn());
      imwrite(prefix + string("bone.jpg"), img, colormap::Bone());
      imwrite(prefix + string("jet.jpg"), img, colormap::Jet());
      imwrite(prefix + string("winter.jpg"), img, colormap::Winter());
      imwrite(prefix + string("rainbow.jpg"), img, colormap::Rainbow());
      imwrite(prefix + string("ocean.jpg"), img, colormap::Ocean());
      imwrite(prefix + string("summer.jpg"), img, colormap::Summer());
      imwrite(prefix + string("spring.jpg"), img, colormap::Spring());
      imwrite(prefix + string("cool.jpg"), img, colormap::Cool());
      imwrite(prefix + string("hsv.jpg"), img, colormap::HSV());
      imwrite(prefix + string("pink.jpg"), img, colormap::Pink());
      imwrite(prefix + string("hot.jpg"), img, colormap::Hot());
      imwrite(prefix + string("mkpj1.jpg"), img, colormap::MKPJ1());
      imwrite(prefix + string("mkpj2.jpg"), img, colormap::MKPJ2());
      // using a colormap is as simple as doing
      colormap::Jet jet;
      Mat img_jet = jet(img);
      imshow("img_jet0", img_jet);
      // or you can use the imshow wrapper
      imshow("img_jet1", img, colormap::Jet());
      // draw the images
      waitKey(0);
  }

Results
-------

+-----------------------+----------------------------------------------------+
| Colormap Class        | Colormapped Lena                                   |
+=======================+====================================================+
| colormap::Autumn      | .. image:: /img/tutorial/colormap/lena_autumn.jpg  |
+-----------------------+----------------------------------------------------+
| colormap::Bone        | .. image:: /img/tutorial/colormap/lena_bone.jpg    |
+-----------------------+----------------------------------------------------+
| colormap::Cool        | .. image:: /img/tutorial/colormap/lena_cool.jpg    |
+-----------------------+----------------------------------------------------+
| colormap::Hot         | .. image:: /img/tutorial/colormap/lena_hot.jpg     |
+-----------------------+----------------------------------------------------+
| colormap::HSV         | .. image:: /img/tutorial/colormap/lena_hsv.jpg     |
+-----------------------+----------------------------------------------------+
| colormap::Jet         | .. image:: /img/tutorial/colormap/lena_jet.jpg     |
+-----------------------+----------------------------------------------------+
| colormap::MKPJ1       | .. image:: /img/tutorial/colormap/lena_mkpj1.jpg   |
+-----------------------+----------------------------------------------------+
| colormap::MKPJ2       | .. image:: /img/tutorial/colormap/lena_mkpj2.jpg   |
+-----------------------+----------------------------------------------------+
| colormap::Ocean       | .. image:: /img/tutorial/colormap/lena_ocean.jpg   |
+-----------------------+----------------------------------------------------+
| colormap::Pink        | .. image:: /img/tutorial/colormap/lena_pink.jpg    |
+-----------------------+----------------------------------------------------+
| colormap::Rainbow     | .. image:: /img/tutorial/colormap/lena_rainbow.jpg |
+-----------------------+----------------------------------------------------+
| colormap::Spring      | .. image:: /img/tutorial/colormap/lena_spring.jpg  |
+-----------------------+----------------------------------------------------+
| colormap::Summer      | .. image:: /img/tutorial/colormap/lena_summer.jpg  |
+-----------------------+----------------------------------------------------+
| colormap::Winter      | .. image:: /img/tutorial/colormap/lena_winter.jpg  |
+-----------------------+----------------------------------------------------+

And here are the :ocv:func:`imshow` results:

.. image:: /img/tutorial/colormap/lena_imshow_jet.jpg
