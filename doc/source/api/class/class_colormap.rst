Working with ColorMaps
======================

.. highlight:: cpp

imwrite
-------

Provides a wrapper to :ocv:func:`imwrite` for use with a given :ocv:class`ColorMap`.

.. ocv:function:: void imwrite(const string& filename, InputArray img, const colormap::ColorMap& cm, const vector<int>& params = vector<int>())

This makes applying a :ocv:class:`colormap::ColorMap` and storing the result as easy as writing:

.. code-block:: cpp

  imwrite("image.png", img, colormap::Jet());
 
See :ocv:func:`imwrite` for the list of supported formats and flags description.

imshow
------

Provides a wrapper to :ocv:func:`imshow` for use with a given :ocv:class`ColorMap`.

.. ocv:function:: void imshow(const string& winname, InputArray img, const colormap::ColorMap& cm)


This makes applying a :ocv:class:`colormap::ColorMap` and storing the result as easy as writing:

.. code-block:: cpp

  imwshow("image", img, colormap::Jet());

See :ocv:func:`imshow` for the list of supported formats and flags description.

ColorMap
--------

.. ocv:class:: colormap::ColorMap

It's a fact, that human perception isn't built for observing fine changes in grayscale images. Human sensors (eyes and stuff) are more sensitive to observing changes between colors, so you often need to recolor your grayscale images to get a clue about them. :ocv:class:`colormap::ColorMap` is the base class for colormaps in OpenCV. The following colormaps are included in libfacerec and equivalent to their GNU Octave/MATLAB counterparts:

+-----------------------+----------------------------------------------------+
| Class                 | Scale                                              |
+=======================+====================================================+
| colormap::Autumn      | .. image:: /img/colormaps/colorscale_autumn.jpg    |
+-----------------------+----------------------------------------------------+
| colormap::Bone        | .. image:: /img/colormaps/colorscale_bone.jpg      |
+-----------------------+----------------------------------------------------+
| colormap::Cool        | .. image:: /img/colormaps/colorscale_cool.jpg      |
+-----------------------+----------------------------------------------------+
| colormap::Hot         | .. image:: /img/colormaps/colorscale_hot.jpg       |
+-----------------------+----------------------------------------------------+
| colormap::HSV         | .. image:: /img/colormaps/colorscale_hsv.jpg       |
+-----------------------+----------------------------------------------------+
| colormap::Jet         | .. image:: /img/colormaps/colorscale_jet.jpg       |
+-----------------------+----------------------------------------------------+
| colormap::MKPJ1       | .. image:: /img/colormaps/colorscale_mkpj1.jpg     |
+-----------------------+----------------------------------------------------+
| colormap::MKPJ2       | .. image:: /img/colormaps/colorscale_mkpj2.jpg     |
+-----------------------+----------------------------------------------------+
| colormap::Ocean       | .. image:: /img/colormaps/colorscale_ocean.jpg     |
+-----------------------+----------------------------------------------------+
| colormap::Pink        | .. image:: /img/colormaps/colorscale_pink.jpg      |
+-----------------------+----------------------------------------------------+
| colormap::Rainbow     | .. image:: /img/colormaps/colorscale_rainbow.jpg   |
+-----------------------+----------------------------------------------------+
| colormap::Spring      | .. image:: /img/colormaps/colorscale_spring.jpg    |
+-----------------------+----------------------------------------------------+
| colormap::Summer      | .. image:: /img/colormaps/colorscale_summer.jpg    |
+-----------------------+----------------------------------------------------+
| colormap::Winter      | .. image:: /img/colormaps/colorscale_winter.jpg    |
+-----------------------+----------------------------------------------------+


Applying the Jet colormap on a Matrix is then as easy as writing:

.. code-block:: cpp

  colormap::Jet jet; // default: 256-levels
  Mat colored = jet(img);

There's also a the wrapper function for imshow:

.. code-block:: cpp

  imwshow("image", img, colormap::Jet());

And a wrapper to imwrite:

.. code-block:: cpp

  imwrite("image.png", img, colormap::Jet());
  
colormap::ColorMap::init
------------------------

.. ocv:function:: void colormap::ColorMap::init(int n)

colormap::ColorMap::operator()
------------------------------

.. ocv:function:: Mat colormap::Colormap::operator()(InputArray src) const

colormap::ColorMap::linear_colormap(InputArray,InputArray,InputArray,InputArray,int)
------------------------------------------------------------------------------------

.. ocv:function:: Mat colormap::ColorMap::linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, int n) const

colormap::ColorMap::linear_colormap(InputArray,InputArray,InputArray,InputArray,float,float,float)
--------------------------------------------------------------------------------------------------

.. ocv:function:: Mat colormap::ColorMap::linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, float begin, float end, float n) const

colormap::ColorMap::linear_colormap(InputArray,InputArray,InputArray,InputArray,InputArray)
-------------------------------------------------------------------------------------------

.. ocv:function:: Mat linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, InputArray xi) const

