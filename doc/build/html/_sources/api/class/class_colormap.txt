ColorMaps in OpenCV
===================

.. highlight:: cpp

It's a fact, that human perception isn't built for observing fine changes in grayscale images. Human sensors (eyes and stuff) are more sensitive to observing changes between colors, so you often need to recolor your grayscale images to get a clue about them. :ocv:class:`colormap::ColorMap` is the base class for colormaps in OpenCV. The following colormaps are included in libfacerec, feel free to add your own:

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

Applying the ``colormap::Jet`` on a given image ``img`` is then as easy as writing:

.. code-block:: cpp

  colormap::Jet jet;
  Mat colored = jet(img);

There's also a the wrapper function for imshow:

.. code-block:: cpp

  imwshow("image", img, colormap::Jet());

And a wrapper to imwrite:

.. code-block:: cpp

  imwrite("image.png", img, colormap::Jet());

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

colormap::ColorMap
------------------

.. ocv:class:: colormap::ColorMap

.. code-block:: cpp

  namespace colormap {

    class ColorMap {

    public:

        // Applies the colormap on a given image.
        Mat operator()(InputArray src) const;

        // Setup base map to interpolate from.
        virtual void init(int n) = 0;

        // Interpolates from a base colormap.
        static Mat linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, int n);

        // Interpolates from a base colormap.
        static Mat linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, float begin, float end, float n);

        // Interpolates from a base colormap.
        static Mat linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, InputArray xi);
    };
  }
  

colormap::ColorMap::init
------------------------

Initializes the lookup table for this ColorMap (must be overriden by a derived class).

.. ocv:function:: void colormap::ColorMap::init(int n)

colormap::ColorMap::operator()
------------------------------

Applies this ColorMap on a given image.

.. ocv:function:: Mat colormap::Colormap::operator()(InputArray src) const

colormap::ColorMap::linear_colormap
-----------------------------------

Returns a linear interpolated colormap.

.. ocv:function:: Mat colormap::ColorMap::linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, int n) const

* ``X`` Points corresponding to a color value in ``r``, ``g`` and ``b``.
* ``r``, ``g``, ``b`` Red, Green, Blue value.
* ``n`` Number of points to interpolate (determines how smooth the colormap is).

.. ocv:function:: Mat colormap::ColorMap::linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, float begin, float end, float n) const

* ``begin`` Interpolation start.
* ``end`` Interpolation end.

.. ocv:function:: Mat linear_colormap(InputArray X, InputArray r, InputArray g, InputArray b, InputArray xi) const

* ``xi`` Interpolation points.
