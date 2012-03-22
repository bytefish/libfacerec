Local Binary Patterns Histograms (LBPH)
=======================================

.. highlight:: cpp

Please see the :doc:`API Examples page </api/examples>` for example programs.

LBPH
----

.. ocv:class:: LBPH

Implements the Local Binary Patterns Histograms as described in [Ahonen04]_. 
Only the model-specific API is explained. ::

  //  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
  //  patterns: Application to face recognition." IEEE Transactions on Pattern
  //  Analysis and Machine Intelligence, 28(12):2037-2041.
  //
  class LBPH : public FaceRecognizer {

  public:
      using FaceRecognizer::save;
      using FaceRecognizer::load;

      // Initializes this LBPH Model. The current implementation is rather fixed
      // as it uses the Extended Local Binary Patterns per default.
      //
      // radius, neighbors are used in the local binary patterns creation.
      // grid_x, grid_y control the grid size of the spatial histograms.
      LBPH(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8);

      // Initializes and computes this LBPH Model. The current implementation is
      // rather fixed as it uses the Extended Local Binary Patterns per default.
      //
      // (radius=1), (neighbors=8) are used in the local binary patterns creation.
      // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
      LBPH(InputArray src, InputArray labels, int radius=1, int neighbors=8, int grid_x=8, int grid_y=8);
      
      // Destructor.
      ~LBPH() {}

      // Computes a LBPH model with images in src and
      // corresponding labels in labels.
      void train(InputArray src, InputArray labels);

      // Predicts the label of a query image in src.
      int predict(InputArray src) const;

      // See cv::FaceRecognizer::load.
      void load(const FileStorage& fs);

      // See cv::FaceRecognizer::save.
      void save(FileStorage& fs) const;

      // Getter functions.
      int neighbors() const;
      int radius() const;
      int grid_x() const;
      int grid_y() const;

  };

LBPH::LBPH
----------

.. ocv:function:: LBPH::LBPH(InputArray src, InputArray labels, int radius=1, int neighbors=8, int grid_x=8, int grid_y=8)
.. ocv:function:: LBPH::LBPH(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8)


LBPH::save
----------

See :ocv:func:`FaceRecognizer::save`.

LBPH::load
----------

See :ocv:func:`FaceRecognizer::load`.

LBPH::train
-----------

.. ocv:function:: void train(InputArray src, InputArray labels)

See :ocv:func:`FaceRecognizer::train`.

LBPH::predict
-------------

.. ocv:function:: int predict(InputArray src) const

See :ocv:func:`FaceRecognizer::predict`.

LBPH::neighbors
---------------

.. ocv:function:: int LBPH::neighbors() const

LBPH::radius
------------

.. ocv:function:: int LBPH::radius() const

LBPH::grid_x
------------

.. ocv:function:: int LBPH::grid_x() const

LBPH::grid_y
------------

.. ocv:function:: int LBPH::grid_y() const
