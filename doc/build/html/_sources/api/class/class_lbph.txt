LBPH
====

.. ocv:class:: LBPH

Implements the Local Binary Patterns Histograms as described in [Ahonen04]_. 
Only the model-specific API is explained. ::

  //  Ahonen T, Hadid A. and Pietikäinen M. "Face description with local binary
  //  patterns: Application to face recognition." IEEE Transactions on Pattern
  //  Analysis and Machine Intelligence, 28(12):2037-2041.
  //
  class LBPH : public FaceRecognizer {

  private:
      int _grid_x;
      int _grid_y;
      int _radius;
      int _neighbors;

      vector<Mat> _histograms;
      vector<int> _labels;

  public:
      using FaceRecognizer::save;
      using FaceRecognizer::load;

      // Initializes this LBPH Model. The current implementation is rather fixed
      // as it uses the Extended Local Binary Patterns per default.
      //
      // radius, neighbors are used in the local binary patterns creation.
      // grid_x, grid_y control the grid size of the spatial histograms.
      LBPH(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8) :
          _grid_x(grid_x),
          _grid_y(grid_y),
          _radius(radius),
          _neighbors(neighbors) {}

      // Initializes and computes this LBPH Model. The current implementation is
      // rather fixed as it uses the Extended Local Binary Patterns per default.
      //
      // (radius=1), (neighbors=8) are used in the local binary patterns creation.
      // (grid_x=8), (grid_y=8) controls the grid size of the spatial histograms.
      LBPH(InputArray src,
              InputArray labels,
              int radius=1, int neighbors=8,
              int grid_x=8, int grid_y=8) :
                  _grid_x(grid_x),
                  _grid_y(grid_y),
                  _radius(radius),
                  _neighbors(neighbors) {
          train(src, labels);
      }

      ~LBPH() { }

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
      int neighbors() const { return _neighbors; }
      int radius() const { return _radius; }
      int grid_x() const { return _grid_x; }
      int grid_y() const { return _grid_y; }

  };

LBPH::LBPH(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8)
---------------------------------------------------------------------


LBPH::LBPH(InputArray src, InputArray labels, int radius=1, int neighbors=8, int grid_x=8, int grid_y=8)
--------------------------------------------------------------------------------------------------------

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
