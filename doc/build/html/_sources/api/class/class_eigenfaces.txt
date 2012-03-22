Eigenfaces Method
=================

.. highlight:: cpp

The Eigenfaces Method is basically a Principal Component Analysis (PCA) with 
1-Nearest Neighbor classifier. So what is a PCA? The PCA was independently 
proposed by `Karl Pearson <http://en.wikipedia.org/wiki/Karl_Pearson>`_ (1901) 
and `Harold Hotelling <http://en.wikipedia.org/wiki/Harold_Hotelling>`_ (1933) 
to turn a set of possibly correlated variables into a smaller set of 
uncorrelated variables. The idea is, that a high-dimensional dataset is often 
described by correlated variables and therefore only a few meaningful dimensions 
account for most of the information. The PCA method finds the directions with 
the greatest variance in the data, called principal components.

Please see the :doc:`API Examples page </api/examples>` for example programs.

Eigenfaces
----------

.. ocv:class:: Eigenfaces

Implements the Eigenfaces Method as described in [TP91]_. Only the model-specific 
API is explained. ::

  class Eigenfaces : public FaceRecognizer {

  public:
      using FaceRecognizer::save;
      using FaceRecognizer::load;

      // Initializes an empty Eigenfaces model.
      Eigenfaces(int num_components = 0);

      // Initializes and computes an Eigenfaces model with images in src and
      // corresponding labels in labels. num_components will be kept for
      // classification.
      Eigenfaces(InputArray src, InputArray labels,
              int num_components = 0);

      // Computes an Eigenfaces model with images in src and corresponding labels
      // in labels.
      void train(InputArray src, InputArray labels);

      // Predicts the label of a query image in src.
      int predict(const InputArray src) const;

      // See cv::FaceRecognizer::load.
      void load(const FileStorage& fs);

      // See cv::FaceRecognizer::save.
      void save(FileStorage& fs) const;

      // Returns the eigenvectors of this PCA.
      Mat eigenvectors() const;

      // Returns the eigenvalues of this PCA.
      Mat eigenvalues() const;

      // Returns the sample mean of this PCA.
      Mat mean() const;

      // Returns the number of components used in this PCA.
      int num_components() const;
  };
  
Eigenfaces::Eigenfaces
----------------------

Initializes and learns an Eigenfaces model for given data, labels and stores a given number of components.

.. ocv:function:: Eigenfaces::Eigenfaces(int num_components = 0)
.. ocv:function:: Eigenfaces::Eigenfaces(InputArray src, InputArray labels, int num_components = 0) 

Initializes and trains an Eigenfaces model with images in src and corresponding 
labels in ``labels`` (if given). ``num_components`` number of components are 
kept for classification. If no number of components is given (default 0), it is 
automatically determined from given data in :ocv:func:`Eigenfaces::train`.

If (and only if) ``num_components`` <= 0, then ``num_components`` is set to 
(N-1) in ocv:func:`Eigenfaces::train`, with *N* being the total number of 
samples in ``src``.


Eigenfaces::save
----------------

.. ocv:function::  void Eigenfaces::save(const string& filename) const
.. ocv:function::  void Eigenfaces::save(FileStorage& fs) const

See :ocv:func:`FaceRecognizer::save`.

Eigenfaces::load
----------------
.. ocv:function:: void Eigenfaces::load(const string& filename)
.. ocv:function:: void Eigenfaces::load(const FileStorage& fs)

See :ocv:func:`FaceRecognizer::load`.

Eigenfaces::train
-----------------

.. ocv:function:: void Eigenfaces::train(InputArray src, InputArray labels)

See :ocv:func:`FaceRecognizer::train`.

Eigenfaces::predict
-------------------

.. ocv:function:: int Eigenfaces::predict(InputArray src) const

See :ocv:func:`FaceRecognizer::predict`.

Eigenfaces::eigenvalues
-----------------------

Returns the eigenvalues corresponding to each of the eigenvectors.

.. ocv:function:: Mat Eigenfaces::eigenvalues() const

Regarding the data alignment, the eigenvalues are stored in a 1D vector as row. 
They are sorted in a descending order.


Eigenfaces::eigenvectors
------------------------

Returns the eigenvectors of this model.

.. ocv:function:: Mat Eigenfaces::eigenvectors() const

Regarding the data alignment, the i-th eigenvectors is stored in the i-th column 
of this matrix. The eigenvectors are sorted in a descending order by their 
eigenvalue.

Eigenfaces::mean
----------------

Returns the sample mean of this model.

.. ocv:function:: Mat Eigenfaces::mean() const

The mean is stored as a 1D vector in a row.

Eigenfaces::num_components
--------------------------

Returns the number of components (number of Eigenfaces) used for classification.

.. ocv:function:: int Eigenfaces::num_components() const

This number may be 0 for initialized objects. It may be set during the training.



