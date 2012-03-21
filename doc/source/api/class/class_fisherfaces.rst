Fisherfaces Method
==================

Fisherfaces
-----------

.. highlight:: cpp

.. ocv:class:: Fisherfaces

Implements the Fisherfaces Method as described in [Belhumeur97]_. Only the 
model-specific API is explained. ::

  // Belhumeur, P. N., Hespanha, J., and Kriegman, D. "Eigenfaces vs. Fisher-
  // faces: Recognition using class specific linear projection.". IEEE
  // Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997),
  // 711–720.
  class Fisherfaces: public FaceRecognizer {

  public:
      using FaceRecognizer::save;
      using FaceRecognizer::load;

      // Initializes an empty Fisherfaces model.
      Fisherfaces(int num_components = 0);

      // Initializes and computes a Fisherfaces model with images in src and
      // corresponding labels in labels. num_components will be kept for
      // classification.
      Fisherfaces(InputArray src,
              InputArray labels,
              int num_components = 0);

      ~Fisherfaces() { }

      // Computes a Fisherfaces model with images in src and corresponding labels
      // in labels.
      void train(InputArray src, InputArray labels);

      // Predicts the label of a query image in src.
      int predict(InputArray src) const;

      // See cv::FaceRecognizer::load.
      virtual void load(const FileStorage& fs);

      // See cv::FaceRecognizer::save.
      virtual void save(FileStorage& fs) const;

      // Returns the eigenvectors of this Fisherfaces model.
      Mat eigenvectors() const;

      // Returns the eigenvalues of this Fisherfaces model.
      Mat eigenvalues() const;

      // Returns the sample mean of this Fisherfaces model.
      Mat mean() const;

      // Returns the number of components used in this Fisherfaces model.
      int num_components() const;
  };  

Fisherfaces::Fisherfaces
------------------------

Initializes and trains a Fisherfaces model for given data, labels and stores a given number of components.

.. ocv:function:: Fisherfaces::Fisherfaces(int num_components = 0) 
.. ocv:function:: Fisherfaces::Fisherfaces(InputArray src, InputArray labels, int num_components = 0) 

Initializes and trains a Fisherfaces model with images in src and corresponding 
labels in ``labels`` (if given). ``num_components`` number of components are 
kept for classification. If no number of components is given (default 0), it
is automatically determined from given data in :ocv:func:`Fisherfaces::train`.

If (and only if) ``num_components`` <= 0, then ``num_components`` is set to 
(C-1) in ocv:func:`train`, with *C* being the number of unique classes in 
``labels``.

Fisherfaces::save
-----------------

.. ocv:function::  void Fisherfaces::save(const string& filename) const
.. ocv:function::  void Fisherfaces::save(FileStorage& fs) const

See :ocv:func:`FaceRecognizer::save`.

Fisherfaces::load
-----------------

.. ocv:function:: void Fisherfaces::load(const string& filename)
.. ocv:function:: void Fisherfaces::load(const FileStorage& fs)

See :ocv:func:`FaceRecognizer::load`.

Fisherfaces::train
------------------

.. ocv:function:: void Fisherfaces::train(InputArray src, InputArray labels)

See :ocv:func:`FaceRecognizer::train`.

Fisherfaces::predict
--------------------

.. ocv:function:: int Fisherfaces::predict(InputArray src) const

See :ocv:func:`FaceRecognizer::predict`.

Fisherfaces::eigenvalues
------------------------

.. ocv:function:: Mat Fisherfaces::eigenvalues() const

See :ocv:func:`Eigenfaces::eigenvalues`.

Fisherfaces::eigenvectors
-------------------------

.. ocv:function:: Mat Fisherfaces::eigenvectors() const

See :ocv:func:`Eigenfaces::eigenvectors`.

Fisherfaces::mean
-----------------

.. ocv:function:: Mat Fisherfaces::mean() const

See :ocv:func:`Eigenfaces::mean`.

Fisherfaces::num_components
---------------------------

.. ocv:function:: int Fisherfaces::num_components() const

See :ocv:func:`Eigenfaces::num_components`.

