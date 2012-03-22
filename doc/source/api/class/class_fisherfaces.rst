Fisherfaces Method
==================

.. highlight:: cpp

The Linear Discriminant Analysis (LDA) is a class-specific dimensionality reduction. It was invented by the great statistician `Sir R. A. Fisher <http://en.wikipedia.org/wiki/Ronald_Fisher>`_, who successfully used it for classifying flowers in his 1936 paper *"The use of multiple measurements in taxonomic problems"* ([Fisher36]_). 

A Linear Discriminant Analysis is closely related to a Principal Component Analysis. The PCA finds a linear combination of features that maximizes the total variance in data. While this is clearly a powerful way to represent data, it doesn't consider any classes and so a lot of discriminative information *may* be lost when throwing some components away. This *can* yield bad results, especially when it comes to classification. So in order to find a combination of features that separates best between classes, the Linear Discriminant Analysis instead maximizes the ratio of between-classes to within-classes scatter. The idea is, that same classes should cluster tightly together.

This was also recognized by `Belhumeur <http://www.cs.columbia.edu/~belhumeur/>`_, `Hespanha <http://www.ece.ucsb.edu/~hespanha/>`_ and `Kriegman <http://cseweb.ucsd.edu/~kriegman/>`_ and so they applied a Discriminant Analysis to face recognition in their paper *"Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection"* ([Belhumeur97]_). The Eigenfaces approach by Pentland and Turk as described in "Eigenfaces for Recognition" ([TP91]_) was a revolutionary one, but the original paper already discusses the negative effects of images with changes in background, light and perspective. So on datasets with differences in the setup, the Principal Component Analysis is likely to find the wrong components for classification and can perform poorly.

This class implements the Fisherfaces method as described in [Belhumeur97]_.

Please see the :doc:`API Examples page </api/examples>` for example programs.

Fisherfaces
-----------

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
      Fisherfaces(InputArray src, InputArray labels, int num_components = 0);

      ~Fisherfaces() {}

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

