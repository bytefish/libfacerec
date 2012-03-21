libfacerec API
==============

.. highlight:: cpp

.. index:: FaceRecognizer

FaceRecognizer
--------------

.. ocv:class:: FaceRecognizer

All face recognition models in libfacerec are derived from the abstract base 
class :ocv:class:`FaceRecognizer`, which provides a unified access to all the 
libraries implemented face recognition algorithms. ::

  namespace cv {

  class FaceRecognizer {
  public:

      //! virtual destructor
      virtual ~FaceRecognizer() {}

      // Trains a FaceRecognizer.
      virtual void train(InputArray src, InputArray labels) = 0;

      // Gets a prediction from a FaceRecognizer.
      virtual int predict(InputArray src) const = 0;

      // Serializes this object to a given filename.
      virtual void save(const string& filename) const;

      // Deserializes this object from a given filename.
      virtual void load(const string& filename);

      // Serializes this object to a given cv::FileStorage.
      virtual void save(FileStorage& fs) const = 0;

      // Deserializes this object from a given cv::FileStorage.
      virtual void load(const FileStorage& fs) = 0;
  };


FaceRecognizer::~FaceRecognizer
*******************************

.. ocv:function:: FaceRecognizer::~FaceRecognizer()

The destructor of the base class is declared as virtual. So, it is safe to 
write the following code: 

.. code-block:: cpp

    FaceRecongnizer* model;
    if(use_eigenfaces)
        model = new Eigenfaces(... /* Eigenfaces params */);
    else
        model = new LBPH(... /* LBP Histogram params */);
    ...
    delete model;
  
FaceRecognizer::train
*********************

Trains a FaceRecognizer with given data and associated labels.

.. ocv:function:: void FaceRecognizer::train(InputArray src, InputArray labels)

Every model subclassing :ocv:class:`FaceRecognizer` must be able to work with 
image data (``src``) given as a ``vector<Mat>``. This is important, because it's 
impossible to make general assumptions about the dimensionality of input 
samples. The Local Binary Patterns for example process 2D images, while 
Eigenfaces and Fisherfaces method reshape all images in ``src`` to a data 
matrix.

The associated labels in ``labels`` have to be given either in a 1D vector (a 
row or a column) of ``CV_32SC1`` or a ``vector<int>``.

The following example shows how to learn a Fisherfaces model with libfacerec:

.. code-block:: cpp

  // holds images and labels
  vector<Mat> images;
  vector<int> labels;
  // images for first person
  images.push_back(imread("person0/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
  images.push_back(imread("person0/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
  images.push_back(imread("person0/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
  // images for second person
  images.push_back(imread("person1/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
  images.push_back(imread("person1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
  images.push_back(imread("person1/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
  // create a new Fisherfaces model
  Fisherfaces model(images, labels);
  // ... or you could do
  ///Fisherfaces model;
  ///model.train(images,labels);

FaceRecognizer::predict
***********************

.. ocv:function:: int FaceRecognizer::predict(InputArray src) const

Predicts the label for a given query image in ``src``. 

The suffix ``const`` means that prediction does not affect the internal model 
state, so the method can be safely called from within different threads.

The following example shows how to get a prediction from a trained model:

.. code-block:: cpp

  Mat mQuery = imread("person1/3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  int predicted = model.predict(mQuery);

FaceRecognizer::save
********************

Saves a :ocv:class:`FaceRecognizer`` and its model state.

.. ocv:function:: FaceRecognizer::save(const string& filename) const
.. ocv:function:: FaceRecognizer::save(FileStorage& fs) const


Every :ocv:class:`FaceRecognizer` has to overwrite ``FaceRecognizer::save(FileStorage& fs)``
to save the model state. ``FaceRecognizer::save(FileStorage& fs)`` is then 
called by ``FaceRecognizer::save(const string& filename)``, to ease saving a 
model.

The suffix ``const`` means that prediction does not affect the internal model 
state, so the method can be safely called from within different threads.


FaceRecognizer::load
********************

Loads a :ocv:class:`FaceRecognizer` and its model state.

.. ocv:function:: FaceRecognizer::load(const string& filename)
.. ocv:function:: FaceRecognizer::load(FileStorage& fs)

Loads a persisted model and state from a given XML or YAML file . Every 
``FaceRecognizer`` has to overwrite ``FaceRecognizer::load(FileStorage& fs)`` 
to load the model state. ``FaceRecognizer::load(FileStorage& fs)`` in turn gets 
called by ``FaceRecognizer::load(const string& filename)``, to ease saving a 
model.

Eigenfaces
----------

.. ocv:class:: Eigenfaces

Implements the Eigenfaces Method as described in [TP91]_. Only the model-specific 
API is explained. ::

  class Eigenfaces : public FaceRecognizer {

  private:
      int _num_components;
      vector<Mat> _projections;
      vector<int> _labels;
      Mat _eigenvectors;
      Mat _eigenvalues;
      Mat _mean;

  public:
      using FaceRecognizer::save;
      using FaceRecognizer::load;

      // Initializes an empty Eigenfaces model.
      Eigenfaces(int num_components = 0) :
          _num_components(num_components) { }

      // Initializes and computes an Eigenfaces model with images in src and
      // corresponding labels in labels. num_components will be kept for
      // classification.
      Eigenfaces(InputArray src, InputArray labels,
              int num_components = 0) :
          _num_components(num_components) {
          train(src, labels);
      }

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
      Mat eigenvectors() const { return _eigenvectors; }

      // Returns the eigenvalues of this PCA.
      Mat eigenvalues() const { return _eigenvalues; }

      // Returns the sample mean of this PCA.
      Mat mean() const { return _mean; }

      // Returns the number of components used in this PCA.
      int num_components() const { return _num_components; }
  };
  
Eigenfaces::Eigenfaces
**********************

.. ocv:function:: FaceRecognizer::save(FileStorage& fs) const

Eigenfaces::eigenvalues
***********************

Eigenfaces::eigenvectors
************************

Eigenfaces::mean
****************

Eigenfaces::num_components
**************************

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

  private:
      int _num_components;
      Mat _eigenvectors;
      Mat _eigenvalues;
      Mat _mean;
      vector<Mat> _projections;
      vector<int> _labels;

  public:
      using FaceRecognizer::save;
      using FaceRecognizer::load;

      // Initializes an empty Fisherfaces model.
      Fisherfaces(int num_components = 0) :
          _num_components(num_components) {}

      // Initializes and computes a Fisherfaces model with images in src and
      // corresponding labels in labels. num_components will be kept for
      // classification.
      Fisherfaces(const vector<Mat>& src,
              const vector<int>& labels,
              int num_components = 0) :
          _num_components(num_components) {
          train(src, labels);
      }

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
      Mat eigenvectors() const { return _eigenvectors; }

      // Returns the eigenvalues of this Fisherfaces model.
      Mat eigenvalues() const { return _eigenvalues; }

      // Returns the sample mean of this Fisherfaces model.
      Mat mean() const { return _eigenvalues; }

      // Returns the number of components used in this Fisherfaces model.
      int num_components() const { return _num_components; }
  };  

Fisherfaces::Fisherfaces(int num_components = 0)
************************************************

.. ocv:function:: FaceRecognizer::save(FileStorage& fs) const

Fisherfaces::eigenvalues
************************

Fisherfaces::eigenvectors
*************************

Fisherfaces::mean
*****************

Fisherfaces::num_components
***************************


