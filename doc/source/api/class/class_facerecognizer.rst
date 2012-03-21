FaceRecognizer (Abstract Base Class)
====================================

.. highlight:: cpp

FaceRecognizer
--------------

.. ocv:class:: FaceRecognizer

All face recognition models in libfacerec are derived from the abstract base 
class :ocv:class:`FaceRecognizer`, which provides a unified access to all face 
recongition algorithms. ::

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
-------------------------------

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
---------------------

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
-----------------------

.. ocv:function:: int FaceRecognizer::predict(InputArray src) const

Predicts the label for a given query image in ``src``. 

The suffix ``const`` means that prediction does not affect the internal model 
state, so the method can be safely called from within different threads.

The following example shows how to get a prediction from a trained model:

.. code-block:: cpp

  Mat mQuery = imread("person1/3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  int predicted = model.predict(mQuery);

FaceRecognizer::save
--------------------

Saves a :ocv:class:`FaceRecognizer` and its model state.

.. ocv:function:: void FaceRecognizer::save(const string& filename) const
.. ocv:function:: void FaceRecognizer::save(FileStorage& fs) const


Every :ocv:class:`FaceRecognizer` has to overwrite ``FaceRecognizer::save(FileStorage& fs)``
to save the model state. ``FaceRecognizer::save(FileStorage& fs)`` is then 
called by ``FaceRecognizer::save(const string& filename)``, to ease saving a 
model.

The suffix ``const`` means that prediction does not affect the internal model 
state, so the method can be safely called from within different threads.

FaceRecognizer::load
--------------------

Loads a :ocv:class:`FaceRecognizer` and its model state.

.. ocv:function:: void FaceRecognizer::load(const string& filename)
.. ocv:function:: void FaceRecognizer::load(FileStorage& fs)

Loads a persisted model and state from a given XML or YAML file . Every 
:ocv:class:`FaceRecognizer` has to overwrite ``FaceRecognizer::load(FileStorage& fs)`` 
to enable loading the model state. ``FaceRecognizer::load(FileStorage& fs)`` in 
turn gets called by ``FaceRecognizer::load(const string& filename)``, to ease 
saving a model.
