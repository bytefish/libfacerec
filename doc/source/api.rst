libfacerec API
==============

.. highlight:: cpp

.. index:: FaceRecognizer

FaceRecognizer
--------------

All face recognition models are derived from the abstract base class 
``cv::FaceRecognizer``, which provides a unified access to all implemented 
face recognition algorithms.

.. code-block:: cpp

  namespace cv {

  class FaceRecognizer {
  public:

      //! virtual destructor
      virtual ~FaceRecognizer() {}

      // Trains a FaceRecognizer.
      virtual void train(InputArray src, InputArray labels) = 0;

      // Gets a prediction from a FaceRecognizer.
      virtual int predict(InputArray src) = 0;

      // Serializes this object to a given filename.
      virtual void save(const string& filename) const;

      // Deserializes this object from a given filename.
      virtual void load(const string& filename);

      // Serializes this object to a given cv::FileStorage.
      virtual void save(FileStorage& fs) const = 0;

      // Deserializes this object from a given cv::FileStorage.
      virtual void load(const FileStorage& fs) = 0;
  };
  
FaceRecognizer::train(...)
--------------------------

Trains a FaceRecognizer.

.. ocv:function:: void FaceRecognizer::train(InputArray src, InputArray labels)


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
