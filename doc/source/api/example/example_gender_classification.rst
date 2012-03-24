Gender Classification with OpenCV (and libfacerec)
==================================================

Introduction
------------

You want to decide wether a given face is *male* or *female*. In this tutorial you'll learn how to perform a gender classification with OpenCV and libfacerec. 

Prerequisites
-------------

First of all you'll need some sample images of male and female faces. I've decided to search faces of celebrities using `Google Images <http://www.google.com/images>`_ with the faces filter turned on (my god, they have great algorithms at `Google <http://www.google.com>`_!). My database has 8 male and 5 female subjects, each with 10 images. Here are the names, if you don't know who to search:

* Angelina Jolie
* Arnold Schwarzenegger
* Brad Pitt
* Emma Watson
* George Clooney
* Jennifer Lopez
* Johnny Depp
* Justin Timberlake
* Katy Perry
* Keanu Reeves
* Naomi Watts
* Patrick Stewart
* Tom Cruise

All images were chosen to have a frontal face perspective, were aligned at the eyes and have been cropped to equal size, just like this set of George Clooney images:

.. image:: /img/tutorial/gender_classification/clooney.png

Choice of Algorithm
-------------------

If we want to decide wether a person is *male* or *female*, we must use a class-specific method to learn the discriminative features of both classes. The Eigenfaces method is based on the Principal ComponentAnalysis, an unsupervised statistical model not suitable for this task. The Fisherfaces method yields a class-specific linear projection, so it is much better suited for a gender classification task. A detailed writeup on gender classification with the Fisherfaces method is given at `<http://www.bytefish.de/blog/gender_classification>`_. For a subject-dependent cross-validation the Fisherfaces method achieves a 99% recognition rate on my preprocessed dataset. A subject-dependent cross-validation simply means, images of the subject under test were also included in the training set (different images of the same person). 

Have a look at the way a cross-validation splits a Dataset *D* with 3 classes (*c0*,*c1*,*c2*) each with 3 observations (*o0*,*o1*,*o2*) into a (non-overlapping) Test Subset *A* and Training Subset *B*: 

.. code-block:: none

      o0 o1 o2        o0 o1 o2        o0 o1 o2  
  c0 | A  B  B |  c0 | B  A  B |  c0 | B  B  A |
  c1 | A  B  B |  c1 | B  A  B |  c1 | B  B  A |
  c2 | A  B  B |  c2 | B  A  B |  c2 | B  B  A |

Allthough the folds are not overlapping (training data is *never* used for testing) the training set contains images of persons we want to know the gender from. So the prediction may depend on the subject and the method finds the closest match to a persons image, but not the gender. What we aim for is a split by class:

.. code-block:: none

      o0 o1 o2        o0 o1 o2        o0 o1 o2  
  c0 | A  A  A |  c0 | B  B  B |  c0 | B  B  B |
  c1 | B  B  B |  c1 | A  A  A |  c1 | B  B  B |
  c2 | B  B  B |  c2 | B  B  B |  c2 | A  A  A |

With this strategy the cross-validation becomes subject-independent, because *images of a subject are never used for learning the model*. The Fisherfaces Method achieves a 98% recognition rate for a subject-independent cross-validation, so it works great... as long as your data is correctly aligned.

gender.txt
----------

In the sample code I will read filenames to images from a CSV file *gender.txt*, which looks like this for my sample images:

.. code-block:: none

  /home/philipp/facerec/data/gender/male/crop_keanu_reeves/keanu_reeves_01.jpg;0
  /home/philipp/facerec/data/gender/male/crop_keanu_reeves/keanu_reeves_02.jpg;0
  /home/philipp/facerec/data/gender/male/crop_keanu_reeves/keanu_reeves_03.jpg;0
  ...
  /home/philipp/facerec/data/gender/female/crop_katy_perry/katy_perry_01.jpg;1
  /home/philipp/facerec/data/gender/female/crop_katy_perry/katy_perry_02.jpg;1
  /home/philipp/facerec/data/gender/female/crop_katy_perry/katy_perry_03.jpg;1
  ...
  /home/philipp/facerec/data/gender/male/crop_brad_pitt/brad_pitt_01.jpg;0
  /home/philipp/facerec/data/gender/male/crop_brad_pitt/brad_pitt_02.jpg;0
  /home/philipp/facerec/data/gender/male/crop_brad_pitt/brad_pitt_03.jpg;0
  ...
  /home/philipp/facerec/data/gender/female/crop_emma_watson/emma_watson_08.jpg;1
  /home/philipp/facerec/data/gender/female/crop_emma_watson/emma_watson_02.jpg;1
  /home/philipp/facerec/data/gender/female/crop_emma_watson/emma_watson_03.jpg;1


You see were this leads to: label ``0`` is for class *male* and label ``1`` is for *female* subjects.

Source Code
-----------

.. code-block:: cpp

  #include "opencv2/opencv.hpp"
  #include "opencv2/highgui/highgui.hpp"

  #include <iostream>
  #include <fstream>
  #include <sstream>

  #include "facerec.hpp"

  using namespace cv;
  using namespace std;

  void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
      std::ifstream file(filename.c_str(), ifstream::in);
      if (!file)
          throw std::exception();
      string line, path, classlabel;
      while (getline(file, line)) {
          stringstream liness(line);
          getline(liness, path, separator);
          getline(liness, classlabel);
          images.push_back(imread(path,0));
          labels.push_back(atoi(classlabel.c_str()));
      }
  }

  int main(int argc, const char *argv[]) {
      // check for command line arguments
      if (argc != 2) {
          cout << "usage: " << argv[0] << " <csv.ext>" << endl;
          exit(1);
      }
      // path to your CSV
      string fn_csv = string(argv[1]);
      // images and corresponding labels
      vector<Mat> images;
      vector<int> labels;
      // read in the data
      try {
          read_csv(fn_csv, images, labels);
      } catch (exception& e) {
          cerr << "Error opening file \"" << fn_csv << "\"." << endl;
          exit(1);
      }
      // get width and height
      int width = images[0].cols;
      int height = images[0].rows;
      // get test instances
      Mat testSample = images[images.size() - 1];
      int testLabel = labels[labels.size() - 1];
      // ... and delete last element
      images.pop_back();
      labels.pop_back();
      // build the Fisherfaces model
      Fisherfaces model(images, labels);
      // test model
      int predicted = model.predict(testSample);
      cout << "predicted class = " << predicted << endl;
      cout << "actual class = " << testLabel << endl;
      // get the eigenvectors
      Mat W = model.eigenvectors();
      // show first 10 fisherfaces
      for (int i = 0; i < min(10, W.cols); i++) {
          // get eigenvector #i
          Mat ev = W.col(i).clone();
          // reshape to original site
          Mat grayscale = toGrayscale(ev.reshape(1, height));
          // show image (with Jet colormap)
          imshow(num2str(i), grayscale, colormap::Jet());
      }
      waitKey(0);
      return 0;
  }
  
Results
-------

If you run the program with your *gender.txt*, you'll see the Fisherface that best separates male and female images:

.. image:: /img/tutorial/gender_classification/fisherface_0.png

And the prediction should yield the correct gender:

.. code-block:: none

  predicted class = 1
  actual class = 1
