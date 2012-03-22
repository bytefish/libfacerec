Fisherfaces Example
===================

Introduction
------------

Imagine we want to learn the Fisherfaces of the `Yale Facedatabase A <http://cvc.yale.edu/projects/yalefaces/yalefaces.html>`_, 
and the filenames and class labels of the subjects are given in a simple CSV file *faces.txt*, which looks like this:

.. code-block:: none

  /home/philipp/facerec/data/yalefaces_a/s01/subject01.centerlight;0
  /home/philipp/facerec/data/yalefaces_a/s01/subject01.sad;0
  /home/philipp/facerec/data/yalefaces_a/s01/subject01.sleepy;0
  ...
  /home/philipp/facerec/data/yalefaces_a/s02/subject02.centerlight;1
  /home/philipp/facerec/data/yalefaces_a/s02/subject02.sad;1
  /home/philipp/facerec/data/yalefaces_a/s02/subject02.sleepy;1
  ...

Source Code
-----------

The following program reads in the images and associated class labels and then uses the Fisherfaces class of libfacerec to learn the Fisherfaces.

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
          imshow(num2str(i), grayscale, colormap::Bone());
      }
      waitKey(0);
      return 0;
  }

Results
-------

This yields the first 10 Fisherfaces (a Bone colormap was applied):

.. image:: /img/tutorial/fisherfaces_yale_a.png

So you see... learning the Fisherfaces is just as easy as writing:

.. code-block:: cpp

  Fisherfaces model(images, labels);
  
and generating a prediction from the learned model is simply:

.. code-block:: cpp

  int predicted = model.predict(testSample);
