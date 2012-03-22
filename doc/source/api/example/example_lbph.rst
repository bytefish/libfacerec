Local Binary Patterns (LBPH) Histograms Example
===============================================

Introduction
------------

Imagine we want to learn the Local Binary Patterns Histograms of the `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_, 
and the filenames and class labels of the subjects are given in a simple CSV file *faces.txt*, which looks like this:

.. code-block:: none

  /home/philipp/facerec/data/at/s1/1.pgm;0
  /home/philipp/facerec/data/at/s1/2.pgm;0
  ...
  /home/philipp/facerec/data/at/s2/1.pgm;1
  /home/philipp/facerec/data/at/s2/2.pgm;1
  ...
  /home/philipp/facerec/data/at/s40/1.pgm;39
  /home/philipp/facerec/data/at/s40/2.pgm;39
  ...

Source Code
-----------

The following program reads in the images and associated class labels and then 
uses the :ocv:class:`LBPH` class of libfacerec to learn the Local Binary Patterns 
Histograms for the given images. Then a prediction is generated for an unseen 
query image.

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
      LBPH model(images, labels);
      // get a prediction
      int predicted = model.predict(testSample);
      // show results
      cout << "predicted class = " << predicted << endl;
      cout << "actual class = " << testLabel << endl;
      waitKey(0);
      return 0;
  }

Results
-------

This model generates Local Binary Patterns Histograms as reference data. Showing 
the histograms might be possible, but I didn't expose this function in the API.

However, you should see a fine prediction from the model:

.. code-block:: cpp

  predicted class = 37
  actual class = 37
