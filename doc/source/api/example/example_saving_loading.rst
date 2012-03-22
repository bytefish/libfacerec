Saving and Loading a FaceRecognizer
===================================

Introduction
------------

Saving and loading a :ocv:class:`FaceRecognizer` is very important. Training a 
FaceRecognizer can be a very time-intense task, plus it's often impossible
to ship the whole face database to the user of your product. 

The task of saving and loading a FaceRecognizer is very, very easy with 
libfacerec. You only have to call :ocv:func:`FaceRecognizer::load` for loading 
and :ocv:func:`FaceRecognizer::save` for saving a :ocv:class:`FaceRecognizer`.

Imagine we want to learn the Eigenfaces of the `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_ 
store the model to a YAML file and then load it again. From the loaded model 
we'll show the first 10 Eigenfaces.

Filenames and class labels of the subjects are given in a simple CSV file 
*faces.txt*, which looks like this:

.. code-block:: none

  /home/philipp/facerec/data/at/s1/1.pgm;0
  /home/philipp/facerec/data/at/s1/2.pgm;0
  ...
  /home/philipp/facerec/data/at/s2/1.pgm;1
  /home/philipp/facerec/data/at/s2/2.pgm;1
  ...
  /home/philipp/facerec/data/at/s40/1.pgm;39
  /home/philipp/facerec/data/at/s40/2.pgm;39
  
  
Source Code
-----------

The following program:

* Reads in the data.
* Learns the Eigenfaces (as ``model0``).
* Stores the model (``model0``) to ``eigenfaces_at.yml``.
* Initializes & Loads a new model (``model1``) ``eigenfaces_at.yml``.
* Shows the first 10 Eigenfaces of ``model1``.


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
      Eigenfaces model0(images, labels);
      // then save the model
      model0.save("eigenfaces_at.yml");
      // now load it from another object
      Eigenfaces model1;
      model1.load("eigenfaces_at.yml");
      // get the eigenvectors
      Mat W = model1.eigenvectors();
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

``eigenfaces_at.yml`` contains the model state, we'll simply show the first 10 
lines with ``head eigenfaces_at.yml``: 

.. code-block:: none

  philipp@mango:~/github/libfacerec-build$ head eigenfaces_at.yml
  %YAML:1.0
  num_components: 399
  mean: !!opencv-matrix
     rows: 1
     cols: 10304
     dt: d
     data: [ 8.5558897243107765e+01, 8.5511278195488714e+01,
         8.5854636591478695e+01, 8.5796992481203006e+01,
         8.5952380952380949e+01, 8.6162907268170414e+01,
         8.6082706766917283e+01, 8.5776942355889716e+01,

And here are the Eigenfaces:

.. image:: /img/tutorial/stored_loaded_eigenfaces_at.png
