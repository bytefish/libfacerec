################################
# Face Recognition with OpenCV #
################################

Introduction
============

`OpenCV (Open Source Computer Vision) <http://opencv.willowgarage.com>`_ is a popular computer vision library started by `Intel <http://www.intel.com>`_ in 1999. The cross-platform library sets its focus on real-time image processing and includes patent-free implementations of the latest computer vision algorithms. In 2008 `Willow Garage <http://www.willowgarage.com>`_ took over support and OpenCV 2.3.1 now comes with a programming interface to C, C++, `Python <http://www.python.org>`_ and `Android <http://www.android.com>`_. OpenCV is released under a BSD license, so it is used in academic and commercial projects alike, such as `Google Streetview <http://www.google.com/streetview>`.

OpenCV 2.4 now comes with :ocv:class:`FaceRecognizer` for face recognition, so you can start experimenting right away! This document is the guide I've wished for, when I was working myself into face recognition. It gives you an introduction into the algorithms behind and shows you how to use them. You don't need to copy and paste from this page, because full source code listings available in the ``samples/cpp`` folder coming with OpenCV. Although it might be interesting for advanced users, I've decided to leave the implementation details out, as I am afraid they confuse new users.

All code in this document is released under the `BSD license <http://www.opensource.org/licenses/bsd-license>`, so feel free to use it for your projects.

Face Recognition
================

Face recognition is an easy task for humans. Experiments in [Tu06]_ have shown, that even one to three day old babies are able to distinguish between known faces. So how hard could it be for a computer? It turns out we know little about human recognition to date. Are inner features (eyes, nose, mouth) or outer features (head shape, hairline) used for a successful face recognition? How do we analyze an image and how does the brain encode it? It was shown by `David Hubel <http://en.wikipedia.org/wiki/David_H._Hubel>`_ and `Torsten Wiesel <http://en.wikipedia.org/wiki/Torsten_Wiesel>`_, that our brain has specialized nerve cells responding to specific local features of a scene, such as lines, edges, angles or movement. Since we don't see the world as scattered pieces, our visual cortex must somehow combine the different sources of information into useful patterns. Automatic face recognition is all about extracting those meaningful features from an image, putting them into a useful representation and performing some kind of classification on them.

Face recognition based on the geometric features of a face is probably the most intuitive approach to face recognition. One of the first automated face recognition systems was described in [Kanade73]_: marker points (position of eyes, ears, nose, ...) were used to build a feature vector (distance between the points, angle between them, ...). The recognition was performed by calculating the euclidean distance between feature vectors of a probe and reference image. Such a method is robust against changes in illumination by its nature, but has a huge drawback: the accurate registration of the marker points is complicated, even with state of the art algorithms. Some of the latest work on geometric face recognition was carried out in [Bru92]_. A 22-dimensional feature vector was used and experiments on large datasets have shown, that geometrical features alone my not carry enough information for face recognition.

The Eigenfaces method described in [PT91]_ took a holistic approach to face recognition: A facial image is a point from a high-dimensional image space and a lower-dimensional representation is found, where classification becomes easy. The lower-dimensional subspace is found with Principal Component Analysis, which identifies the axes with maximum variance. While this kind of transformation is optimal from a reconstruction standpoint, it doesn't take any class labels into account. Imagine a situation where the variance is generated from external sources, let it be light. The axes with maximum variance do not necessarily contain any discriminative information at all, hence a classification becomes impossible. So a class-specific projection with a Linear Discriminant Analysis was applied to face recognition in [BHK97]_. The basic idea is to minimize the variance within a class, while maximizing the variance between the classes at the same time (see Figure :ref:scatter_matrices). 

Recently various methods for a local feature extraction emerged. To avoid the high-dimensionality of the input data only local regions of an image are described, the extracted features are (hopefully) more robust against partial occlusion, illumation and small sample size. Algorithms used for a local feature extraction are Gabor Wavelets ([Wiskott97]_), Discrete Cosinus Transform ([Cardinaux2006]_) and Local Binary Patterns ([Ahonen04]_, [Maturana09]_, [Rodriguez2006]_). It's still an open research question how to preserve spatial information when applying a local feature extraction, because spatial information is potentially useful information.

Face Database
==============

Let's get some data to experiment with first. I don't want to do a toy example here. We are doing face recognition, so you'll need some face images! You can either create your own database or start with one of the available databases, `http://face-rec.org/databases/ <http://face-rec.org/databases>`_ gives an up-to-date overview. Three interesting databases are (parts of the description are quoted from `http://face-rec.org <http://face-rec.org>.`_:

* `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_ The AT&T Facedatabase, sometimes also referred to as *ORL Database of Faces*, contains ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement).
	
* `Yale Facedatabase A <http://cvc.yale.edu/projects/yalefaces/yalefaces.html>`_ The AT&T Facedatabase is good for initial tests, but it's a fairly easy database. The Eigenfaces method already has a 97% recognition rate, so you won't see any improvements with other algorithms. The Yale Facedatabase A is a more appropriate dataset for initial experiments, because the recognition problem is harder. The database consists of 15 people (14 male, 1 female) each with 11 grayscale images sized :math:`320 \times 243` pixel. There are changes in the light conditions (center light, left light, right light), facial expressions (happy, normal, sad, sleepy, surprised, wink) and glasses (glasses, no-glasses). 

  Bad news is it's not available for public download anymore, because the original server seems to be down. You can find some sites mirroring it (`like the MIT <http://vismod.media.mit.edu/vismod/classes/mas622-00/datasets/>`_), but I can't make guarantees about the integrity. If you need to crop and align the images yourself, read my notes at `bytefish.de/blog/fisherfaces <http://bytefish.de/blog/fisherfaces>`_.
	
* `Extended Yale Facedatabase B <http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html>`_ The Extended Yale Facedatabase B contains 2414 images of 38 different people in its cropped version. The focus of this database is set on extracting features that are robust to illumination, the images have almost no variation in emotion/occlusion/... . I personally think, that this dataset is too large for the experiments I perform in this document. You better use the `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_ for intial testing. A first version of the Yale Facedatabase B was used in [Belhumeur97]_ to see how the Eigenfaces and Fisherfaces method perform under heavy illumination changes. [Lee2005]_ used the same setup to take 16128 images of 28 people. The Extended Yale Facedatabase B is the merge of the two databases, which is now known as Extended Yalefacedatabase B.

Preparing the data
+++++++++++++++++++

Once we have acquired some data, we'll need to read it in our program. In the demo I have decided to read the images from a very simple CSV file. Why? Because it's the simplest platform-independent approach I can think of. However, if you know a simpler solution please ping me about it. Basically all the CSV file needs to contain are lines composed of a ``filename`` followed by a ``;`` followed by the ``label`` (as *integer number*), making up a line like this: 

.. code-block::

    /path/to/image.ext;0
    
Let's dissect the line. ``/path/to/image.ext`` is the path to an image, probably something like this if you are in Windows: ``C:/faces/person0/image0.jpg``. Then there is the separator ``;`` and finally we assign the label ``0`` to the image. Think of the label as the subject (the person) this image belongs to, so same subjects (persons) should have the same label.

Download the AT&T Facedatabase from AT&T Facedatabase and the corresponding CSV file from at.txt, which looks like this (file is without … of course): 

.. code-block::

    ./at/s1/1.pgm;0
    ./at/s1/2.pgm;0
    ...
    ./at/s2/1.pgm;1
    ./at/s2/2.pgm;1
    ...
    ./at/s40/1.pgm;39
    ./at/s40/2.pgm;39

Imagine I have extracted the files to ``D:/data/at`` and have downloaded the CSV file to ``D:/data/at.txt``. Then you would simply need to Search & Replace ``./`` with ``D:/data/``. You can do that in an editor of your choice, every sufficiently advanced editor can do this. Once you have a CSV file with valid filenames and labels, you can run any of the demos by passing the path to the CSV file as parameter: 

.. code-block::

    facerec_demo.exe D:/data/at.txt

Eigenfaces
==========

The problem with the image representation we are given is its high dimensionality. Two-dimensional :math:`p \times q` grayscale images span a :math:`m = pq`-dimensional vector space, so an image with :math:`100 \times 100` pixels lies in a :math:`10,000`-dimensional image space already. The question is: Are all dimensions equally useful for us? We can only make a decision if there's any variance in data, so what we are looking for are the components that account for most of the information. The Principal Component Analysis (PCA) was independently proposed by `Karl Pearson <http://en.wikipedia.org/wiki/Karl_Pearson>`_ (1901) and `Harold Hotelling <http://en.wikipedia.org/wiki/Harold_Hotelling>`_ (1933) to turn a set of possibly correlated variables into a smaller set of uncorrelated variables. The idea is, that a high-dimensional dataset is often described by correlated variables and therefore only a few meaningful dimensions account for most of the information. The PCA method finds the directions with the greatest variance in the data, called principal components.

Algorithmic Description
-----------------------

Let :math:`X = \{ x_{1}, x_{2}, \ldots, x_{n} \}` be a random vector with observations :math:`x_i \in R^{d}`.

1. Compute the mean :math:`\mu = \frac{1}{n} \sum_{i=1}^{n} x_{i}`

  .. math::
  
    \mu = \frac{1}{n} \sum_{i=1}^{n} x_{i}
    
2. Compute the the Covariance Matrix `S`

  .. math::
  
    S = \frac{1}{n} \sum_{i=1}^{n} (x_{i} - \mu) (x_{i} - \mu)^{T}`
    
3. Compute the eigenvalues :math:`\lambda_{i}` and eigenvectors :math:`v_{i}` of :math:`S`

  .. math:: 
    
    S v_{i} = \lambda_{i} v_{i}, i=1,2,\ldots,n
    
4. Order the eigenvectors descending by their eigenvalue. The :math:`k` principal components are the eigenvectors corresponding to the :math:`k` largest eigenvalues.

The :math:`k` principal components of the observed vector :math:`x` are then given by:

.. math::

	y = W^{T} (x - \mu)


where :math:`W = (v_{1}, v_{2}, \ldots, v_{k})`. The reconstruction from the PCA basis is given by:

.. math::

	x = W y + \mu
	
where :math:`W = (v_{1}, v_{2}, \ldots, v_{k})`.


The Eigenfaces method then performs face recognition by:

* Projecting all training samples into the PCA subspace.
* Projecting the query image into the PCA subspace.
* Finding the nearest neighbor between the projected training images and the projected query image.

Still there's one problem left to solve. Imagine we are given :math:`400` images sized :math:`100 \times 100` pixel. The Principal Component Analysis solves the covariance matrix :math:`S = X X^{T}`, where :math:`{size}(X) = 10000 \times 400` in our example. You would end up with a :math:`10000 \times 10000` matrix, roughly :math:`0.8 GB`. Solving this problem isn't feasible, so we'll need to apply a trick. From your linear algebra lessons you know that a :math:`M \times N` matrix with :math:`M > N` can only have :math:`N - 1` non-zero eigenvalues. So it's possible to take the eigenvalue decomposition :math:`S = X^{T} X` of size :math:`N \times N` instead:

.. math::

	X^{T} X v_{i} = \lambda_{i} v{i}


and get the original eigenvectors of :math:`S = X X^{T}` with a left multiplication of the data matrix:

.. math::

	X X^{T} (X v_{i}) = \lambda_{i} (X v_{i})

The resulting eigenvectors are orthogonal, to get orthonormal eigenvectors they need to be normalized to unit length. I don't want to turn this into a publication, so please look into [Duda2001]_ for the derivation and proof of the equations.

Eigenfaces in OpenCV
--------------------

Now it's time for the fun stuff. 

Training and Prediction
+++++++++++++++++++++++

	The Eigenfaces and Fisherfaces method both share common methods, so we'll define a base prediction model in Listing \ref{lst:eigenfaces_base_model}. I don't want to do a full k-Nearest Neighbor implementation here, because (1) the number of neighbors doesn't really matter for both methods and (2) it would confuse people. If you are implementing it in a language of your choice, you should really separate the feature extraction and classification from the model itself. A real generic approach is given in my \href{https://www.github.com/bytefish/facerec}{facerec} framework. However, feel free to extend these basic classes for your needs.

Projection and Reconstruction
++++++++++++++++++++++++++++++

.. code-block::

    [HIER EIGENFACES]


I've used the jet colormap, so you can see how the grayscale values are distributed within the specific Eigenfaces. You can see, that the Eigenfaces do not only encode facial features, but also the illumination in the images (see the left light in Eigenface \#4, right light in Eigenfaces \#5):

    [img eigenfaces]

We've already seen in Equation \ref{eqn:pca_reconstruction}, that we can reconstruct a face from its lower dimensional approximation. So let's see how many Eigenfaces are needed for a good reconstruction. I'll do a subplot with $10,30,\ldots,310$ Eigenfaces:

.. code-block:

    [[HIER EIGENFACES RECONSTRUCTION]]

10 Eigenvectors are obviously not sufficient for a good image reconstruction, 50 Eigenvectors may already be sufficient to encode important facial features. You'll get a good reconstruction with approximately 300 Eigenvectors for the AT\&T Facedatabase. There are rule of thumbs how many Eigenfaces you should choose for a successful face recognition, but it heavily depends on the input data. \cite{zhao2003frl} is the perfect point to start researching for this.



Fisherfaces
============

The Principal Component Analysis (PCA), which is the core of the Eigenfaces method, finds a linear combination of features that maximizes the total variance in data. While this is clearly a powerful way to represent data, it doesn't consider any classes and so a lot of discriminative information *may* be lost when throwing components away. Imagine a situation where the variance in your data is generated by an external source, let it be the light. The components identified by a PCA do not necessarily contain any discriminative information at all, so the projected samples are smeared together and a classification becomes impossible (see `http://www.bytefish.de/wiki/pca_lda_with_gnu_octave <http://www.bytefish.de/wiki/pca_lda_with_gnu_octave>`_ for an example).

The Linear Discriminant Analysis performs a class-specific dimensionality reduction and was invented by the great statistician `Sir R. A. Fisher <http://en.wikipedia.org/wiki/Ronald_Fisher>`_. He successfully used it for classifying flowers in his 1936 paper *The use of multiple measurements in taxonomic problems* [Fisher36]_. In order to find the combination of features that separates best between classes the Linear Discriminant Analysis maximizes the ratio of between-classes to within-classes scatter, instead of maximizing the overall scatter. The idea is simple: same classes should cluster tightly together, while different classes are as far away as possible from each other in the lower-dimensional representation. This was also recognized by `Belhumeur <http://www.cs.columbia.edu/~belhumeur/>`_, `Hespanha <http://www.ece.ucsb.edu/~hespanha/>`_ and `Kriegman <http://cseweb.ucsd.edu/~kriegman/>`_ and so they applied a Discriminant Analysis to face recognition in [BHK97]_. 


\begin{figure}
	\begin{center}
		\includegraphics[scale=1.70]{img/fisherfaces/multiclasslda}
		\captionof{figure}{This figure shows the scatter matrices $S_{B}$ and $S_{W}$ for a 3 class problem. $\mu$ represents the total mean and $[\mu_{1},\mu_{2},\mu_{3}]$ are the class means.}
		\label{fig:scatter_matrices}
	\end{center}
\end{figure}

Algorithmic Description
-----------------------

Let :math:`X` be a random vector with samples drawn from :math:`c` classes:


.. math::

    X & = & \{X_1,X_2,\ldots,X_c\} \\
    X_i & = & \{x_1, x_2, \ldots, x_n\}

The scatter matrices :math:`S_{B}` and `S_{W}` are calculated as:

.. math::

    S_{B} & = & \sum_{i=1}^{c} N_{i} (\mu_i - \mu)(\mu_i - \mu)^{T} \\
    S_{W} & = & \sum_{i=1}^{c} \sum_{x_{j} \in X_{i}} (x_j - \mu_i)(x_j - \mu_i)^{T}

, where :math:`\mu` is the total mean:

..math::

    \mu = \frac{1}{N} \sum_{i=1}^{N} x_i


And :math:`\mu_i` is the mean of class :math:`i \in \{1,\ldots,c\}`:

.. math::
 
    \mu_i = \frac{1}{|X_i|} \sum_{x_j \in X_i} x_j

Fisher's classic algorithm now looks for a projection :math:`W`, that maximizes the class separability criterion:

.. math::

    W_{opt} = \operatorname{arg\,max}_{W} \frac{|W^T S_B W|}{|W^T S_W W|}


Following [BHK97]_, a solution for this optimization problem is given by solving the General Eigenvalue Problem:

.. math::

    S_{B} v_{i} & = & \lambda_{i} S_w v_{i} \nonumber \\
    S_{W}^{-1} S_{B} v_{i} & = & \lambda_{i} v_{i}

There's one problem left to solve: The rank of :math:`S_{W}` is at most :math:`(N-c)`, with :math:`N` samples and :math:`c` classes. In pattern recognition problems the number of samples :math:`N` is almost always samller than the dimension of the input data (the number of pixels), so the scatter matrix :math:`S_{W}` becomes singular (see [Raudys1991]_). In [BHK97]_ this was solved by performing a Principal Component Analysis on the data and projecting the samples into the :math:`(N-c)`-dimensional space. A Linear Discriminant Analysis was then performed on the reduced data, because :math:`S_{W}` isn't singular anymore.

The optimization problem can then be rewritten as:

.. math::

    W_{pca} & = & \operatorname{arg\,max}_{W} |W^T S_T W| \\
    W_{fld} & = & \operatorname{arg\,max}_{W} \frac{|W^T W_{pca}^T S_{B} W_{pca} W|}{|W^T W_{pca}^T S_{W} W_{pca} W|}

The transformation matrix :math:`W`, that projects a sample into the :math:`(c-1)`-dimensional space is then given by:

.. math::
    
    W = W_{fld}^{T} W_{pca}^{T}

Fisherfaces in OpenCV
---------------------

Training and Prediction
++++++++++++++++++++++++

Projection and Reconstruction
+++++++++++++++++++++++++++++

For this example I am going to use the Yale Facedatabase A, just because the plots are nicer. Each Fisherface has the same length as an original image, thus it can be displayed as an image. We'll again load the data, learn the Fisherfaces and make a subplot of the first 16 Fisherfaces.

\ifx\python\undefined
	\lstinputlisting[caption={\href{src/m/example\_fisherfaces.m}{src/m/example\_fisherfaces.m} \label{lst:example_fisherfaces_fisherfaces}}, language=matlab, linerange={1-23}]{src/m/example_fisherfaces.m}
\else
	\lstinputlisting[caption={\href{src/py/scripts/example\_fisherfaces.py}{src/py/scripts/example\_fisherfaces.py} \label{lst:example_fisherfaces_fisherfaces}}, language=python, linerange={1-23}]{src/py/scripts/example_fisherfaces.py}
\fi

The Fisherfaces method learns a class-specific transformation matrix, so the they do not capture illumination as obviously as the Eigenfaces method. The Discriminant Analysis instead finds the facial features to discriminate between the persons. It's important to mention, that the performance of the Fisherfaces heavily depends on the input data as well. Practically said: if you learn the Fisherfaces for well-illuminated pictures only and you try to recognize faces in bad-illuminated scenes, then method is likely to find the wrong components (just because those features may not be predominant on bad illuminated images). This is somewhat logical, since the method had no chance to learn the illumination.

\ifx\python\undefined
	\begin{center}
		\includegraphics[scale=0.6]{img/fisherfaces/octave_fisherfaces_fisherfaces}
	\end{center}
\else
	\begin{center}
		\includegraphics[scale=0.6]{img/fisherfaces/python_fisherfaces_fisherfaces}
	\end{center}
\fi

The Fisherfaces allow a reconstruction of the projected image, just like the Eigenfaces did. But since we only identified the features to distinguish between subjects, you can't expect a nice reconstruction of the original image. For the Fisherfaces method we'll project the sample image onto each of the Fisherfaces instead. So you'll have a nice visualization, which feature each of the Fisherfaces describes.

\ifx\python\undefined
	\lstinputlisting[caption={\href{src/m/example\_fisherfaces.m}{src/m/example\_fisherfaces.m} \label{lst:example_fisherfaces_reconstruction}}, language=matlab, linerange={28-40}]{src/m/example_fisherfaces.m}
\else
	\lstinputlisting[caption={\href{src/py/scripts/example\_fisherfaces.py}{src/py/scripts/example\_fisherfaces.py} \label{lst:example_fisherfaces_reconstruction}}, language=python, linerange={25-36}]{src/py/scripts/example_fisherfaces.py}
\fi

The differences may be subtle for the human eyes, but you should be able to see some differences:

.. image:: /img/tutorial/eigenfaces_at.png


  
Local Binary Patterns Histograms
================================


