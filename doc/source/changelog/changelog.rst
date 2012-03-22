Changelog
=========

Release 0.02
------------

Reworked the library to provide separate implementations in cpp files, because 
it's the preferred way of contributing OpenCV libraries. This means the library 
is not header-only anymore. Slight API changes were done, please see the 
documentation for details.

Release highlights
******************

- New Unit Tests (for LBP Histograms) make the library more robust.
- Added a documentation and changelog in reStructuredText including:

  - :doc:`API </api/api>`
  - :doc:`API Examples </api/examples>`
  - :doc:`Literature </bib/literature>`

Release 0.01
------------

Initial release as header-only library.

Release highlights
******************

- Colormaps for OpenCV to enhance the visualization.
- Face Recognition algorithms implemented:

  - Eigenfaces [TP91]_
  - Fisherfaces [Belhumeur97]_
  - Local Binary Patterns Histograms [Ahonen04]_
  
- Added persistence facilities to store the models with a common API.
- Unit Tests (using `gtest <http://code.google.com/p/googletest/>`_).
- Providing a CMakeLists.txt to enable easy cross-platform building.
