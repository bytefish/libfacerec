CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(name "facerec")
set(the_library "opencv_${name}")
project(${the_library})

#SET(OpenCV_DIR /path/to/your/opencv/installation)

#---------------------------------
# packages
#---------------------------------
find_package(OpenCV REQUIRED) # http://opencv.willowgarage.com

#---------------------------------
# includes & sources
#---------------------------------

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB lib_hdrs "include/*.hpp")
file(GLOB lib_srcs "src/*.cpp")

#---------------------------------
# build the library & link
#---------------------------------
add_library(${the_library} ${lib_srcs} ${lib_hdrs})
target_link_libraries(${the_library} opencv_core opencv_imgproc opencv_highgui)

#---------------------------------
# add tests
#---------------------------------
if(NOT DEFINED ENABLE_TESTS)
    set(ENABLE_TESTS FALSE CACHE BOOL "Enable to make the tests")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test AND ENABLE_TESTS)
	file(GLOB test_srcs "test/*.cpp")
	file(GLOB test_hdrs "test/*.h*")
	set(the_test_target "opencv_test_${name}")
	add_executable(${the_test_target} ${test_srcs} ${test_hdrs})
	# link to opencv_ts, as it is not in OpenCV_LIBS for OpenCV 2.3.1 
	target_link_libraries(${the_test_target} ${the_library} opencv_core opencv_ts)
	enable_testing()
	get_target_property(LOC ${the_test_target} LOCATION)
	add_test(${the_test_target} "${LOC}")
endif()

#---------------------------------
# add samples
#---------------------------------
if(NOT DEFINED ENABLE_SAMPELS)
    set(ENABLE_SAMPLES FALSE CACHE BOOL "Enable to make the samples")
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/samples AND ENABLE_SAMPLES) 
	# probably there's a simpler way to link against multiple executables...
	add_executable(facerec_demo samples/facerec_demo.cpp)
	target_link_libraries(facerec_demo ${the_library} opencv_core opencv_imgproc opencv_highgui)
	
	add_executable(facerec_video samples/facerec_video.cpp)
	target_link_libraries(facerec_video ${the_library} opencv_core opencv_imgproc opencv_highgui opencv_objdetect opencv_imgproc)
	
	add_executable(facerec_eigenfaces samples/facerec_eigenfaces.cpp)
	target_link_libraries(facerec_eigenfaces ${the_library} opencv_core opencv_imgproc opencv_highgui)
	
	add_executable(facerec_fisherfaces samples/facerec_fisherfaces.cpp)
	target_link_libraries(facerec_fisherfaces ${the_library} opencv_core opencv_imgproc opencv_highgui)

	add_executable(facerec_lbph samples/facerec_lbph.cpp)
	target_link_libraries(facerec_lbph ${the_library} opencv_core opencv_imgproc opencv_highgui)
endif()
