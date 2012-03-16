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
target_link_libraries(${the_library} ${OpenCV_LIBS})

#---------------------------------
# add tests
#---------------------------------

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
	file(GLOB test_srcs "test/*.cpp")
	file(GLOB test_hdrs "test/*.h*")
	set(the_test_target "opencv_test_${name}")
	add_executable(${the_test_target} ${test_srcs} ${test_hdrs})
	target_link_libraries(${the_test_target} ${OpenCV_LIBS} ${the_library})
	enable_testing()
	get_target_property(LOC ${the_test_target} LOCATION)
	add_test(${the_test_target} "${LOC}")
endif()

#---------------------------------
# compile a sample executable
#---------------------------------
add_executable(fisherfaces src/main.cpp)
target_link_libraries(fisherfaces ${OpenCV_LIBS} ${the_library})