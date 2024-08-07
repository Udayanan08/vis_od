cmake_minimum_required(VERSION 3.12)

project(visual_odometry	 VERSION 0.0.1
			  DESCRIPTION "HELLO"
			  LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -g")

include_directories("/usr/include/eigen3")

message(${CMAKE_CXX_COMPILER_ID})
message(${CMAKE_CXX_COMPILER_VERSION})

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_libraries(${PYTHON_LIBRARIES})


add_library(feature src/feature.cpp)
target_link_libraries(feature ${OpenCV_LIBS} yaml-cpp )

add_executable(vis_od src/vis_od.cpp )
target_link_libraries(vis_od feature)
