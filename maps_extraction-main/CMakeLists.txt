cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(pc_tree)

find_package(PCL 1.3 REQUIRED COMPONENTS)
find_package(OpenCV REQUIRED)


include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pc_tree src/pc_tree.cpp)
target_link_libraries(pc_tree ${PCL_LIBRARIES} ${OpenCV_LIBS})
set(CMAKE_CXX_FLAGS "-std=c++11")



