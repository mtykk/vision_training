cmake_minimum_required(VERSION 3.16)

project(vision_training 
        VERSION 0.0.1
        LANGUAGES CXX)

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(./include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(task1_image src/task1_image/main.cpp)
add_executable(task2_fit src/task2_fit/main.cpp)
target_link_libraries(task1_image PRIVATE ${OpenCV_LIBS})
target_link_libraries(task2_fit PRIVATE ${OpenCV_LIBS} Ceres::ceres)