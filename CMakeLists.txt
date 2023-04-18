cmake_minimum_required(VERSION 3.0.2)
project(GECCO-2023-LAHS
        LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)   # See below (1)
add_definitions(-std=c++17)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE RELEASE)

add_executable(experiment "experiment.cpp")
add_subdirectory("coco")
find_package(GSL REQUIRED)
target_link_libraries(experiment PUBLIC coco GSL::gsl GSL::gslcblas)

IF (BUILD_TESTING)
    enable_testing()
    add_executable(mytest test.cpp)
    target_link_libraries(mytest PUBLIC coco GSL::gsl GSL::gslcblas)
    add_test(NAME mytest
             COMMAND "$<TARGET_FILE:mytest>")
ENDIF(BUILD_TESTING)

