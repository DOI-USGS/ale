
#===============================================================================
#      The main build file for building ale using CMake.
#===============================================================================
# CMake initialization

# Specify the required version of CMake.
# cmake 3.10 required for ctest/gtest integration
cmake_minimum_required(VERSION 3.10)
project(ale VERSION 0.0.1 DESCRIPTION "Abstraction Library for Ephemerides ")

# include what we need
include(GNUInstallDirs)
include(cmake/gtest.cmake)
include(GoogleTest)
include(CTest)

set(CMAKE_CXX_STANDARD 11)

# Third Party Dependencies
find_package(GSL    REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PythonLibs REQUIRED COMPONENTS Development Interpreter Compiler)

# Library setup
add_library(ale SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ale.cpp)
set_target_properties(ale PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)
set(ALE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/"
                     "${CMAKE_CURRENT_SOURCE_DIR}/include/json")

target_include_directories(ale
                           PRIVATE
                           ${GSL_INCLUDE_DIRS}
                           ${EIGEN3_INCLUDE_DIR}
                           ${PYTHON_INCLUDE_DIR}
                           PUBLIC
                          ${ALE_INCLUDE_DIRS})

# Setup for GoogleTest
find_package(Threads)
target_link_libraries(ale
                      PRIVATE
                      ${GSL_LIBRARIES}
                      ${PYTHON_LIBRARY}
                      PUBLIC
                      gtest ${CMAKE_THREAD_LIBS_INIT})

# Setup for ctest
enable_testing()
add_subdirectory(tests/ctests)

# Setup for code coverage
# default to off
set(COVERAGE OFF CACHE BOOL "Coverage")
if(COVERAGE)
    target_compile_options(ale PRIVATE --coverage -O0)
    target_link_libraries(ale PRIVATE --coverage -O0)
endif()

# Install commands
install(TARGETS ale LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${ALE_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
