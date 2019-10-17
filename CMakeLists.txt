
#===============================================================================
#      The main build file for building ale using CMake.
#===============================================================================
# CMake initialization

# Specify the required version of CMake.
# cmake 3.10 required for ctest/gtest integration
cmake_minimum_required(VERSION 3.10)
project(ale VERSION 0.6.0 DESCRIPTION "Abstraction Library for Ephemerides ")

# include what we need
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)

# Third Party Dependencies
find_package(GSL    REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Python REQUIRED COMPONENTS Development)
find_package(nlohmann_json REQUIRED)

# Library setup
add_library(ale SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ale.cpp)
set_target_properties(ale PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)
set(ALE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/")

target_include_directories(ale
                           PUBLIC
                           ${ALE_INCLUDE_DIRS})

target_link_libraries(ale
                      PRIVATE
                      GSL::gsl
                      GSL::gslcblas
                      Eigen3::Eigen
                      Python::Python
                      nlohmann_json::nlohmann_json)

# Optional build tests
option (BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    include(cmake/gtest.cmake)
    include(GoogleTest)
    include(CTest)

    find_package (Threads)
    enable_testing()
    add_subdirectory(tests/ctests)

    # Setup for code coverage
    # default to off
    set(COVERAGE OFF CACHE BOOL "Coverage")
    if(COVERAGE)
        target_compile_options(ale PRIVATE --coverage -O0)
        target_link_libraries(ale PRIVATE --coverage -O0)
    endif()
endif()

# Install commands
install(TARGETS ale LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${ALE_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
