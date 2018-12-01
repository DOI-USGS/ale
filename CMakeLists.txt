
#===============================================================================
#      The main build file for building EAL using CMake.
#===============================================================================
# CMake initialization

# Specify the required version of CMake.
# cmake 3.10 required for ctest/gtest integration
cmake_minimum_required(VERSION 3.10)
project(usgsephem VERSION 0.0.1 DESCRIPTION "usgs ephemerides library")

# include what we need
include(GNUInstallDirs)
include(cmake/gtest.cmake)
include(GoogleTest)
include(CTest)

set(CMAKE_CXX_STANDARD 11)

# Third Party Dependencies
find_package(GSL)

# Library setup
add_library(usgsephem SHARED
            src/eal.cpp
)
set_target_properties(usgsephem PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)
set(EPHEM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/"
                       "${CMAKE_CURRENT_SOURCE_DIR}/include/json")
target_include_directories(usgsephem
                           PRIVATE
                           ${GSL_INCLUDE_DIRS}
                           PUBLIC
                           ${EPHEM_INCLUDE_DIRS}
)

# Setup for GoogleTest
find_package (Threads)
target_link_libraries(usgsephem
                      PRIVATE
                      ${GSL_LIBRARIES}
                      PUBLIC
                      gtest ${CMAKE_THREAD_LIBS_INIT})

# Setup for ctest
enable_testing()
add_subdirectory(tests)

# Setup for code coverage
# default to off
set(COVERAGE OFF CACHE BOOL "Coverage")
if(COVERAGE)
    target_compile_options(usgsephem PRIVATE --coverage -O0)
    target_link_libraries(usgsephem PRIVATE --coverage -O0)
endif()

# Install commands
install(TARGETS usgsephem LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${EPHEM_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
