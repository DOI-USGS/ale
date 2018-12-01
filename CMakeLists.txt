
#===============================================================================
#      The main build file for building EAL using CMake.
#===============================================================================
# CMake initialization

# Specify the required version of CMake.
# cmake 3.10 required for ctest/gtest integration
cmake_minimum_required(VERSION 3.10)
project(eal VERSION 0.0.1 DESCRIPTION "Ephemeris Abstraction Library")

# include what we need
include(GNUInstallDirs)
include(cmake/gtest.cmake)
include(GoogleTest)
include(CTest)

set(CMAKE_CXX_STANDARD 11)

# Library setup
add_library(eal SHARED
            src/eal.cpp
)
set_target_properties(eal PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)
set(EAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/"
                       "${CMAKE_CURRENT_SOURCE_DIR}/include/json")
target_include_directories(eal
                           PUBLIC
                           "${EAL_INCLUDE_DIRS}"
)

# Setup for GoogleTest
find_package (Threads)
target_link_libraries(eal PUBLIC
                      gtest ${CMAKE_THREAD_LIBS_INIT})

# Setup for ctest
enable_testing()
add_subdirectory(tests)

# Setup for code coverage
# default to off
set(COVERAGE OFF CACHE BOOL "Coverage")
if(COVERAGE)
    target_compile_options(eal PRIVATE --coverage -O0)
    target_link_libraries(eal PRIVATE --coverage -O0)
endif()

# Install commands
install(TARGETS eal LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${EAL_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
