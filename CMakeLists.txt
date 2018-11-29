# cmake 3.10 required for ctest/gtest integration
cmake_minimum_required(VERSION 3.10)
project(usgsephem VERSION 0.0.1 DESCRIPTION "usgs ephemerides library")

# include what we need
include(GNUInstallDirs)
#include(cmake/gtest.cmake)
#include(GoogleTest)
#include(CTest)

set(CMAKE_CXX_STANDARD 11)

# Library setup
add_library(usgsephem SHARED
            src/TestObject.cpp
            src/eal.cpp
)
set_target_properties(usgsephem PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)
set(EPHEM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(usgsephem
                           PUBLIC
                           "${EPHEM_INCLUDE_DIRS}"
)

# Setup for code coverage
# default to off
set(COVERAGE OFF CACHE BOOL "Coverage")
if(COVERAGE)
    target_compile_options(usgsephem PRIVATE --coverage -O0)
    target_link_libraries(usgsephem PRIVATE --coverage -O0)
endif()

# Setup for GoogleTest
#find_package (Threads)
#target_link_libraries(usgsephem
#                      gtest ${CMAKE_THREAD_LIBS_INIT})

# Setup for ctest
#enable_testing()
#add_subdirectory(tests)

# Install commands
install(TARGETS usgsephem LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${EPHEM_INCLUDE_DIRS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
