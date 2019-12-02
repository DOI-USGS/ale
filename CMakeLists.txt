
#===============================================================================
#      The main build file for building ale using CMake.
#===============================================================================
# CMake initialization

# Specify the required version of CMake.
# cmake 3.10 required for ctest/gtest integration
cmake_minimum_required(VERSION 3.10)
project(ale VERSION 0.6.5 DESCRIPTION "Abstraction Library for Ephemerides ")

# include what we need
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)

# Third Party Dependencies
find_package(GSL           REQUIRED)
find_package(Eigen3 3.3    REQUIRED NO_MODULE)
find_package(Python        REQUIRED COMPONENTS Development)
find_package(nlohmann_json REQUIRED)

# Library setup
add_library(ale SHARED
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ale.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Rotation.cpp)
set(ALE_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set(ALE_HEADERS "${ALE_BUILD_INCLUDE_DIR}/ale.h"
                "${ALE_BUILD_INCLUDE_DIR}/Rotation.h")
set(ALE_INSTALL_INCLUDE_DIR "include/ale")
set_target_properties(ale PROPERTIES
                      VERSION             ${PROJECT_VERSION}
                      SOVERSION           0)
# Use generator expressions so that downstream projects can use this target
target_include_directories(ale
                           PUBLIC
                           $<BUILD_INTERFACE:${ALE_BUILD_INCLUDE_DIR}>
                           $<INSTALL_INTERFACE:include>)

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

# Generate the package config
configure_file(cmake/config.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
               @ONLY)

# Install the package config
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

# Install the headers
install(FILES ${ALE_HEADERS} DESTINATION ${ALE_INSTALL_INCLUDE_DIR})

# Install the library
install(TARGETS ale
        EXPORT aleTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${ALE_INSTALL_INCLUDE_DIR})

# Install the target
install(EXPORT aleTargets
        NAMESPACE ale::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
