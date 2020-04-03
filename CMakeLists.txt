
#===============================================================================
#      The main build file for building ale using CMake.
#===============================================================================
# CMake initialization

# Specify the required version of CMake.
# cmake 3.10 required for ctest/gtest integration
cmake_minimum_required(VERSION 3.10)
project(ale VERSION 0.7.3 DESCRIPTION "Abstraction Library for Ephemerides ")

# include what we need
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)

set(ALE_BUILD_LOAD ON CACHE BOOL "If the C++ Python load interface should be built.")

# Third Party Dependencies
find_package(Eigen3 3.3    REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)

if(ALE_BUILD_LOAD)
  # If there is an Anaconda environment activated, search that for Python first
  if(EXISTS $ENV{CONDA_PREFIX})
    message("Searching $ENV{CONDA_PREFIX} for Python libraries")
    set(Python_ROOT_DIR $ENV{CONDA_PREFIX})
    set(Python_FIND_STRATEGY LOCATION)
  endif()
  find_package(Python REQUIRED COMPONENTS Development)
endif()

# Library setup
set(ALE_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/")
set(ALE_INSTALL_INCLUDE_DIR "include/ale")
set(ALE_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/InterpUtils.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/Rotation.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/Orientations.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/States.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/Isd.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/Util.cpp)
set(ALE_HEADER_FILES ${ALE_BUILD_INCLUDE_DIR}/InterpUtils.h
                     ${ALE_BUILD_INCLUDE_DIR}/Rotation.h
                     ${ALE_BUILD_INCLUDE_DIR}/Orientations.h
                     ${ALE_BUILD_INCLUDE_DIR}/States.h
                     ${ALE_BUILD_INCLUDE_DIR}/Isd.h
                     ${ALE_BUILD_INCLUDE_DIR}/Util.h)
set(ALE_PRIVATE_LINKS Eigen3::Eigen)
set(ALE_PUBLIC_LINKS nlohmann_json::nlohmann_json)

if(ALE_BUILD_LOAD)
  list(APPEND ALE_SRC_FILES    ${CMAKE_CURRENT_SOURCE_DIR}/src/Ale.cpp)
  list(APPEND ALE_HEADER_FILES ${ALE_BUILD_INCLUDE_DIR}/Ale.h)
  list(APPEND ALE_PRIVATE_LINKS Python::Python)
endif()

add_library(ale SHARED ${ALE_SRC_FILES})

set_target_properties(ale PROPERTIES
                      VERSION   ${PROJECT_VERSION}
                      SOVERSION 0)
# Use generator expressions so that downstream projects can use this target
target_include_directories(ale
                           PUBLIC
                           $<BUILD_INTERFACE:${ALE_BUILD_INCLUDE_DIR}>
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(ale PRIVATE ${ALE_PRIVATE_LINKS}
                          PUBLIC ${ALE_PUBLIC_LINKS})

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
install(FILES ${ALE_HEADER_FILES} DESTINATION ${ALE_INSTALL_INCLUDE_DIR})

# Install the library
install(TARGETS ale
        EXPORT aleTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${ALE_INSTALL_INCLUDE_DIR})

# Install the target
install(EXPORT aleTargets
        NAMESPACE ale::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
