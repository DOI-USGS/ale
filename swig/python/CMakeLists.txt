cmake_minimum_required(VERSION 3.14)

# Setup for SWIG
set(CMAKE_SWIG_FLAGS)
find_package(SWIG REQUIRED)
include(UseSWIG)
list(APPEND CMAKE_SWIG_FLAGS "-DPY3;-keyword")

# Setup for Python linking
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Setup for wrapper library
set(ALE_C_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/ale_c")
set(ALE_C_SOURCES ../ale_c.i)

set_source_files_properties(${ALE_C_SOURCES} PROPERTIES CPLUSPLUS ON)
swig_add_library(ale_c
                 TYPE MODULE
                 LANGUAGE python
                 SOURCES ${ALE_C_SOURCES}
                 OUTPUT_DIR ${ALE_C_OUTPUT_DIR})
target_link_libraries(ale_c ale Python3::Module)
set_target_properties(ale_c PROPERTIES SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
set_target_properties(ale_c PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${ALE_C_OUTPUT_DIR})

# Setup for wrapper tests
# if(SPICEQL_BUILD_TESTS)
#     add_test(NAME python-tests
#         COMMAND python -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/tests
#         WORKING_DIRECTORY ${PYSPICEQL_OUTPUT_DIR})
# else()
#     message(STATUS "Skipping Python Bindings Tests")
# endif()

# Create the files to install the Python wrapper
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in
               ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
               ${ALE_C_OUTPUT_DIR}/__init__.py
               COPYONLY)

# Setup to run setup tools on install 
install(CODE "execute_process(COMMAND pip install . WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
