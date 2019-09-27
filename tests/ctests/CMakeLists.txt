cmake_minimum_required(VERSION 3.10)

# collect all of the test sources
file(GLOB test_source "${CMAKE_SOURCE_DIR}/tests/ctests/*.cpp")

# setup test executable
add_executable(runAleTests ${test_source})
target_link_libraries(runAleTests
                      PUBLIC
                      ale
                      GSL::gsl
                      GSL::gslcblas
                      Eigen3::Eigen
                      gtest)

gtest_discover_tests(runAleTests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/ctests)
