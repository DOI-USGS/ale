cmake_minimum_required(VERSION 3.10)

# collect all of the test sources
file(GLOB test_source "${CMAKE_SOURCE_DIR}/tests/ctests/*.cpp")

# setup test executable
add_executable(runAleTests ${test_source})
target_link_libraries(runAleTests ale ${GSL_LIBRARIES} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

target_include_directories(runAleTests
                            PUBLIC
                            ${ALE_INCLUDE_DIRS}
                            ${GSL_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR})


gtest_discover_tests(runAleTests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/ctests)
