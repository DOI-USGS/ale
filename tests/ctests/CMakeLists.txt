cmake_minimum_required(VERSION 3.10)

# collect all of the test sources
file(GLOB test_source "${CMAKE_SOURCE_DIR}/tests/ctests/*.cpp")

# setup test executable
add_executable(runAleTests ${test_source})
target_link_libraries(runAleTests
                      PRIVATE
                      ale
                      Eigen3::Eigen
                      gtest
                      gmock
                      Threads::Threads
                      nlohmann_json::nlohmann_json
                      )

gtest_discover_tests(runAleTests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/ctests)
