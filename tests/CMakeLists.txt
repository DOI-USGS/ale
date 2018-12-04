cmake_minimum_required(VERSION 3.10)

# collect all of the test sources
file(GLOB test_source "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# setup test executable
add_executable(runEALTests ${test_source})
target_link_libraries(runEALTests eal ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

gtest_discover_tests(runEALTests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
