cmake_minimum_required(VERSION 3.10)

# collect all of the test sources
file(GLOB test_source "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# setup test executable
add_executable(runEALTests ${test_source})
target_link_libraries(runEALTests eal ${GSL_LIBRARIES} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

target_include_directories(runEALTests
                           PRIVATE
                           ${GSL_INCLUDE_DIRS}
                           PUBLIC
                           ${EAL_INCLUDE_DIRS})

gtest_discover_tests(runEALTests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests)
