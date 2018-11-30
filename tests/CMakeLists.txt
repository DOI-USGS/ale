cmake_minimum_required(VERSION 3.10)

file(GLOB test_source "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# Link runEALTests with what we want to test and the GTest and pthread library
add_executable(runEALTests
               TestMain.cpp
               ${test_source})

target_link_libraries(runEALTests usgsephem ${ALLLIBS} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

gtest_discover_tests(runEALTests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/../tests)
