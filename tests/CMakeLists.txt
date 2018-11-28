cmake_minimum_required(VERSION 3.10)

add_dependencies(usgsephem usgsephem)

file(GLOB test_source "${CMAKE_SOURCE_DIR}/tests/*.cpp")

# Link runSpiceRefactorTests with what we want to test and the GTest and pthread library
add_executable(runSpiceRefactorTests
               TestMain.cpp
               ${test_source})

target_link_libraries(runSpiceRefactorTests usgsephem ${ALLLIBS} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)

gtest_discover_tests(runSpiceRefactorTests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/../tests)