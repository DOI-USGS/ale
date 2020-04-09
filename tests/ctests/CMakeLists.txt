cmake_minimum_required(VERSION 3.10)

# collect all of the test sources
set (ALE_TEST_SOURCE ${CMAKE_SOURCE_DIR}/tests/ctests/IsdTests.cpp
                     ${CMAKE_SOURCE_DIR}/tests/ctests/OrientationsTests.cpp
                     ${CMAKE_SOURCE_DIR}/tests/ctests/RotationTests.cpp
                     ${CMAKE_SOURCE_DIR}/tests/ctests/StatesTests.cpp
                     ${CMAKE_SOURCE_DIR}/tests/ctests/TestInterpUtils.cpp
                     ${CMAKE_SOURCE_DIR}/tests/ctests/TestMain.cpp)

if(ALE_BUILD_LOAD)
  list(APPEND ALE_TEST_SOURCE ${CMAKE_SOURCE_DIR}/tests/ctests/LoadTests.cpp)
endif()

# setup test executable
add_executable(runAleTests ${ALE_TEST_SOURCE})
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
