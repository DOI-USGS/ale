if (NOT TARGET gtest)
  set(GOOGLETEST_ROOT googletest/googletest CACHE STRING "Google Test source root")

  include_directories(SYSTEM
      ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}
      ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/include
      )

  set(GOOGLETEST_SOURCES
      ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest-all.cc
      ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest_main.cc
      )

  foreach(_source ${GOOGLETEST_SOURCES})
      set_source_files_properties(${_source} PROPERTIES GENERATED 1)
  endforeach()

  add_library(gtest ${GOOGLETEST_SOURCES})
endif()

if (NOT TARGET gmock)
  set(GOOGLEMOCK_ROOT googletest/googlemock CACHE STRING "Google Mock source root")

  include_directories(SYSTEM
      ${PROJECT_SOURCE_DIR}/${GOOGLEMOCK_ROOT}
      ${PROJECT_SOURCE_DIR}/${GOOGLEMOCK_ROOT}/include
      )

  set(GOOGLEMOCK_SOURCES
      ${PROJECT_SOURCE_DIR}/${GOOGLEMOCK_ROOT}/src/gmock-all.cc
      ${PROJECT_SOURCE_DIR}/${GOOGLEMOCK_ROOT}/src/gmock_main.cc
      )

  foreach(_source ${GOOGLEMOCK_SOURCES})
      set_source_files_properties(${_source} PROPERTIES GENERATED 1)
  endforeach()

  add_library(gmock ${GOOGLEMOCK_SOURCES})
endif()
