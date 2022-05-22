set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/src/tests/data)
message(STATUS "Adding ${TEST_DATA_DIR} to test binary.")

# Utils
add_library(mvgkit_testing_utils SHARED utils.h utils.cc)
target_link_libraries(mvgkit_testing_utils PUBLIC Eigen3::Eigen mvgkit_io
                                                  Boost::filesystem fmt::fmt)

# IO
add_executable(test_io test_io.cc)
target_link_libraries(test_io PRIVATE Eigen3::Eigen mvgkit_io GTest::GTest
                                      GTest::Main mvgkit_testing_utils)
add_test(mvgkit_testIO test_io)
set_tests_properties(
  mvgkit_testIO PROPERTIES ENVIRONMENT "MVGKIT_TEST_DATA_DIR=${TEST_DATA_DIR}")

# Math
add_executable(test_math test_math.cc)
target_link_libraries(test_math Eigen3::Eigen GTest::GTest GTest::Main)
add_test(mvgkit_testMath test_math)

# Geometry
add_executable(test_geometry test_geometry.cc)
target_link_libraries(test_geometry Eigen3::Eigen mvgkit_common
                      mvgkit_testing_utils GTest::GTest GTest::Main)
add_test(mvgkit_testGeometry test_geometry WORKING_DIRECTORY ${TEST_DATA_DIR})

# Fundamental
add_executable(test_fundamental test_fundamental.cc)
target_link_libraries(
  test_fundamental
  mvgkit_io
  mvgkit_stereo_fundamental
  mvgkit_testing_utils
  mvgkit_common
  Eigen3::Eigen
  GTest::GTest
  GTest::Main)
add_test(mvgkit_testFundamental test_fundamental)

# Stereo Common
add_executable(test_stereo_common test_stereo_common.cc)
target_link_libraries(test_stereo_common mvgkit_io mvgkit_testing_utils
                      Eigen3::Eigen GTest::GTest GTest::Main)
add_test(mvgkit_testStereoCommon test_stereo_common)
