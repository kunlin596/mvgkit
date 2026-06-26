set(TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/src/tests/data)
message(STATUS "Adding ${TEST_DATA_DIR} to test binary.")

# Utils
add_library(mvgkit_testing_utils SHARED utils.h utils.cc)
target_link_libraries(mvgkit_testing_utils PUBLIC Eigen3::Eigen mvgkit_common
                                                  Boost::filesystem fmt::fmt)

# IO
add_executable(test_io test_io.cc)
target_link_libraries(test_io PRIVATE Eigen3::Eigen GTest::gtest
                                      GTest::gtest_main mvgkit_testing_utils)
add_test(mvgkit_testIO test_io)

# Math
add_executable(test_math test_math.cc)
target_link_libraries(test_math Eigen3::Eigen GTest::gtest GTest::gtest_main)
add_test(mvgkit_testMath test_math)

# Geometry
add_executable(test_geometry test_geometry.cc)
target_link_libraries(test_geometry Eigen3::Eigen mvgkit_testing_utils
                      GTest::gtest GTest::gtest_main)
add_test(mvgkit_testGeometry test_geometry WORKING_DIRECTORY ${TEST_DATA_DIR})

# Transformation
add_executable(test_transformation test_transformation.cc)
target_link_libraries(test_transformation Eigen3::Eigen mvgkit_testing_utils
                      GTest::gtest GTest::gtest_main)
add_test(mvgkit_testTransformation test_transformation)

# Fundamental
add_executable(test_fundamental test_fundamental.cc)
target_link_libraries(test_fundamental mvgkit_estimation mvgkit_testing_utils
                      Eigen3::Eigen GTest::gtest GTest::gtest_main)
add_test(mvgkit_testFundamental test_fundamental)

# Triangulation
add_executable(test_triangulation test_triangulation.cc)
target_link_libraries(test_triangulation mvgkit_estimation mvgkit_testing_utils
                      Eigen3::Eigen GTest::gtest GTest::gtest_main)
add_test(mvgkit_testTriangulation test_triangulation)

# Essential matrix estimation
add_executable(test_essential test_essential.cc)
target_link_libraries(test_essential mvgkit_estimation mvgkit_testing_utils
                      Eigen3::Eigen GTest::gtest GTest::gtest_main)
add_test(mvgkit_testEssential test_essential)

# PnP
add_executable(test_pnp test_pnp.cc)
target_link_libraries(test_pnp mvgkit_testing_utils mvgkit_estimation
                      Eigen3::Eigen GTest::gtest GTest::gtest_main)
add_test(mvgkit_testPnP test_pnp)

# Every test loads fixtures through mvgkit_testing_utils, which resolves the
# data directory from MVGKIT_TEST_DATA_DIR. Set it for all tests (harmless for
# the few that don't read data) so `ctest` finds the fixtures.
set_tests_properties(
  mvgkit_testIO
  mvgkit_testMath
  mvgkit_testGeometry
  mvgkit_testTransformation
  mvgkit_testFundamental
  mvgkit_testTriangulation
  mvgkit_testEssential
  mvgkit_testPnP
  PROPERTIES ENVIRONMENT "MVGKIT_TEST_DATA_DIR=${TEST_DATA_DIR}")
