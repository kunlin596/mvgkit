add_library(mvgkit_camera SHARED camera.h camera.cc)
target_link_libraries(mvgkit_camera PUBLIC opencv_core Eigen3::Eigen
                                           Sophus::Sophus)

if(WIN32)
  set(LIB_NAME "libmvgkit_camera")
else()
  set(LIB_NAME "mvgkit_camera")
endif()
set_target_properties(mvgkit_camera PROPERTIES OUTPUT_NAME ${LIB_NAME})

set(COMMON_HEADERS camera.h random.h transformation.h math.h)

install(TARGETS mvgkit_camera LIBRARY DESTINATION lib)
install(FILES ${COMMON_HEADERS} DESTINATION include/mvgkit/common)

if(ENABLE_MVGKIT_TESTS)
  add_executable(mvgkit_testMath tests/testMath.cc)
  target_link_libraries(mvgkit_testMath Eigen3::Eigen GTest::gtest
                        GTest::gtest_main)
  add_test(mvgkit_testMath mvgkit_testMath)
endif()
