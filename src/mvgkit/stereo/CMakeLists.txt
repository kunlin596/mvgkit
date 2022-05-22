add_library(mvgkit_stereo_common SHARED common.h common.cc)
target_link_libraries(mvgkit_stereo_common PUBLIC Eigen3::Eigen Sophus::Sophus)

add_library(mvgkit_stereo_eightpoint SHARED eight_point.h eight_point.cc)
target_link_libraries(mvgkit_stereo_eightpoint PUBLIC mvgkit_stereo_common
                                                      mvgkit_common)

add_library(mvgkit_stereo_fundamental SHARED fundamental.h fundamental.cc
                                             essential.h essential.cc)
target_link_libraries(mvgkit_stereo_fundamental PUBLIC mvgkit_stereo_eightpoint
                                                       ${CERES_LIBRARIES})

set(MVGKIT_STEREO_LIBS mvgkit_stereo_common mvgkit_stereo_eightpoint
                       mvgkit_stereo_fundamental)
set(MVGKIT_STEREO_HDRS common.h eight_point.h essential.h fundamental.h)
install(TARGETS ${MVGKIT_STEREO_LIBS} LIBRARY DESTINATION lib)
install(FILES ${MVGKIT_STEREO_HDRS} DESTINATION include/mvgkit/stereo)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter Development)
  find_package(pybind11 CONFIG)

  pybind11_add_module(_mvgkit_stereo_cppimpl pystereo.cc)
  target_link_libraries(_mvgkit_stereo_cppimpl
                        PRIVATE mvgkit_stereo_fundamental)
  target_compile_definitions(_mvgkit_stereo_cppimpl
                             PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
endif()
