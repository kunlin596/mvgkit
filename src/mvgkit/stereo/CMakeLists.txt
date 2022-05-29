add_library(
  mvgkit_stereo SHARED
  common.h
  common.cc
  triangulation.h
  triangulation.cc
  eight_point.h
  eight_point.cc
  fundamental.h
  fundamental.cc
  essential.h
  essential.cc)
target_link_libraries(mvgkit_stereo PUBLIC mvgkit_common Eigen3::Eigen
                                           Sophus::Sophus ${CERES_LIBRARIES})

set(MVGKIT_STEREO_LIBS mvgkit_stereo)
set(MVGKIT_STEREO_HDRS common.h eight_point.h essential.h fundamental.h
                       triangulation.h)
install(TARGETS ${MVGKIT_STEREO_LIBS} LIBRARY DESTINATION lib)
install(FILES ${MVGKIT_STEREO_HDRS} DESTINATION include/mvgkit/stereo)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  find_package(Python COMPONENTS Interpreter Development)
  find_package(pybind11 CONFIG)

  pybind11_add_module(_mvgkit_stereo_cppimpl pystereo.cc)
  target_link_libraries(_mvgkit_stereo_cppimpl PRIVATE mvgkit_stereo)
  target_compile_definitions(_mvgkit_stereo_cppimpl
                             PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
endif()
