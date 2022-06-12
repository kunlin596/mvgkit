add_library(mvgkit_common SHARED math.h camera.h camera.cc geometry.h
                                 geometry.cc ransac.h)
target_link_libraries(mvgkit_common PUBLIC Eigen3::Eigen Sophus::Sophus
                                           opencv_core mvgkit_io)
install(TARGETS mvgkit_common LIBRARY DESTINATION lib)

# Headers
set(COMMON_HEADERS camera.h geometry.h random.h transformation.h math.h
                   ransac.h)
install(FILES ${COMMON_HEADERS} DESTINATION include/mvgkit/common)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(_mvgkit_geometry_cppimpl pygeometry.cc)
  target_link_libraries(_mvgkit_geometry_cppimpl PRIVATE mvgkit_common)
  target_compile_definitions(_mvgkit_geometry_cppimpl
                             PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})

  pybind11_add_module(_mvgkit_camera_cppimpl pycamera.cc)
  target_link_libraries(_mvgkit_camera_cppimpl PRIVATE mvgkit_common)
  target_compile_definitions(_mvgkit_camera_cppimpl
                             PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
endif()
