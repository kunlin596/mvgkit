add_library(mvgkit_camera SHARED camera.h camera.cc)
target_link_libraries(mvgkit_camera PUBLIC opencv_core Eigen3::Eigen
                                           Sophus::Sophus)

if(WIN32)
  set(LIB_NAME "libmvgkit_camera")
else()
  set(LIB_NAME "mvgkit_camera")
endif()
set_target_properties(mvgkit_camera PROPERTIES OUTPUT_NAME ${LIB_NAME})

install(TARGETS mvgkit_camera LIBRARY DESTINATION lib)
install(FILES camera.h DESTINATION include/mvgkit)
