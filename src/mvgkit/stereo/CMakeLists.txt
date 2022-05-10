add_library(mvgkit_stereo SHARED eight_point.h eight_point.cc fundamental.h
                                 fundamental.cc essential.h essential.cc)
target_link_libraries(mvgkit_stereo PUBLIC opencv_core Eigen3::Eigen
                                           Sophus::Sophus fmt::fmt-header-only)

if(WIN32)
  set(LIB_NAME "libmvgkit_stereo")
else()
  set(LIB_NAME "mvgkit_stereo")
endif()
set_target_properties(mvgkit_stereo PROPERTIES OUTPUT_NAME ${LIB_NAME})

install(TARGETS mvgkit_stereo LIBRARY DESTINATION lib)
install(FILES eight_point.h essential.h fundamental.h
        DESTINATION include/mvgkit/stereo)
