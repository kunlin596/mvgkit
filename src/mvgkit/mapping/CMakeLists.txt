# add_subdirectory(mapper)

add_library(
  mvgkit_mapping SHARED
  common/landmark.h
  common/landmark.cc
  common/frame.h
  common/frame.cc
  common/reconstruction.h
  common/reconstruction.cc
  common/bundle_adjustment.h
  common/bundle_adjustment.cc)

if(WIN32)
  set(LIB_NAME "libmvgkit_mapping")
else()
  set(LIB_NAME "mvgkit_mapping")
endif()
set_target_properties(mvgkit_mapping PROPERTIES OUTPUT_NAME ${LIB_NAME})
target_link_libraries(mvgkit_mapping opencv_core Sophus::Sophus)

install(TARGETS mvgkit_mapping LIBRARY DESTINATION lib)
install(FILES landmark.h DESTINATION include/mvgkit/mapping)
