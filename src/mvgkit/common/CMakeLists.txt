file(GLOB MVGKIT_COMMON_HEADER_FILES *.h)
file(GLOB MVGKIT_COMMON_SOURCE_FILES *.cc)

add_library(mvgkit_common SHARED ${MVGKIT_COMMON_HEADER_FILES}
                                 ${MVGKIT_COMMON_SOURCE_FILES})

message(${MVGKIT_COMMON_HEADER_FILES})
message(${MVGKIT_COMMON_SOURCE_FILES})

target_link_libraries(
  mvgkit_common PUBLIC Eigen3::Eigen Sophus::Sophus opencv_core
                       nlohmann_json::nlohmann_json Boost::filesystem fmt::fmt)

install(TARGETS mvgkit_common LIBRARY DESTINATION lib)

# Headers
install(FILES ${MVGKIT_COMMON_HEADER_FILES} DESTINATION include/mvgkit/common)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(pymvgkit_common python/pycommon.cc)

  target_link_libraries(pymvgkit_common PRIVATE mvgkit_common)
  target_compile_definitions(pymvgkit_common
                             PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
endif()
