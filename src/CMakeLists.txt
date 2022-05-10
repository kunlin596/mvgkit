add_subdirectory(mvgkit)
set(CMAKE_CXX_FLAGS "-O3")

pybind11_add_module(_mvgkit_cppimpl pythonbindings.cc)
target_link_libraries(_mvgkit_cppimpl PRIVATE mvgkit_camera mvgkit_stereo
                                              mvgkit_mapping)
target_compile_definitions(_mvgkit_cppimpl
                           PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
