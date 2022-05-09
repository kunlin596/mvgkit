add_subdirectory(common)

pybind11_add_module(_mvgkit_cppimpl pythonbindings.cc)
target_link_libraries(_mvgkit_cppimpl PRIVATE mvgkit_camera)
target_compile_definitions(_mvgkit_cppimpl
                           PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
