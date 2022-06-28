add_library(mvgkit_algorithms SHARED pnp.h pnp.cc)
target_link_libraries(mvgkit_algorithms mvgkit_common ${CERES_LIBRARIES})

install(TARGETS mvgkit_algorithms LIBRARY DESTINATION lib)

# Headers
set(ALGORITHMS_HEADERS pnp.h)
install(FILES ${ALGORITHMS_HEADERS} DESTINATION include/mvgkit/algorithms)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(_mvgkit_algorithms_cppimpl pyalgorithms.cc)
  target_link_libraries(_mvgkit_algorithms_cppimpl PRIVATE mvgkit_algorithms)
  target_compile_definitions(_mvgkit_algorithms_cppimpl
                             PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
endif()
