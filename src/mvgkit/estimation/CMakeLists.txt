file(GLOB MVGKIT_ESTIMATION_HEADER_FILES *.h)
file(GLOB MVGKIT_ESTIMATION_SOURCE_FILES *.cc)

add_library(mvgkit_estimation SHARED ${MVGKIT_ESTIMATION_HEADER_FILES}
                                     ${MVGKIT_ESTIMATION_SOURCE_FILES})

target_link_libraries(mvgkit_estimation mvgkit_common ${CERES_LIBRARIES})

install(TARGETS mvgkit_estimation LIBRARY DESTINATION lib)

# Headers
set(ALGORITHMS_HEADERS pnp.h)
install(FILES ${ALGORITHMS_HEADERS} DESTINATION include/mvgkit/estimation)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(pymvgkit_estimation python/pyestimation.cc)
  target_link_libraries(pymvgkit_estimation PRIVATE mvgkit_estimation)
  target_compile_definitions(pymvgkit_estimation
                             PRIVATE VERSION_INFO=${DMVGKIT_VERSION_INFO})
endif()
