file(GLOB MVGKIT_ESTIMATION_HEADER_FILES *.h)
file(GLOB MVGKIT_ESTIMATION_SOURCE_FILES *.cc)

add_library(mvgkit_estimation SHARED ${MVGKIT_ESTIMATION_HEADER_FILES}
                                     ${MVGKIT_ESTIMATION_SOURCE_FILES})

# `Ceres::ceres` (vs. the legacy ${CERES_LIBRARIES}) carries its include
# directories as usage requirements and is provided by both the system
# find_package and the FetchContent fallback.
target_link_libraries(mvgkit_estimation PUBLIC mvgkit_common Ceres::ceres)

install(TARGETS mvgkit_estimation LIBRARY DESTINATION lib)

# Headers
set(ALGORITHMS_HEADERS pnp.h)
install(FILES ${ALGORITHMS_HEADERS} DESTINATION include/mvgkit/estimation)

# Python bindings
if(BUILD_PYTHON_BINDINGS)
  pybind11_add_module(pymvgkit_estimation python/pyestimation.cc)
  target_link_libraries(pymvgkit_estimation PRIVATE mvgkit_estimation)
  target_compile_definitions(pymvgkit_estimation
                             PRIVATE VERSION_INFO=${MVGKIT_VERSION_INFO})
  # Resolve the co-located mvgkit shared libraries at import time (see the
  # common module's binding for the rationale).
  set_target_properties(pymvgkit_estimation PROPERTIES BUILD_RPATH "$ORIGIN"
                                                       INSTALL_RPATH "$ORIGIN")
endif()
