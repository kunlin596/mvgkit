file(GLOB MVGKIT_FEATURES_HEADER_FILES *.h)
file(GLOB MVGKIT_FEATURES_SOURCE_FILES *.cc)

add_library(mvgkit_features SHARED ${MVGKIT_FEATURES_HEADER_FILES}
                                   ${MVGKIT_FEATURES_SOURCE_FILES})

target_link_libraries(mvgkit_features ${OPENCV2_LIBRARIES} Eigen3::Eigen)
