add_library(mvgkit_io SHARED json_utils.h json_utils.cc)
target_link_libraries(mvgkit_io PUBLIC Eigen3::Eigen
                                       nlohmann_json::nlohmann_json)

set(IO_HEADERS json_utils.h)
install(FILES ${IO_HEADERS} DESTINATION include/mvgkit/io)
