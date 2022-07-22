#pragma once
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

namespace mvgkit {
namespace io {

using json = nlohmann::json;

json
readJson(const std::string& path);

json
readJson(const boost::filesystem::path& path);

Eigen::ArrayXXf
readArray(const json& j, const std::string& key);

} // io
} // mvgkit
