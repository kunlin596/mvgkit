#pragma once
#include <Eigen/Dense>
#include <boost/filesystem.hpp>

namespace mvgkit {
namespace testing_utils {

namespace bfs = boost::filesystem;

bfs::path
resolveDataPath(const bfs::path& dataname);

} // testing
} // mvgkit
