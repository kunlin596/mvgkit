#include "json_utils.h"
#include <boost/assert.hpp>
#include <cassert>
#include <fmt/format.h>
#include <fstream>

namespace {
namespace bfs = boost::filesystem;
}

namespace mvgkit {
namespace io {

json
readJson(const std::string& path)
{
  return readJson(bfs::path(path));
}

json
readJson(const bfs::path& path)
{
  if (!bfs::is_regular_file(path) && path.extension() == ".json") {
    throw std::runtime_error("File is not opened!");
  }
  std::fstream ifs(path.string());
  json j;
  ifs >> j;
  return j;
}

Eigen::ArrayXXf
readArray(const json& j, const std::string& key)
{
  if (!j.contains(key.c_str()) || !j[key.c_str()].is_array()) {
    throw std::runtime_error(fmt::format("No such key: {}!", key));
  };

  json::array_t data = j[key.c_str()];
  size_t numRows = data.size();
  BOOST_ASSERT(numRows > 0);
  size_t numCols = data[0].size();
  BOOST_ASSERT(numCols > 0);

  std::vector<float> buf(numRows * numCols);
  size_t i = 0;
  for (auto row : data) {
    for (auto el : row) {
      buf[i++] = el.get<float>();
    }
  }
  return Eigen::Map<Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
    buf.data(), numRows, numCols);
}

} // io
} // mvgkit
