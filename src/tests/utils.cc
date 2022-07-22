#include "utils.h"
#include "../mvgkit/common/json_utils.h"
#include <fmt/format.h>

namespace mvgkit {
namespace testing_utils {

bfs::path
resolveDataPath(const bfs::path& dataname)
{
  const char* testDataDir = std::getenv("MVGKIT_TEST_DATA_DIR");
  if (testDataDir == nullptr) {
    throw std::runtime_error("Environment variable `MVGKIT_TEST_DATA_DIR` is not set!");
  }
  bfs::path rootPath = bfs::path(std::string(testDataDir));
  bfs::path filePath = rootPath / dataname;
  if (!bfs::exists(filePath)) {
    throw std::runtime_error(fmt::format("File {} not found!", filePath.string()));
  }
  return filePath;
}

} // testing
} // mvgkit
