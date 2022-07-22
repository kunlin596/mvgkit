
#include "../mvgkit/common/json_utils.h"
#include "utils.h"
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <fstream>
#include <gtest/gtest.h>

namespace {
using namespace Eigen;
using namespace mvgkit;
}

TEST(mvgkit_IO_JsonUtils, testReadArray)
{
  using namespace mvgkit::io;
  namespace bfs = boost::filesystem;
  const bfs::path filePath = testing_utils::resolveDataPath("dummy.json");
  json j = io::readJson(filePath);
  ArrayXXf arr = readArray(j, "arr");
  EXPECT_EQ(arr.rows(), 3);
  EXPECT_EQ(arr.cols(), 2);
  EXPECT_NEAR(arr(0, 0), 1.0f, 1e-7);
  EXPECT_NEAR(arr(0, 1), 2.0f, 1e-7);
  EXPECT_NEAR(arr(1, 0), 3.0f, 1e-7);
  EXPECT_NEAR(arr(1, 1), 4.0f, 1e-7);
  EXPECT_NEAR(arr(2, 0), 5.0f, 1e-7);
  EXPECT_NEAR(arr(2, 1), 6.0f, 1e-7);
}
