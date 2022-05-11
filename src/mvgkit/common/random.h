#pragma once
#include <random>
#include <unordered_set>

namespace mvgkit {
namespace common {

std::unordered_set<size_t>
randomChoice(size_t min, size_t max, size_t count)
{
  assert((max - min + 1) <= count);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> d(min, max);
  std::unordered_set<size_t> generated_set;
  while (generated_set.size() < count) {
    generated_set.insert(d(gen));
  }
  return generated_set;
}

} // common
} // mvgkit
