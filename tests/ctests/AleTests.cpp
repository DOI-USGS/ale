#include "gtest/gtest.h"

#include "Ale.h"

#include <stdexcept>

using namespace std;

TEST(PyInterfaceTest, LoadInvalidLabel) {
  std::string label = "Not a Real Label";
  EXPECT_THROW(ale::load(label), invalid_argument);
}


TEST(PyInterfaceTest, LoadValidLabel) {
  std::string label = "../pytests/data/EN1072174528M/EN1072174528M_spiceinit.lbl";
  ale::load(label, "", "isis");
}
