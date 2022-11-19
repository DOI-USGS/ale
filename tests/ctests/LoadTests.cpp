#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "ale/Load.h"

#include <nlohmann/json.hpp>

#include <stdexcept>

using json = nlohmann::json;
using namespace std;
using ::testing::HasSubstr;

TEST(PyInterfaceTest, LoadInvalidLabel) {
  std::string label = "Not a Real Label";
  EXPECT_THROW(ale::load(label), invalid_argument);
}


TEST(PyInterfaceTest, LoadValidLabel) {
  std::string label = "../pytests/data/EN1072174528M/EN1072174528M_spiceinit.lbl";
  ale::load(label, "", "isis");
}

TEST(PyInterfaceTest, LoadValidLabelOnlyIsisSpice) {
  std::string label = "../pytests/data/EN1072174528M/EN1072174528M_spiceinit.lbl";
  ale::load(label, "", "isis", false, true, false);
}

TEST(PyInterfaceTest, LoadValidLabelOnlyNaifSpice) {
  std::string label = "../pytests/data/EN1072174528M/EN1072174528M_spiceinit.lbl";
  try {
    ale::load(label, "", "isis", false, false, true);
    FAIL() << "Should not have been able to generate an ISD" << endl;
  }
  catch (exception &e) {
    EXPECT_THAT(e.what(), HasSubstr("No Valid instrument found for label."));
  }
}