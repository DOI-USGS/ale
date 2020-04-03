#ifndef ALE_INCLUDE_ALE_H
#define ALE_INCLUDE_ALE_H

#include <nlohmann/json.hpp>

#include <string>

namespace ale {
  std::string loads(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);

  nlohmann::json load(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);
}

#endif // ALE_H
