#ifndef ALE_INCLUDE_ALE_H
#define ALE_INCLUDE_ALE_H

#include <string>
#include <vector>


#include <nlohmann/json.hpp>

namespace ale {

  std::string loads(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);

  nlohmann::json load(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);


}

#endif // ALE_H
