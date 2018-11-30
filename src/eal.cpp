
#include "eal.h"

#include <json.hpp>
#include <string>

using json = nlohmann::json;


std::string eal::constructStateFromIsd(const std::string positionRotationData) const {
   // Parse the position and rotation data from isd
   json isd = json::parse(positionRotationData);
   json state;

   state["m_w"] = isd.at("w");
   state["m_x"] = isd.at("x");
   state["m_y"] = isd.at("y");
   state["m_z"] = isd.at("z");

   return state.dump();
}
