
#include "eal.h"

#include <json.hpp>
#include <string>

using json = nlohmann::json;


// Parsing the JSON
std::string constructStateFromIsd(const std::string positionRotationData) {
   // Parse the position and rotation data from isd
   json isd = json::parse(positionRotationData);
   json state;

   state["m_w"] = isd.at("w");
   state["m_x"] = isd.at("x");
   state["m_y"] = isd.at("y");
   state["m_z"] = isd.at("z");

   return state.dump();
 }

// Positional Functions

// Position Data Functions
vector<double> getPosition(vector<vector<double>> coords, vector<double> coeffs, char *interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getVelocity(vector<vector<double>> coords, vector<double> coeffs, char *interp, double time, bool interpolation) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

// Postion Function Functions
vector<double> getPosition(vector<double> coeffs, char *interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getVelocity(vector<double> coeffs, char *interp, double time, bool interpolation) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

// Rotation Data Functions

// Rotation Data Functions
vector<double> getRotation(char *from, char *to, vector<vector<double>> rotations, vector<double> times, char *interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getAngularVelocity(char *from, char *to, vector<vector<double>> rotations, vector<double> times, char *interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

// Rotation Function Functions
vector<double> getRotation(char *from, char *to, vector<double> coefficients, char *interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getAngularVelocity(char *from, char *to, vector<double> coefficients, char *interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}
