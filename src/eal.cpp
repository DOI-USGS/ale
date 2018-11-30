#include "eal.h"

using namespace eal;

// Positional Functions

// Position Data Functions
vector<double> getPosition(vector<vector<double>> coords, vector<double> times,
                           string interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getVelocity(vector<vector<double>> coords, vector<double> times,
                           string interp, double time, bool interpolation) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

// Postion Function Functions
vector<double> getPosition(vector<double> coeffs, string interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getVelocity(vector<double> coeffs, string interp, double time,
                           bool interpolation) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

// Rotation Data Functions

// Rotation Data Functions
vector<double> getRotation(string from, string to, vector<vector<double>> rotations,
                           vector<double> times, string interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getAngularVelocity(string from, string to, vector<vector<double>> rotations,
                                  vector<double> times, string interp, double time, bool interpolation) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

// Rotation Function Functions
vector<double> getRotation(string from, string to, vector<double> coefficients,
                           string interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getAngularVelocity(string from, string to, vector<double> coefficients,
                                  string interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}
