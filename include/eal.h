#ifndef EAL_INCLUDE_EAL_H
#define EAL_INCLUDE_EAL_H

#include <json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

using namespace std;

namespace eal {

  enum interpolation {
    linear,
    spline
  };

  json constructStateFromIsd(const string positionRotationData);

  vector<double> getPosition(vector<vector<double>> coords, vector<double> times,
                             interpolation interp, double time);
  vector<double> getVelocity(vector<vector<double>> coords, vector<double> times,
                             interpolation interp, double time);

  vector<double> getPosition(vector<double> coeffs, double time);
  vector<double> getVelocity(vector<double> coeffs, double time);

  vector<double> getRotation(string from, string to, vector<vector<double>> rotations,
                             vector<double> times, interpolation interp, double time);
  vector<double> getAngularVelocity(string from, string to, vector<vector<double>> rotations,
                                    vector<double> times, interpolation interp, double time);

  vector<double> getRotation(string from, string to, vector<double> coefficients,
                             double time);
  vector<double> getAngularVelocity(string from, string to, vector<double> coefficients,
                                    double time);

}

#endif // EAL_H
