#ifndef EAL_INCLUDE_EAL_H
#define EAL_INCLUDE_EAL_H

#include <json.hpp>
#include <string>
#include <vector>

namespace eal {

  enum interpolation {
    linear,
    spline
  };

  nlohmann::json constructStateFromIsd(const std::string positionRotationData);

  std::vector<double> getPosition(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  interpolation interp, double time);
  std::vector<double> getVelocity(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  interpolation interp, double time);

  std::vector<double> getPosition(std::vector<double> coeffs, double time);
  std::vector<double> getVelocity(std::vector<double> coeffs, double time);

  std::vector<double> getRotation(std::string from, std::string to,
                                  std::vector<std::vector<double>> rotations,
                                  std::vector<double> times,
                                  interpolation interp, double time);
  std::vector<double> getAngularVelocity(std::string from, std::string to,
                                         std::vector<std::vector<double>> rotations,
                                         std::vector<double> times,
                                         interpolation interp, double time);

  std::vector<double> getRotation(std::string from, std::string to,
                                  std::vector<double> coefficients, double time);
  std::vector<double> getAngularVelocity(std::string from, std::string to,
                                         std::vector<double> coefficients, double time);

}

#endif // EAL_H
