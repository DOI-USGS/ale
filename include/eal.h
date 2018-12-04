#ifndef EAL_INCLUDE_EAL_H
#define EAL_INCLUDE_EAL_H

#include <json.hpp>
#include <string>
#include <vector>

#include <gsl/gsl_interp.h>

namespace eal {

  enum derivation {
    none,
    first,
    second
  };

  nlohmann::json constructStateFromIsd(const std::string positionRotationData);

  std::vector<double> getPosition(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, const gsl_interp_type* interp);

  std::vector<double> getVelocity(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, const gsl_interp_type* interp);

  std::vector<double> getPosition(std::vector<double> coeffs, double time);
  std::vector<double> getVelocity(std::vector<double> coeffs, double time);

  std::vector<double> getRotation(std::string from, std::string to,
                                  std::vector<std::vector<double>> rotations,
                                  std::vector<double> times,
                                  double time, const gsl_interp_type* interp);
  std::vector<double> getAngularVelocity(std::string from, std::string to,
                                         std::vector<std::vector<double>> rotations,
                                         std::vector<double> times,
                                         double time, const gsl_interp_type* interp);

  std::vector<double> getRotation(std::string from, std::string to,
                                  std::vector<double> coefficients, double time);
  std::vector<double> getAngularVelocity(std::string from, std::string to,
                                         std::vector<double> coefficients, double time);

  double interpolate(std::vector<double> points, std::vector<double> times, double time, const gsl_interp_type* interp, derivation d);


}

#endif // EAL_H
