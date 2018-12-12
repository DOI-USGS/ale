#ifndef ALE_INCLUDE_ALE_H
#define ALE_INCLUDE_ALE_H

#include <json.hpp>
#include <string>
#include <vector>

#include <gsl/gsl_interp.h>

namespace ale {

  enum interpolation {
    linear,
    spline
  };

  nlohmann::json constructStateFromIsd(const std::string positionRotationData);

  std::vector<double> getPosition(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, interpolation interp);

  std::vector<double> getVelocity(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, const interpolation interp);

  std::vector<double> getPosition(std::vector<std::vector<double>> coeffs, double time);
  std::vector<double> getVelocity(std::vector<std::vector<double>> coeffs, double time);

  std::vector<double> getRotation(std::vector<std::vector<double>> rotations,
                                  std::vector<double> times,
                                  double time, interpolation interp);
  std::vector<double> getAngularVelocity(std::vector<std::vector<double>> rotations,
                                         std::vector<double> times,
                                         double time, interpolation interp);

  std::vector<double> getRotation(std::string from, std::string to,
                                  std::vector<double> coefficients, double time);
  std::vector<double> getAngularVelocity(std::string from, std::string to,
                                         std::vector<double> coefficients, double time);
  double evaluatePolynomial(std::vector<double> coeffs, double time, int d);
  double interpolate(std::vector<double> points, std::vector<double> times, double time, interpolation interp, int d);
}

#endif // ALE_H
