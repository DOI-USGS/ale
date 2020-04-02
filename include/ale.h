#ifndef ALE_INCLUDE_ALE_H
#define ALE_INCLUDE_ALE_H

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace ale {

  /// Interpolation enum for defining different methods of interpolation
  enum interpolation {
    /// Interpolate using linear interpolation
    LINEAR = 0,
    /// Interpolate using a cubic spline
    SPLINE = 1,
    /// Interpolate using Lagrange polynomials up to 8th order
    LAGRANGE = 2,
  };


  // Temporarily moved interpolation and associated helper functions over from States. Will be
  // move to interpUtils in the future.

  /** The following helper functions are used to calculate the reduced states cache and cubic hermite
  to interpolate over it. They were migrated, with minor modifications, from
  Isis::NumericalApproximation **/

  /** Determines the lower index for the interpolation interval. */
  int interpolationIndex(const std::vector<double> &times, double interpTime);

  /** Evaluates a cubic hermite at time, interpTime, between the appropriate two points in x. **/
  double evaluateCubicHermite(const double interpTime, const std::vector<double>& derivs,
                              const std::vector<double>& x, const std::vector<double>& y);

  /** Evaluate velocities using a Cubic Hermite Spline at a time a, within some interval in x, **/
  double evaluateCubicHermiteFirstDeriv(const double interpTime, const std::vector<double>& deriv,
                                       const std::vector<double>& times, const std::vector<double>& y);

  double lagrangeInterpolate(const std::vector<double>& times, const std::vector<double>& values,
                             double time, int order=8);
  double lagrangeInterpolateDerivative(const std::vector<double>& times, const std::vector<double>& values,
                                       double time, int order=8);

  /**
   *@brief Interpolates the spacecrafts position along a path generated from a set of points,
                times, and a time of observation
   *@param points A double vector of points
   *@param times A double vector of times
   *@param time A double to use as the time of observation
   *@param interp An interpolation enum dictating what type of interpolation to use
   *@param d The order of the derivative to generate when interpolating
                      (Currently supports 0, 1, and 2)
   *@return
   */
  double interpolate(std::vector<double> points, std::vector<double> times, double time, interpolation interp, int d);
  std::string loads(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);

  nlohmann::json load(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);


}

#endif // ALE_H
