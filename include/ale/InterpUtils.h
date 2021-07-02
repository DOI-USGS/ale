#ifndef ALE_INCLUDE_INTERP_UTILS_H
#define ALE_INCLUDE_INTERP_UTILS_H

#include <vector>

#include "ale/Vectors.h"

namespace ale {

  enum RotationInterpolation {
    SLERP, // Spherical interpolation
    NLERP // Normalized linear interpolation
  };

  /// Interpolation enum for defining different methods of interpolation
  enum PositionInterpolation {
    /// Interpolate using linear interpolation
    LINEAR = 0,
    /// Interpolate using a cubic spline
    SPLINE = 1,
    /// Interpolate using Lagrange polynomials up to 8th order
    LAGRANGE = 2,
  };

  /**
   * Linearly interpolate between two values.
   *
   * @param x The first value.
   * @param y The second value.
   * @param t The distance to interpolate. 0 is x and 1 is y.
   */
  double linearInterpolate(double x, double y, double t);

  /**
   * Linearly interpolate between two vectors.
   *
   * @param x The first vectors.
   * @param y The second vectors.
   * @param t The distance to interpolate. 0 is x and 1 is y.
   */
  std::vector<double> linearInterpolate(const std::vector<double> &x, const std::vector<double> &y, double t);

  /**
   * Linearly interpolate between two 3D vectors.
   *
   * @param x The first vectors.
   * @param y The second vectors.
   * @param t The distance to interpolate. 0 is x and 1 is y.
   */
  Vec3d linearInterpolate(const Vec3d &x, const Vec3d &y, double t);

  /**
   * Compute the index of the first time to use when interpolating at a given time.
   *
   * @param times The ordered vector of times to search. Must have at least 2 times.
   * @param interpTime The time to search for the interpolation index of.
   * @return int The index of the time that comes before interpTime. If there is
   *             no time that comes before interpTime, then returns 0. If all
   *             times come before interpTime, then returns the second to last
   *             index.
   */
  int interpolationIndex(const std::vector<double> &times, double interpTime);

  /**
   * Merge, sort, and remove duplicates from two vectors
   */
   std::vector<double> orderedVecMerge(const std::vector<double> &x, const std::vector<double> &y);

   /**
    * Evaluates a cubic hermite at time, interpTime, between the appropriate two points in x.
    *
    * migrated from Isis::NumericalApproximation
    */
   double evaluateCubicHermite(const double interpTime, const std::vector<double>& derivs,
                            const std::vector<double>& x, const std::vector<double>& y);

   /**
    * Evaluate velocities using a Cubic Hermite Spline at a time a, within some interval in x.
    *
    * migrated from Isis::NumericalApproximation
    */
   double evaluateCubicHermiteFirstDeriv(const double interpTime, const std::vector<double>& deriv,
                                     const std::vector<double>& times, const std::vector<double>& y);

   /**
    * Interpolate a set of values using lagrange polynomials.
    *
    * @param times The vector of times to interpolate over
    * @param values The vector of values to interpolate between
    * @param time The time to interpolate at
    * @param order The order of the lagrange polynomials to use
    */
   double lagrangeInterpolate(const std::vector<double>& times, const std::vector<double>& values,
                           double time, int order=8);

   /**
    * Interpolate the first derivative of a set of values using lagrange polynomials.
    *
    * @param times The vector of times to interpolate over
    * @param values The vector of values to interpolate between
    * @param time The time to interpolate at
    * @param order The order of the lagrange polynomials to use
    */
   double lagrangeInterpolateDerivative(const std::vector<double>& times, const std::vector<double>& values,
                                     double time, int order=8);

   /**
    *@brief Interpolates the spacecraft's position along a path generated from a set of points,
              times, and a time of observation
    *@param points A double vector of points
    *@param times A double vector of times
    *@param time A double to use as the time of observation
    *@param interp An interpolation enum dictating what type of interpolation to use
    *@param d The order of the derivative to generate when interpolating
                    (Currently supports 0, 1, and 2)
    *@return
   */
   double interpolate(std::vector<double> points, std::vector<double> times, double time, PositionInterpolation interp, int d);

}

#endif
