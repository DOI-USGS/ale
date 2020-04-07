#include "InterpUtils.h"

#include <exception>

#include <algorithm>
#include <unordered_set>

namespace ale {
  double linearInterpolate(double x, double y, double t) {
    return x + t * (y - x);
  }

  std::vector<double> linearInterpolate(const std::vector<double> &x, const std::vector<double> &y, double t) {
    if (x.size() != y.size()) {
      throw std::invalid_argument("X and Y vectors must be the same size.");
    }
    std::vector<double> interpVec(x.size());
    for (size_t i = 0; i < interpVec.size(); i++) {
      interpVec[i] = linearInterpolate(x[i], y[i], t);
    }
    return interpVec;
  }

  Vec3d linearInterpolate(const Vec3d &x, const Vec3d &y, double t) {
    Vec3d interpVec;

    interpVec.x = linearInterpolate(x.x, y.x, t);
    interpVec.y = linearInterpolate(x.y, y.y, t);
    interpVec.z = linearInterpolate(x.z, y.z, t);

    return interpVec;
  }

  int interpolationIndex(const std::vector<double> &times, double interpTime) {
    if (times.size() < 2){
      throw std::invalid_argument("There must be at least two times.");
    }
    auto nextTimeIt = std::upper_bound(times.begin(), times.end(), interpTime);
    if (nextTimeIt == times.end()) {
      --nextTimeIt;
    }
    if (nextTimeIt != times.begin()) {
      --nextTimeIt;
    }
    return std::distance(times.begin(), nextTimeIt);
  }

  std::vector<double> orderedVecMerge(const std::vector<double> &x, const std::vector<double> &y) {
    std::unordered_set<double> mergedSet;
    for (double val: x) {
      mergedSet.insert(val);
    }
    for (double val: y) {
      mergedSet.insert(val);
    }
    std::vector<double> merged;
    merged.assign(mergedSet.begin(), mergedSet.end());
    std::sort(merged.begin(), merged.end());
    return merged;
  }


  // Temporarily moved over from States.cpp. Will be moved into interpUtils in the future.

  /** The following helper functions are used to calculate the reduced states cache and cubic hermite
  to interpolate over it. They were migrated, with minor modifications, from
  Isis::NumericalApproximation **/


  /** Evaluates a cubic hermite at time, interpTime, between the appropriate two points in x. **/
  double evaluateCubicHermite(const double interpTime, const std::vector<double>& derivs,
                              const std::vector<double>& x, const std::vector<double>& y) {
    if( (derivs.size() != x.size()) || (derivs.size() != y.size()) ) {
       throw std::invalid_argument("EvaluateCubicHermite - The size of the first derivative vector does not match the number of (x,y) data points.");
    }

    // Find the interval in which "a" exists
    int lowerIndex = interpolationIndex(x, interpTime);

    double x0, x1, y0, y1, m0, m1;
    // interpTime is contained within the interval (x0,x1)
    x0 = x[lowerIndex];
    x1 = x[lowerIndex+1];
    // the corresponding known y-values for x0 and x1
    y0 = y[lowerIndex];
    y1 = y[lowerIndex+1];
    // the corresponding known tangents (slopes) at (x0,y0) and (x1,y1)
    m0 = derivs[lowerIndex];
    m1 = derivs[lowerIndex+1];

    double h, t;
    h = x1 - x0;
    t = (interpTime - x0) / h;
    return (2 * t * t * t - 3 * t * t + 1) * y0 + (t * t * t - 2 * t * t + t) * h * m0 + (-2 * t * t * t + 3 * t * t) * y1 + (t * t * t - t * t) * h * m1;
  }

  /** Evaluate velocities using a Cubic Hermite Spline at a time a, within some interval in x, **/
 double evaluateCubicHermiteFirstDeriv(const double interpTime, const std::vector<double>& deriv,
                                       const std::vector<double>& times, const std::vector<double>& y) {
    if(deriv.size() != times.size()) {
       throw std::invalid_argument("EvaluateCubicHermiteFirstDeriv - The size of the first derivative vector does not match the number of (x,y) data points.");
    }

    // find the interval in which "interpTime" exists
    int lowerIndex = interpolationIndex(times, interpTime);

    double x0, x1, y0, y1, m0, m1;

    // interpTime is contained within the interval (x0,x1)
    x0 = times[lowerIndex];
    x1 = times[lowerIndex+1];

    // the corresponding known y-values for x0 and x1
    y0 = y[lowerIndex];
    y1 = y[lowerIndex+1];

    // the corresponding known tangents (slopes) at (x0,y0) and (x1,y1)
    m0 = deriv[lowerIndex];
    m1 = deriv[lowerIndex+1];

    double h, t;
    h = x1 - x0;
    t = (interpTime - x0) / h;
    if(h != 0.0) {
      return ((6 * t * t - 6 * t) * y0 + (3 * t * t - 4 * t + 1) * h * m0 + (-6 * t * t + 6 * t) * y1 + (3 * t * t - 2 * t) * h * m1) / h;

    }
    else {
      throw std::invalid_argument("Error in evaluating cubic hermite velocities, values at"
                                  "lower and upper indicies are exactly equal.");
    }
  }

  double lagrangeInterpolate(const std::vector<double>& times,
                             const std::vector<double>& values,
                             double time, int order) {
    // Ensure the times and values have the same length
    if (times.size() != values.size()) {
      throw std::invalid_argument("Times and values must have the same length.");
    }

    // Get the correct interpolation window
    int index = interpolationIndex(times, time);
    int windowSize = std::min(index + 1, (int) times.size() - index - 1);
    windowSize = std::min(windowSize, (int) order / 2);
    int startIndex = index - windowSize + 1;
    int endIndex = index + windowSize + 1;

    // Interpolate
    double result = 0;
    for (int i = startIndex; i < endIndex; i++) {
      double weight = 1;
      double numerator = 1;
      for (int j = startIndex; j < endIndex; j++) {
        if (i == j) {
          continue;
        }
        weight *= times[i] - times[j];
        numerator *= time - times[j];
      }
      result += numerator * values[i] / weight;
    }
    return result;
  }

  double lagrangeInterpolateDerivative(const std::vector<double>& times,
                                       const std::vector<double>& values,
                                       double time, int order) {
    // Ensure the times and values have the same length
    if (times.size() != values.size()) {
      throw std::invalid_argument("Times and values must have the same length.");
    }

    // Get the correct interpolation window
    int index = interpolationIndex(times, time);
    int windowSize = std::min(index + 1, (int) times.size() - index - 1);
    windowSize = std::min(windowSize, (int) order / 2);
    int startIndex = index - windowSize + 1;
    int endIndex = index + windowSize + 1;

    // Interpolate
    double result = 0;
    for (int i = startIndex; i < endIndex; i++) {
      double weight = 1;
      double derivativeWeight = 0;
      double numerator = 1;
      for (int j = startIndex; j < endIndex; j++) {
        if (i == j) {
          continue;
        }
        weight *= times[i] - times[j];
        numerator *= time - times[j];
        derivativeWeight += 1.0 / (time - times[j]);
      }
      result += numerator * values[i] * derivativeWeight / weight;
    }
    return result;
  }

 double interpolate(std::vector<double> points, std::vector<double> times, double time, interpolation interp, int d) {
   size_t numPoints = points.size();
   if (numPoints < 2) {
     throw std::invalid_argument("At least two points must be input to interpolate over.");
   }
   if (points.size() != times.size()) {
     throw std::invalid_argument("Must have the same number of points as times.");
   }

   int order;
   switch(interp) {
     case LINEAR:
       order = 2;
       break;
     case SPLINE:
       order = 4;
       break;
     case LAGRANGE:
       order = 8;
       break;
     default:
       throw std::invalid_argument("Invalid interpolation option, must be LINEAR, SPLINE, or LAGRANGE.");
   }

   double result;
   switch(d) {
     case 0:
       result = lagrangeInterpolate(times, points, time, order);
       break;
     case 1:
       result = lagrangeInterpolateDerivative(times, points, time, order);
       break;
     default:
       throw std::invalid_argument("Invalid derivitive option, must be 0 or 1.");
   }

   return result;
 }

}
