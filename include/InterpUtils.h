#ifndef ALE_INCLUDE_INTERP_UTILS_H
#define ALE_INCLUDE_INTERP_UTILS_H

#include <vector>

#include "Util.h"

namespace ale {
  /**
   * Linearly interpolate between two values.
   * @param x The first value.
   * @param y The second value.
   * @param t The distance to interpolate. 0 is x and 1 is y.
   */
  double linearInterpolate(double x, double y, double t);

  /**
   * Linearly interpolate between two vectors.
   * @param x The first vectors.
   * @param y The second vectors.
   * @param t The distance to interpolate. 0 is x and 1 is y.
   */
  std::vector<double> linearInterpolate(const std::vector<double> &x, const std::vector<double> &y, double t);

  ale::Vec3d linearInterpolate(const ale::Vec3d &x, const ale::Vec3d &y, double t);

  /**
   * Compute the index of the first time to use when interpolating at a given time.
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

}

#endif
