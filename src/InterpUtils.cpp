#include "InterpUtils.h"

#include <algorithm>

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
}
