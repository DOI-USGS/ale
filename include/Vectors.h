#ifndef ALE_VECTORS_H
#define ALE_VECTORS_H

#include <stdexcept>

namespace ale {
  /** A 3D cartesian vector */
  struct Vec3d {
    double x;
    double y;
    double z;

    // Accepts an {x,y,z} vector
    Vec3d(const std::vector<double>& vec) {
      if (vec.size() != 3) {
        throw std::invalid_argument("Input vector must have 3 entries.");
      }
      x = vec[0];
      y = vec[1];
      z = vec[2];
    };

    Vec3d(double x, double y, double z) : x(x), y(y), z(z) {};
    Vec3d() : x(0.0), y(0.0), z(0.0) {};
  };
}

#endif
