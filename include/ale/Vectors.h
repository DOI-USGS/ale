#ifndef ALE_VECTORS_H
#define ALE_VECTORS_H

#include <stdexcept>
#include <vector>
#include <math.h>

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
    Vec3d &operator*=(double scalar) {
      x *= scalar;
      y *= scalar;
      z *= scalar;
      return *this;
    };

    Vec3d &operator+=(Vec3d addend) {
      x += addend.x;
      y += addend.y;
      z += addend.z;
      return *this;
    };

    Vec3d &operator-=(Vec3d addend) {
      x -= addend.x;
      y -= addend.y;
      z -= addend.z;
      return *this;
    };

    double norm() const {
      return sqrt(x*x + y*y + z*z);
    }
  };

  Vec3d operator*(double scalar, Vec3d vec);

  Vec3d operator*(Vec3d vec, double scalar);

  Vec3d operator+(Vec3d leftVec, const Vec3d &rightVec);

  Vec3d operator-(Vec3d leftVec, const Vec3d &rightVec);
}

#endif
