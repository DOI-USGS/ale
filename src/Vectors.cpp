#include "ale/Vectors.h"

namespace ale {
  Vec3d operator*(double scalar, Vec3d vec) {
    return vec *= scalar;
  }

  Vec3d operator*(Vec3d vec, double scalar) {
    return vec *= scalar;
  }

  Vec3d operator+(Vec3d leftVec, const Vec3d &rightVec) {
    return leftVec += rightVec;
  }

  Vec3d operator-(Vec3d leftVec, const Vec3d &rightVec) {
    return leftVec -= rightVec;
  }
}
