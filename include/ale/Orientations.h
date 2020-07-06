#ifndef ALE_ORIENTATIONS_H
#define ALE_ORIENTATIONS_H

#include <vector>

#include "ale/Rotation.h"

namespace ale {
  class Orientations {
  public:
    /**
     * Construct a default empty orientation object
     */
    Orientations() {};

    /**
     * Construct an orientation object give a set of rotations
     * and optionally angular velocities at specific times.
     */
    Orientations(
      const std::vector<Rotation> &rotations,
      const std::vector<double> &times,
      const std::vector<Vec3d> &avs = std::vector<ale::Vec3d>(),
      const Rotation &constRot = Rotation(1, 0, 0, 0),
      const std::vector<int> const_frames = std::vector<int>(),
      const std::vector<int> time_dependent_frames = std::vector<int>()
    );

    /**
     * Orientations destructor
     */
    ~Orientations() {};

    /**
     * Const accessor methods
     */
    std::vector<Rotation> getRotations() const;
    std::vector<ale::Vec3d> getAngularVelocities() const;
    std::vector<double> getTimes() const;
    std::vector<int> getConstantFrames() const;
    std::vector<int> getTimeDependentFrames() const;
    Rotation getConstantRotation() const;

    /**
     * Get the time dependent component of the interpolated rotation at a specific time.
     */
    Rotation interpolateTimeDep(
      double time,
      RotationInterpolation interpType=SLERP
    ) const;

    /**
     * Get the interpolated rotation at a specific time.
     */
    Rotation interpolate(
      double time,
      RotationInterpolation interpType=SLERP
    ) const;

    /**
     * Get the interpolated angular velocity at a specific time
     */
    ale::Vec3d interpolateAV(double time) const;

    ale::Vec3d rotateVectorAt(
      double time,
      const ale::Vec3d &vector,
      RotationInterpolation interpType=SLERP,
      bool invert=false
    ) const;

    /**
     * Rotate a position or state vector at a specific time
     */
    ale::State rotateStateAt(
      double time,
      const ale::State &state,
      RotationInterpolation interpType=SLERP,
      bool invert=false
    ) const;

    /**
     * Add an additional constant multiplication.
     * This is equivalent to left multiplication by a constant rotation
     */
    Orientations &addConstantRotation(const Rotation &addedConst);

    /**
     * Multiply this orientation by another orientation
     */
    Orientations &operator*=(const Orientations &rhs);

    /**
     * Multiply this orientation by a constant rotation
     */
    Orientations &operator*=(const Rotation &rhs);

    /**
     * Invert the orientations.
     * Note that inverting a set of orientations twice does not result in
     * the original orientations. the constant rotation is applied after the
     * time dependent rotation. This means in the inverse, the constant
     * rotation is applied first and then the time dependent rotation.
     * So, the inverted set of orientations is entirely time dependent.
     * Then, the original constant rotations cannot be recovered when inverting
     * again. The set of orientations will still operate the same way, but its
     * internal state will not be the same.
     */
     Orientations inverse() const;

  private:
    std::vector<Rotation> m_rotations;
    std::vector<ale::Vec3d> m_avs;
    std::vector<double> m_times;
    std::vector<int> m_timeDepFrames;
    std::vector<int> m_constFrames;
    Rotation m_constRotation;
  };

  Orientations operator*(Orientations lhs, const Rotation &rhs);
  Orientations operator*(const Rotation &lhs, Orientations rhs);
  Orientations operator*(Orientations lhs, const Orientations &rhs);
}

#endif
