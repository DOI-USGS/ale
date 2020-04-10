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
     * Multiply this orientation by another orientation
     */
    Orientations &operator*=(const Orientations &rhs);
    /**
     * Multiply this orientation by a constant rotation
     */
    Orientations &operator*=(const Rotation &rhs);

  private:
    std::vector<Rotation> m_rotations;
    std::vector<ale::Vec3d> m_avs;
    std::vector<double> m_times;
    std::vector<int> m_timeDepFrames;
    std::vector<int> m_constFrames;
    Rotation m_constRotation;
  };
}

#endif
