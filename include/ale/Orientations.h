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
     *
     * @param rotations The rotations defining the Orientations
     * @param times The times for the rotations and angular velocities
     * @param avs The angular velocity at each time
     * @param constRot An additional constant rotation that is applied after
     *                 the time dependent rotations
     * @param const_frames The frame ids that constRot rotates through
     * @param time_dependent_frames The frame ids that rotations rotate through
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
     * Get the vector of time dependent rotations
     */
    std::vector<Rotation> getRotations() const;
    /**
     * Get the vector of angular velocities
     */
    std::vector<ale::Vec3d> getAngularVelocities() const;
    /**
     * Get the vector of times
     */
    std::vector<double> getTimes() const;
    /**
     * Get the frames that the constant rotation rotates through
     */
    std::vector<int> getConstantFrames() const;
    /**
     * Get the frames that the time dependent rotations rotate through
     */
    std::vector<int> getTimeDependentFrames() const;
    /**
     * Get the constant rotation
     */
    Rotation getConstantRotation() const;

    /**
     * Get the time dependent component of the interpolated rotation at a specific time.
     *
     * @param time The time to interpolate at
     * @param interpType The type of interpolation to use
     *
     * @return The time dependent rotation at the input time
     */
    Rotation interpolateTimeDep(
      double time,
      RotationInterpolation interpType=SLERP
    ) const;

    /**
     * Get the interpolated rotation at a specific time.
     *
     * @param time The time to interpolate at
     * @param interpType The type of interpolation to use
     *
     * @return The full rotation at the input time
     */
    Rotation interpolate(
      double time,
      RotationInterpolation interpType=SLERP
    ) const;

    /**
     * Get the interpolated angular velocity at a specific time.
     * Angular velocities are interpolated linearly to match up with the assumptions
     * of SLERP.
     *
     * @param time The time to interpolate the angular velocity at
     *
     * @return The angular velocity at the input time
     */
    ale::Vec3d interpolateAV(double time) const;

    /**
     * Rotate a 3d vector at a specific time
     *
     * @param time The time to rotate the vector at
     * @param vector The input vector to rotate
     * @param interpType The interpolation type to use
     * @param invert If the rotation should be inverted
     *
     * @return The rotated 3d vector
     */
    ale::Vec3d rotateVectorAt(
      double time,
      const ale::Vec3d &vector,
      RotationInterpolation interpType=SLERP,
      bool invert=false
    ) const;

    /**
     * Rotate a state vector at a specific time
     *
     * @param time The time to rotate the vector at
     * @param state The input state to rotate
     * @param interpType The interpolation type to use
     * @param invert If the rotation should be inverted
     *
     * @return The rotated state
     */
    ale::State rotateStateAt(
      double time,
      const ale::State &state,
      RotationInterpolation interpType=SLERP,
      bool invert=false
    ) const;

    /**
     * Add an additional constant rotation after this.
     * This is equivalent to left multiplication by a constant rotation.
     *
     * @param addedConst The additional constant rotation to apply after this
     *
     * @return A refernce to this after the update
     */
    Orientations &addConstantRotation(const Rotation &addedConst);

    /**
     * Multiply this set of orientations by another set of orientations
     *
     * @param rhs The set of orientations to apply before this set
     *
     * @return A refernce to this after the update
     */
    Orientations &operator*=(const Orientations &rhs);

    /**
     * Add an additional constant rotation before this.
     * This is equivalent to right multiplication by a constant rotation.
     *
     * @param rhs The additional constant rotation to apply before this
     *
     * @return A refernce to this after the update
     */
    Orientations &operator*=(const Rotation &rhs);

    /**
     * Invert the set orientations.
     *
     * Note that inverting a set of orientations twice does not result in
     * the original orientations. the constant rotation is applied after the
     * time dependent rotation. This means in the inverse, the constant
     * rotation is applied first and then the time dependent rotation.
     * So, the inverted set of orientations is entirely time dependent.
     * Then, the original constant rotations cannot be recovered when inverting
     * again. The set of orientations will still operate the same way, but its
     * internal state will not be the same.
     *
     * Similarly, the angular velocities will not be the same as we do not assume
     * the angular acceleration to be continuous.
     *
     * @return A new set of orientations that are inverted.
     */
     Orientations inverse() const;

  private:
    std::vector<Rotation> m_rotations; //!< The set of time dependent rotations.
    std::vector<ale::Vec3d> m_avs; //!< The set of angular velocities. Empty if there are no angular velocities.
    std::vector<double> m_times; //!< The set of times
    std::vector<int> m_timeDepFrames; //!< The frame IDs that the time dependent rotations rotate through.
    std::vector<int> m_constFrames; //!< The frame IDs that the constant rotation rotates through.
    Rotation m_constRotation; //!< The constant rotation applied after the time dependent rotations.
  };

  /**
   * Apply a constant rotation before a set of Orientations
   *
   * @param lhs The set of Orientations
   * @param rhs The constant rotation
   *
   * @return A new set of orientations combining the constant rotation and old orientations
   */
  Orientations operator*(Orientations lhs, const Rotation &rhs);

  /**
   * Apply a constant rotation after a set of Orientations
   *
   * @param lhs The constant rotation
   * @param rhs The set of Orientations
   *
   * @return A new set of orientations combining the constant rotation and old orientations
   */
  Orientations operator*(const Rotation &lhs, Orientations rhs);

  /**
   * Apply two Orientations in a row
   *
   * @param lhs The second orientation to apply
   * @param rhs The first orientation to apply
   *
   * @return A new set of orientations combining both orientations
   */
  Orientations operator*(Orientations lhs, const Orientations &rhs);
}

#endif
