#ifndef ALE_ROTATION_H
#define ALE_ROTATION_H

#include <memory>
#include <vector>

#include "ale/States.h"
#include "ale/Vectors.h"

namespace ale {

  /**
   * A generic 3D rotation.
   */
  class Rotation {
    public:
      /**
       * Construct a default identity rotation.
       */
      Rotation();
      
      /**
       * Construct a rotation from a quaternion
       *
       * @param w The scalar component of the quaternion.
       * @param x The x value of the vector component of the quaternion.
       * @param y The y value of the vector component of the quaternion.
       * @param z The z value of the vector component of the quaternion.
       */
      Rotation(double w, double x, double y, double z);

      /**
       * Construct a rotation from a rotation matrix.
       *
       * @param matrix The rotation matrix in row-major order.
       */
      Rotation(const std::vector<double>& matrix);

      /**
       * Construct a rotation from a set of Euler angle rotations.
       *
       * @param angles A vector of rotations about the axes.
       * @param axes The vector of axes to rotate about, in order.
       *             0 is X, 1 is Y, and 2 is Z.
       */
      Rotation(const std::vector<double>& angles, const std::vector<int>& axes);

      /**
       * Construct a rotation from a rotation about an axis.
       *
       * @param axis The axis of rotation.
       * @param theta The rotation about the axis in radians.
       */
      Rotation(const std::vector<double>& axis, double theta);
      ~Rotation();

      // Special member functions
      Rotation(Rotation && other) noexcept;
      Rotation& operator=(Rotation && other) noexcept;

      Rotation(const Rotation& other);
      Rotation& operator=(const Rotation& other);

      // Type specific accessors
      /**
       * The rotation as a quaternion.
       *
       * @return The rotation as a scalar-first quaternion (w, x, y, z).
       */
      std::vector<double> toQuaternion() const;

      /**
       * The rotation as a rotation matrix.
       *
       * @return The rotation as a rotation matrix in row-major order.
       */
      std::vector<double> toRotationMatrix() const;

      /**
       * Create a state rotation matrix from the rotation and an angular velocity.
       *
       * @param av The angular velocity vector.
       *
       * @return The state rotation matrix in row-major order.
       */
      std::vector<double> toStateRotationMatrix(const Vec3d &av) const;

      /**
       * The rotation as Euler angles.
       *
       * @param axes The axis order. 0 is X, 1 is Y, and 2 is Z.
       *
       * @return The rotations about the axes in radians.
       */
      std::vector<double> toEuler(const std::vector<int>& axes) const;

      /**
       * The rotation as a rotation about an axis.
       *
       * @return the axis of rotation and rotation in radians.
       */
      std::pair<std::vector<double>, double> toAxisAngle() const;

      // Generic rotation operations
      /**
       * Rotate a vector
       *
       * @param vector The vector to rotate. Can be a 3 element position or 6 element state.
       * @param av The angular velocity to use when rotating state vectors. Defaults to 0.
       *
       * @return The rotated vector.
       */
      Vec3d operator()(const Vec3d &av) const;

      State operator()(const State &state, const Vec3d& av = Vec3d(0.0, 0.0, 0.0)) const;

      /**
       * Get the inverse rotation.
       */
      Rotation inverse() const;

      /**
       * Chain this rotation with another rotation.
       *
       * Rotations are sequenced right to left.
       */
      Rotation operator*(const Rotation& rightRotation) const;

      /**
       * Interpolate between this rotation and another rotation.
       *
       * @param t The distance to interpolate. 0 is this and 1 is the next rotation.
       * @param interpType The type of rotation interpolation to use.
       *
       * @param The interpolated rotation.
       */
      Rotation interpolate(const Rotation& nextRotation, double t, RotationInterpolation interpType) const;

    private:
      // Implementation class
      class Impl;
      // Pointer to internal rotation implementation.
      std::unique_ptr<Impl> m_impl;
  };
}

#endif
