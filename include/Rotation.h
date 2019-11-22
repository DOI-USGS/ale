#ifndef ALE_ROTATION_H
#define ALE_ROTATION_H

#include <memory>
#include <vector>

namespace ale {

  enum RotationInterpolation {
    slerp,
    nlerp
  };

  class Rotation {
    public:
      Rotation();
      Rotation(double w, double x, double y, double z);
      Rotation(const std::vector<double>& matrix);
      Rotation(const std::vector<double>& angles, const std::vector<int>& axes);
      Rotation(const std::vector<double>& axis, double theta);
      ~Rotation();

      // Special member functions
      Rotation(Rotation && other) noexcept;
      Rotation& operator=(Rotation && other) noexcept;

      Rotation(const Rotation& other);
      Rotation& operator=(const Rotation& other);

      // Type specific accessors
      std::vector<double> toQuaternion() const;
      std::vector<double> toRotationMatrix() const;
      std::vector<double> toEuler(const std::vector<int>& axes) const;
      std::pair<std::vector<double>, double> toAxisAngle() const;

      // Generic rotation operations
      std::vector<double> operator()(const std::vector<double>& vector) const;
      std::vector<double> operator()(const std::vector<double>& vector, const std::vector<double>& av) const;
      Rotation inverse() const;
      Rotation operator*(const Rotation& rightRotation) const;
      Rotation interpolate(const Rotation& nextRotation, double t, RotationInterpolation interpType) const;

    private:
      // pimpl
      class Impl;
      std::unique_ptr<Impl> m_impl;
  };

  std::vector<double> rotateAt(
    double interpTime, // time to interpolate rotation at
    const std::vector<double>& vector, // state vector to rotate, could be 3 or 6 elements
    const std::vector<double>& times, // rotation times
    const std::vector<Rotation>& rotations, // rotations
    const std::vector<std::vector<double>>& avs = std::vector<std::vector<double>>(), // rotation angular velocities
    RotationInterpolation interpType = slerp, // rotation interpolation type
    bool invert = false // if the rotation direction should be inverted
  );

  Rotation interpolateRotation(
    double interpTime, // time to interpolate rotation at
    const std::vector<double>& times, // rotation times
    const std::vector<Rotation>& rotations, // rotations
    RotationInterpolation interpType = slerp // rotation interpolation type
  );

  std::vector<double> stateRotation(
    const Rotation &rotation,
    const std::vector<double> &av
  );

  int interpolationIndex(double interpTime, std::vector<double> times);

  double linearInterpolate(double x, double y, double t);

  std::vector<double> linearInterpolate(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double t
  );
}

#endif
