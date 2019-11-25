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
      std::vector<double> toStateRotationMatrix(const std::vector<double> &av) const;
      std::vector<double> toEuler(const std::vector<int>& axes) const;
      std::pair<std::vector<double>, double> toAxisAngle() const;

      // Generic rotation operations
      std::vector<double> operator()(const std::vector<double>& vector, const std::vector<double>& av = {0.0, 0.0, 0.0}) const;
      Rotation inverse() const;
      Rotation operator*(const Rotation& rightRotation) const;
      Rotation interpolate(const Rotation& nextRotation, double t, RotationInterpolation interpType) const;

    private:
      // pimpl
      class Impl;
      std::unique_ptr<Impl> m_impl;
  };
}

#endif
