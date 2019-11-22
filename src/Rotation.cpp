#include "Rotation.h"

#include <algorithm>
#include <exception>

#include <Eigen/Geometry>

namespace ale {

///////////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////////

  Eigen::Vector3d axis(int axisIndex) {
    switch (axisIndex) {
      case 0:
        return Eigen::Vector3d::UnitX();
        break;
      case 1:
        return Eigen::Vector3d::UnitY();
        break;
      case 2:
        return Eigen::Vector3d::UnitZ();
        break;
      default:
        throw std::invalid_argument("Axis index must be 0, 1, or 2.");
    }
  }


  // This is actually the transpose of the skew AV matrix because we define AV
  // as the AV from the destination to the source. This matches how NAIF
  // defines AV.
  Eigen::Quaterniond::Matrix3 avSkewMatrix(
        const std::vector<double>& av
  ) {
    Eigen::Quaterniond::Matrix3 avMat;
    avMat <<  0.0,    av[2], -av[1],
             -av[2],  0.0,    av[0],
              av[1], -av[0],  0.0;
    return avMat;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Rotation Impl class
  ///////////////////////////////////////////////////////////////////////////////

  class Rotation::Impl {
    public:
      Impl() : quat(Eigen::Quaterniond::Identity()) { }


      Impl(double w, double x, double y, double z) : quat(w, x, y, z) { }


      Impl(const std::vector<double>& matrix) {
        if (matrix.size() != 9) {
          throw std::invalid_argument("Rotation matrix must be 3 by 3.");
        }
        quat = Eigen::Quaterniond(Eigen::Quaterniond::Matrix3(matrix.data()));
      }


      Impl(const std::vector<double>& angles, const std::vector<int>& axes) {
        if (angles.empty() || axes.empty()) {
          throw std::invalid_argument("Angles and axes must be non-empty.");
        }
        if (angles.size() != axes.size()) {
          throw std::invalid_argument("Number of angles and axes must be equal.");
        }
        quat = Eigen::Quaterniond::Identity();

        for (size_t i = 0; i < angles.size(); i++) {
          quat *= Eigen::Quaterniond(Eigen::AngleAxisd(angles[i], axis(axes[i])));
        }
      }


      Impl(const std::vector<double>& axis, double theta) {
        if (axis.size() != 3) {
          throw std::invalid_argument("Rotation axis must have 3 elements.");
        }
        Eigen::Vector3d eigenAxis((double *) axis.data());
        quat = Eigen::Quaterniond(Eigen::AngleAxisd(theta, eigenAxis.normalized()));
      }


      Eigen::Quaterniond quat;
  };

  ///////////////////////////////////////////////////////////////////////////////
  // Rotation Class
  ///////////////////////////////////////////////////////////////////////////////

  Rotation::Rotation() :
        m_impl(new Impl()) { }


  Rotation::Rotation(double w, double x, double y, double z) :
        m_impl(new Impl(w, x, y, z)) { }


  Rotation::Rotation(const std::vector<double>& matrix) :
        m_impl(new Impl(matrix)) { }


  Rotation::Rotation(const std::vector<double>& angles, const std::vector<int>& axes) :
        m_impl(new Impl(angles, axes)) { }


  Rotation::Rotation(const std::vector<double>& axis, double theta) :
        m_impl(new Impl(axis, theta)) { }


  Rotation::~Rotation() = default;


  Rotation::Rotation(Rotation && other) noexcept = default;


  Rotation& Rotation::operator=(Rotation && other) noexcept = default;


  // unique_ptr doesn't have a copy constructor so we have to define one
  Rotation::Rotation(const Rotation& other) : m_impl(new Impl(*other.m_impl)) { }


  // unique_ptr doesn't have an assignment operator so we have to define one
  Rotation& Rotation::operator=(const Rotation& other) {
    if (this != &other) {
      m_impl.reset(new Impl(*other.m_impl));
    }
    return *this;
  }


  std::vector<double> Rotation::toQuaternion() const {
    Eigen::Quaterniond normalized = m_impl->quat.normalized();
    return {normalized.w(), normalized.x(), normalized.y(), normalized.z()};
  }


  std::vector<double> Rotation::toRotationMatrix() const {
    Eigen::Quaterniond::RotationMatrixType mat = m_impl->quat.toRotationMatrix();
    return std::vector<double>(mat.data(), mat.data() + mat.size());
  }


  std::vector<double> Rotation::toEuler(const std::vector<int>& axes) const {
    if (axes.size() != 3) {
      throw std::invalid_argument("Must have 3 axes to convert to Euler angles.");
    }
    Eigen::Vector3d angles = m_impl->quat.toRotationMatrix().eulerAngles(
          axes[0],
          axes[1],
          axes[2]);
    return std::vector<double>(angles.data(), angles.data() + angles.size());
  }


  std::pair<std::vector<double>, double> Rotation::toAxisAngle() const {
    Eigen::AngleAxisd eigenAxisAngle(m_impl->quat);
    std::pair<std::vector<double>, double> axisAngle;
    axisAngle.first = std::vector<double>(
          eigenAxisAngle.axis().data(),
          eigenAxisAngle.axis().data() + eigenAxisAngle.axis().size()
    );
    axisAngle.second = eigenAxisAngle.angle();
    return axisAngle;
  }


  std::vector<double> Rotation::operator()(const std::vector<double>& vector) const {
    Eigen::Map<Eigen::Vector3d> eigenVector((double *)vector.data());
    Eigen::Vector3d rotatedVector = m_impl->quat._transformVector(eigenVector);
    return std::vector<double>(rotatedVector.data(), rotatedVector.data() + rotatedVector.size());
  }


  std::vector<double> Rotation::operator()(
        const std::vector<double>& vector,
        const std::vector<double>& av
  ) const {
    Eigen::Map<Eigen::Vector3d> positionVector((double *)vector.data());
    Eigen::Map<Eigen::Vector3d> velocityVector((double *)vector.data() + 3);
    Eigen::Quaterniond::Matrix3 rotMat = m_impl->quat.toRotationMatrix();
    Eigen::Quaterniond::Matrix3 avMat = avSkewMatrix(av);
    Eigen::Quaterniond::Matrix3 rotationDerivative = rotMat * avMat;
    Eigen::Vector3d rotatedPosition = rotMat * positionVector;
    Eigen::Vector3d rotatedVelocity = rotMat * velocityVector + rotationDerivative * positionVector;
    return {rotatedPosition(0), rotatedPosition(1), rotatedPosition(2),
            rotatedVelocity(0), rotatedVelocity(1), rotatedVelocity(2)};
  }


  Rotation Rotation::inverse() const {
    Eigen::Quaterniond inverseQuat = m_impl->quat.inverse();
    return Rotation(inverseQuat.w(), inverseQuat.x(), inverseQuat.y(), inverseQuat.z());
  }


  Rotation Rotation::operator*(const Rotation& rightRotation) const {
    Eigen::Quaterniond combinedQuat = m_impl->quat * rightRotation.m_impl->quat;
    return Rotation(combinedQuat.w(), combinedQuat.x(), combinedQuat.y(), combinedQuat.z());
  }


  Rotation Rotation::interpolate(
        const Rotation& nextRotation,
        double t,
        RotationInterpolation interpType
  ) const {
    Eigen::Quaterniond interpQuat;
    switch (interpType) {
      case slerp:
        interpQuat = m_impl->quat.slerp(t, nextRotation.m_impl->quat);
        break;
      case nlerp:
        interpQuat = Eigen::Quaterniond(
              linearInterpolate(m_impl->quat.w(), nextRotation.m_impl->quat.w(), t),
              linearInterpolate(m_impl->quat.x(), nextRotation.m_impl->quat.x(), t),
              linearInterpolate(m_impl->quat.y(), nextRotation.m_impl->quat.y(), t),
              linearInterpolate(m_impl->quat.z(), nextRotation.m_impl->quat.z(), t)
        );
        interpQuat.normalize();
        break;
      default:
        throw std::invalid_argument("Unsupported rotation interpolation type.");
        break;
    }
    return Rotation(interpQuat.w(), interpQuat.x(), interpQuat.y(), interpQuat.z());
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Rotation Interpolation Functions
  ///////////////////////////////////////////////////////////////////////////////

  std::vector<double> rotateAt(
        double interpTime, // time to interpolate rotation at
        const std::vector<double>& vector, // state vector to rotate, could be 3 or 6 elements
        const std::vector<double>& times, // rotation times
        const std::vector<Rotation>& rotations, // rotations
        const std::vector<std::vector<double>>& avs, // rotation angular velocities
        RotationInterpolation interpType, // rotation interpolation type
        bool invert
  ) {
    int prevTimeIndex = interpolationIndex(interpTime, times);
    double t = (interpTime - times[prevTimeIndex]) / (times[prevTimeIndex + 1] - times[prevTimeIndex]);
    Rotation interpRotation = rotations[prevTimeIndex].interpolate(rotations[prevTimeIndex+1], t, interpType);
    if (invert) {
      interpRotation = interpRotation.inverse();
    }
    if (vector.size() == 3) {
      return interpRotation(vector);
    }
    else if (vector.size() == 6) {
      std::vector<double> interpAv;
      if (avs.empty()) {
        interpAv.resize(3, 0.0);
      }
      else {
        interpAv = linearInterpolate(avs[prevTimeIndex], avs[prevTimeIndex+1], t);
      }
      return interpRotation(vector, interpAv);
    }
    else {
      throw std::invalid_argument("Vector to rotate must be 3 or 6 elements.");
    }
  }


  Rotation interpolateRotation(
    double interpTime, // time to interpolate rotation at
    const std::vector<double>& times, // rotation times
    const std::vector<Rotation>& rotations, // rotations
    RotationInterpolation interpType // rotation interpolation type
  ) {
    int prevTimeIndex = interpolationIndex(interpTime, times);
    double t = (interpTime - times[prevTimeIndex]) / (times[prevTimeIndex + 1] - times[prevTimeIndex]);
    return rotations[prevTimeIndex].interpolate(rotations[prevTimeIndex+1], t, interpType);
  }


  std::vector<double> stateRotation(
    const Rotation &rotation,
    const std::vector<double> &av
  ) {
    std::vector<double> vecRotMat = rotation.toRotationMatrix();
    Eigen::Map<Eigen::Quaterniond::Matrix3> rotMat((double*)vecRotMat.data());
    Eigen::Quaterniond::Matrix3 avMat = avSkewMatrix(av);
    Eigen::Quaterniond::Matrix3 dtMat = rotMat * avMat;
    return {rotMat(0,0), rotMat(0,1), rotMat(0,2), 0.0,         0.0,         0.0,
            rotMat(1,0), rotMat(1,1), rotMat(1,2), 0.0,         0.0,         0.0,
            rotMat(2,0), rotMat(2,1), rotMat(2,2), 0.0,         0.0,         0.0,
            dtMat(0,0),  dtMat(0,1),  dtMat(0,2),  rotMat(0,0), rotMat(0,1), rotMat(0,2),
            dtMat(1,0),  dtMat(1,1),  dtMat(1,2),  rotMat(1,0), rotMat(1,1), rotMat(1,2),
            dtMat(2,0),  dtMat(2,1),  dtMat(2,2),  rotMat(2,0), rotMat(2,1), rotMat(2,2)};
  }


  int interpolationIndex(double interpTime, std::vector<double> times) {
    // Find the next time
    auto nextTimeIt = std::upper_bound(times.begin(), times.end(), interpTime);
    if (nextTimeIt == times.begin()) {
      ++nextTimeIt;
    }
    else if (nextTimeIt == times.end()) {
      --nextTimeIt;
    }
    return std::distance(times.begin(), nextTimeIt);
  }


  double linearInterpolate(double x, double y, double t) {
    return x + t * (y - x);
  }


  std::vector<double> linearInterpolate(
        const std::vector<double>& x,
        const std::vector<double>& y,
        double t
  ) {
    if (x.empty() || y.empty()) {
      throw std::invalid_argument("Vectors to interpolate must be non-empty.");
    }
    if (x.size() != y.size()) {
      throw std::invalid_argument("Vectors to interpolate must have the same size.");
    }
    std::vector<double> interpVector(x.size());
    for (size_t i = 0; i < interpVector.size(); i++) {
      interpVector[i] = linearInterpolate(x[i], y[i], t);
    }
    return interpVector;
  }

}
