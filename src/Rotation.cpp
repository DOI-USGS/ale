#include "ale/Rotation.h"

#include <exception>

#include <Eigen/Geometry>

#include "ale/InterpUtils.h"

namespace ale {

///////////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////////


  // Helper function to convert an axis number into a unit Eigen vector down that axis.
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


  /**
   * Create the skew symmetric matrix used when computing the derivative of a
   * rotation matrix.
   *
   * This is actually the transpose of the skew AV matrix because we define AV
   * as the AV from the destination to the source. This matches how NAIF
   * defines AV.
   */
  Eigen::Quaterniond::Matrix3 avSkewMatrix(const Vec3d& av) {
    Eigen::Quaterniond::Matrix3 avMat;
    avMat <<  0.0,    av.z, -av.y,
             -av.z,  0.0,    av.x,
              av.y, -av.x,  0.0;
    return avMat;
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Rotation Impl class
  ///////////////////////////////////////////////////////////////////////////////

  // Internal representation of the rotation as an Eigen Double Quaternion
  class Rotation::Impl {
    public:
      Impl() : quat(Eigen::Quaterniond::Identity()) { }


      Impl(double w, double x, double y, double z) : quat(w, x, y, z) { }


      Impl(const std::vector<double>& matrix) {
        if (matrix.size() != 9) {
          throw std::invalid_argument("Rotation matrix must be 3 by 3.");
        }
        // The data is in row major order, so take the transpose to get column major order
        quat = Eigen::Quaterniond(Eigen::Quaterniond::Matrix3(matrix.data()).transpose());
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
    // The matrix is stored in column major, but we want to output in row semiMajor
    // so take the transpose
    Eigen::Quaterniond::RotationMatrixType mat = m_impl->quat.toRotationMatrix().transpose();
    return std::vector<double>(mat.data(), mat.data() + mat.size());
  }


  std::vector<double> Rotation::toStateRotationMatrix(const Vec3d &av) const {
    Eigen::Quaterniond::Matrix3 rotMat = m_impl->quat.toRotationMatrix();
    Eigen::Quaterniond::Matrix3 avMat = avSkewMatrix(av);
    Eigen::Quaterniond::Matrix3 dtMat = rotMat * avMat;
    return {rotMat(0,0), rotMat(0,1), rotMat(0,2), 0.0,         0.0,         0.0,
            rotMat(1,0), rotMat(1,1), rotMat(1,2), 0.0,         0.0,         0.0,
            rotMat(2,0), rotMat(2,1), rotMat(2,2), 0.0,         0.0,         0.0,
            dtMat(0,0),  dtMat(0,1),  dtMat(0,2),  rotMat(0,0), rotMat(0,1), rotMat(0,2),
            dtMat(1,0),  dtMat(1,1),  dtMat(1,2),  rotMat(1,0), rotMat(1,1), rotMat(1,2),
            dtMat(2,0),  dtMat(2,1),  dtMat(2,2),  rotMat(2,0), rotMat(2,1), rotMat(2,2)};
  }


  std::vector<double> Rotation::toEuler(const std::vector<int>& axes) const {
    if (axes.size() != 3) {
      throw std::invalid_argument("Must have 3 axes to convert to Euler angles.");
    }
    if (axes[0] < 0 || axes[0] > 2 ||
        axes[1] < 0 || axes[1] > 2 ||
        axes[2] < 0 || axes[2] > 2) {
      throw std::invalid_argument("Invalid axis number.");
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


  Vec3d Rotation::operator()(const Vec3d &vector) const {
    Eigen::Vector3d eigenVector(vector.x, vector.y, vector.z);
    Eigen::Vector3d rotatedVector = m_impl->quat._transformVector(eigenVector);
    return Vec3d(rotatedVector[0], rotatedVector[1], rotatedVector[2]);
  }

  State Rotation::operator()(
        const State& state,
        const Vec3d& av
  ) const {
    Vec3d position = state.position;
    Vec3d velocity = state.velocity;

    Eigen::Vector3d positionVector(position.x, position.y, position.z);
    Eigen::Vector3d velocityVector(velocity.x, velocity.y, velocity.z);
    Eigen::Quaterniond::Matrix3 rotMat = m_impl->quat.toRotationMatrix();
    Eigen::Quaterniond::Matrix3 avMat = avSkewMatrix(av);
    Eigen::Quaterniond::Matrix3 rotationDerivative = rotMat * avMat;
    Eigen::Vector3d rotatedPosition = rotMat * positionVector;
    Eigen::Vector3d rotatedVelocity = rotMat * velocityVector + rotationDerivative * positionVector;

    return State({rotatedPosition(0), rotatedPosition(1), rotatedPosition(2),
                  rotatedVelocity(0), rotatedVelocity(1), rotatedVelocity(2)});
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
      case SLERP:
        interpQuat = m_impl->quat.slerp(t, nextRotation.m_impl->quat);
        break;
      case NLERP:
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

}
