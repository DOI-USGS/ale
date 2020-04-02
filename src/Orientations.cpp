#include "Orientations.h"

#include "InterpUtils.h"

namespace ale {

  Orientations::Orientations(
    const std::vector<Rotation> &rotations,
    const std::vector<double> &times,
    const std::vector<ale::Vec3d> &avs
  ) :
    m_rotations(rotations), m_avs(avs), m_times(times) {
    if (m_rotations.size() < 2 || m_times.size() < 2) {
      throw std::invalid_argument("There must be at least two rotations and times.");
    }
    if (m_rotations.size() != m_times.size()) {
      throw std::invalid_argument("The number of rotations and times must be the same.");
    }
    if ( !m_avs.empty() && (m_avs.size() != m_times.size()) ) {
      throw std::invalid_argument("The number of angular velocities and times must be the same.");
    }
  }


  std::vector<Rotation> Orientations::rotations() const {
    return m_rotations;
  }


  std::vector<ale::Vec3d> Orientations::angularVelocities() const {
    return m_avs;
  }


  std::vector<double> Orientations::times() const {
    return m_times;
  }


  Rotation Orientations::interpolate(
    double time,
    RotationInterpolation interpType
  ) const {
    int interpIndex = interpolationIndex(m_times, time);
    double t = (time - m_times[interpIndex]) / (m_times[interpIndex + 1] - m_times[interpIndex]);
    return m_rotations[interpIndex].interpolate(m_rotations[interpIndex + 1], t, interpType);
  }


  ale::Vec3d Orientations::interpolateAV(double time) const {
    int interpIndex = interpolationIndex(m_times, time);
    double t = (time - m_times[interpIndex]) / (m_times[interpIndex + 1] - m_times[interpIndex]);
    ale::Vec3d interpAv = ale::Vec3d(linearInterpolate(m_avs[interpIndex], m_avs[interpIndex + 1], t));
    return interpAv;
  }

  ale::Vec3d Orientations::rotateVectorAt(
    double time,
    const ale::Vec3d &vector,
    RotationInterpolation interpType,
    bool invert
  ) const {
    Rotation interpRot = interpolate(time, interpType);
    return interpRot(vector);
  }


  ale::State Orientations::rotateStateAt(
    double time,
    const ale::State &state,
    RotationInterpolation interpType,
    bool invert
  ) const {
    Rotation interpRot = interpolate(time, interpType);
    ale::Vec3d av(0.0, 0.0, 0.0);
    if (!m_avs.empty()) {
      av = interpolateAV(time);
    }
    if (invert) {
      ale::Vec3d negAv = interpRot(av);
      av = {-negAv.x, -negAv.y, -negAv.z};
      interpRot = interpRot.inverse();
    }
    return interpRot(state, av);
  }


  Orientations &Orientations::operator*=(const Orientations &rhs) {
    std::vector<double> mergedTimes = orderedVecMerge(m_times, rhs.m_times);
    std::vector<Rotation> mergedRotations;
    std::vector<ale::Vec3d> mergedAvs;
    for (double time: mergedTimes) {
      Rotation rhsRot = rhs.interpolate(time);
      mergedRotations.push_back(interpolate(time)*rhsRot);
      ale::Vec3d combinedAv = rhsRot.inverse()(interpolateAV(time));
      ale::Vec3d rhsAv = rhs.interpolateAV(time);
      combinedAv.x += rhsAv.x;
      combinedAv.y += rhsAv.y;
      combinedAv.z += rhsAv.z;
      mergedAvs.push_back(combinedAv);
    }

    m_times = mergedTimes;
    m_rotations = mergedRotations;
    m_avs = mergedAvs;

    return *this;
  }


  Orientations &Orientations::operator*=(const Rotation &rhs) {
    std::vector<Rotation> updatedRotations;
    for (size_t i = 0; i < m_rotations.size(); i++) {
      updatedRotations.push_back(m_rotations[i]*rhs);
    }

    Rotation inverse = rhs.inverse();
    std::vector<Vec3d> updatedAvs;
    for (size_t i = 0; i < m_avs.size(); i++) {
      updatedAvs.push_back(inverse(m_avs[i]));
    }

    m_rotations = updatedRotations;
    m_avs = updatedAvs;

    return *this;
  }
}
