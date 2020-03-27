#include "Orientations.h"

#include "InterpUtils.h"

namespace ale {

  Orientations::Orientations(
    const std::vector<Rotation> &rotations,
    const std::vector<double> &times,
    const std::vector<std::vector<double>> &avs
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


  std::vector<std::vector<double>> Orientations::angularVelocities() const {
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


  std::vector<double> Orientations::interpolateAV(double time) const {
    int interpIndex = interpolationIndex(m_times, time);
    double t = (time - m_times[interpIndex]) / (m_times[interpIndex + 1] - m_times[interpIndex]);
    std::vector<double> interpAv = linearInterpolate(m_avs[interpIndex], m_avs[interpIndex + 1], t);
    return interpAv;
  }


  std::vector<double> Orientations::rotateAt(
    double time,
    const std::vector<double> &vector,
    RotationInterpolation interpType,
    bool invert
  ) const {
    Rotation interpRot = interpolate(time, interpType);
    std::vector<double> av = {0.0, 0.0, 0.0};
    if (!m_avs.empty()) {
      av = interpolateAV(time);
    }
    if (invert) {
      std::vector<double> negAv = interpRot(av);
      av = {-negAv[0], -negAv[1], -negAv[2]};
      interpRot = interpRot.inverse();
    }
    return interpRot(vector, av);
  }


  Orientations &Orientations::operator*=(const Orientations &rhs) {
    std::vector<double> mergedTimes = orderedVecMerge(m_times, rhs.m_times);
    std::vector<Rotation> mergedRotations;
    std::vector<std::vector<double>> mergedAvs;
    for (double time: mergedTimes) {
      Rotation rhsRot = rhs.interpolate(time);
      mergedRotations.push_back(interpolate(time)*rhsRot);
      std::vector<double> combinedAv = rhsRot.inverse()(interpolateAV(time));
      std::vector<double> rhsAv = rhs.interpolateAV(time);
      for (size_t i = 0; i < rhsAv.size(); i++) {
        combinedAv[i] += rhsAv[i];
      }
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
    std::vector<std::vector<double>> updatedAvs;
    for (size_t i = 0; i < m_avs.size(); i++) {
      updatedAvs.push_back(inverse(m_avs[i]));
    }

    m_rotations = updatedRotations;
    m_avs = updatedAvs;

    return *this;
  }
}
