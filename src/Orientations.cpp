#include "Orientations.h"

#include "InterpUtils.h"

namespace ale {

  Orientations::Orientations(
    const std::vector<Rotation> &rotations,
    const std::vector<double> &times,
    int source,
    int destination,
    const std::vector<std::vector<double>> &avs
  ) :
    m_rotations(rotations), m_avs(avs), m_times(times),
    m_source(source), m_destination(destination) {
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


  int Orientations::source() const {
    return m_source;
  }


  int Orientations::destination() const {
    return m_destination;
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
}
