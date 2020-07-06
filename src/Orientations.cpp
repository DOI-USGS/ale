#include "ale/Orientations.h"

#include "ale/InterpUtils.h"

namespace ale {

  Orientations::Orientations(
    const std::vector<Rotation> &rotations,
    const std::vector<double> &times,
    const std::vector<Vec3d> &avs,
    const Rotation &const_rot,
    const std::vector<int> const_frames,
    const std::vector<int> time_dependent_frames
  ) :
    m_rotations(rotations), m_avs(avs), m_times(times), m_timeDepFrames(time_dependent_frames), m_constFrames(const_frames), m_constRotation(const_rot) {
    if (m_rotations.size() < 1 || m_times.size() < 1) {
      throw std::invalid_argument("There must be at least one rotation and time.");
    }
    if (m_rotations.size() != m_times.size()) {
      throw std::invalid_argument("The number of rotations and times must be the same.");
    }
    if ( !m_avs.empty() && (m_avs.size() != m_times.size()) ) {
      throw std::invalid_argument("The number of angular velocities and times must be the same.");
    }
  }


  std::vector<Rotation> Orientations::getRotations() const {
    return m_rotations;
  }


  std::vector<Vec3d> Orientations::getAngularVelocities() const {
    return m_avs;
  }


  std::vector<double> Orientations::getTimes() const {
    return m_times;
  }

  std::vector<int> Orientations::getTimeDependentFrames() const {
    return m_timeDepFrames;
  }

  std::vector<int> Orientations::getConstantFrames() const {
    return m_constFrames;
  }

  Rotation Orientations::getConstantRotation() const {
    return m_constRotation;
  }

  Rotation Orientations::interpolateTimeDep(
    double time,
    RotationInterpolation interpType
  ) const {
    Rotation timeDepRotation;
    if (m_times.size() > 1) {
      int interpIndex = interpolationIndex(m_times, time);
      double t = (time - m_times[interpIndex]) / (m_times[interpIndex + 1] - m_times[interpIndex]);
      timeDepRotation = m_rotations[interpIndex].interpolate(m_rotations[interpIndex + 1], t, interpType);
    }
    else if (m_avs.empty()) {
      timeDepRotation = m_rotations.front();
    }
    else {
      double t = time - m_times.front();
      std::vector<double> axis = {m_avs.front().x, m_avs.front().y, m_avs.front().z};
      double angle = t * m_avs.front().norm();
      Rotation newRotation(axis, angle);
      timeDepRotation = newRotation * m_rotations.front();
    }
    return timeDepRotation;
  }

  Rotation Orientations::interpolate(
    double time,
    RotationInterpolation interpType
  ) const {
    return m_constRotation * interpolateTimeDep(time, interpType);
  }


  Vec3d Orientations::interpolateAV(double time) const {
    Vec3d interpAv;
    if (m_times.size() > 1) {
      int interpIndex = interpolationIndex(m_times, time);
      double t = (time - m_times[interpIndex]) / (m_times[interpIndex + 1] - m_times[interpIndex]);
      interpAv = Vec3d(linearInterpolate(m_avs[interpIndex], m_avs[interpIndex + 1], t));
    }
    else {
      interpAv = m_avs.front();
    }
    return interpAv;
  }

  Vec3d Orientations::rotateVectorAt(
    double time,
    const Vec3d &vector,
    RotationInterpolation interpType,
    bool invert
  ) const {
    Rotation interpRot = interpolate(time, interpType);
    if (invert) {
      interpRot = interpRot.inverse();
    }
    return interpRot(vector);
  }


  State Orientations::rotateStateAt(
    double time,
    const State &state,
    RotationInterpolation interpType,
    bool invert
  ) const {
    Rotation interpRot = interpolate(time, interpType);
    Vec3d av(0.0, 0.0, 0.0);
    if (!m_avs.empty()) {
      av = interpolateAV(time);
    }
    if (invert) {
      Vec3d negAv = interpRot(av);
      av = {-negAv.x, -negAv.y, -negAv.z};
      interpRot = interpRot.inverse();
    }
    return interpRot(state, av);
  }


  Orientations &Orientations::addConstantRotation(const Rotation &addedConst) {
    m_constRotation = addedConst * m_constRotation;
    return *this;
  }


  Orientations &Orientations::operator*=(const Orientations &rhs) {
    std::vector<double> mergedTimes = orderedVecMerge(m_times, rhs.m_times);
    std::vector<Rotation> mergedRotations;
    std::vector<Vec3d> mergedAvs;
    for (double time: mergedTimes) {
      // interpolate includes the constant rotation, so invert it to undo that
      Rotation inverseConst = m_constRotation.inverse();
      Rotation rhsRot = rhs.interpolate(time);
      mergedRotations.push_back(inverseConst*interpolate(time)*rhsRot);
      Vec3d combinedAv = rhsRot.inverse()(interpolateAV(time));
      Vec3d rhsAv = rhs.interpolateAV(time);
      combinedAv += rhsAv;
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

    Rotation inverseRhs = rhs.inverse();
    std::vector<Vec3d> updatedAvs;
    for (size_t i = 0; i < m_avs.size(); i++) {
      updatedAvs.push_back(inverseRhs(m_avs[i]));
    }

    m_rotations = updatedRotations;
    m_avs = updatedAvs;

    return *this;
  }


  Orientations Orientations::inverse() const {
    std::vector<Rotation> newRotations;
    // The time dependent rotation is applied and the constant rotation is applied second,
    // so we have to subsume the constant rotations into the time dependent rotations
    // in the inverse.
    Rotation constInverseRotation = m_constRotation.inverse();
    for (size_t i = 0; i < m_rotations.size(); i++) {
      newRotations.push_back(m_rotations[i].inverse() * constInverseRotation);
    }

    std::vector<Vec3d> rotatedAvs;
    for (size_t i = 0; i < m_avs.size(); i++) {
      Vec3d rotatedAv = -1.0 * (m_constRotation * m_rotations[i])(m_avs[i]);
      rotatedAvs.push_back(rotatedAv);
    }

    // Because the constant rotation was subsumed by the time dependet rotations, everything
    // is a time dependent rotation in the inverse.
    std::vector<int> newTimeDepFrames;
    std::vector<int>::const_reverse_iterator timeDepIt = m_timeDepFrames.crbegin();
    for (; timeDepIt != m_timeDepFrames.crend(); timeDepIt++) {
      newTimeDepFrames.push_back(*timeDepIt);
    }
    std::vector<int>::const_reverse_iterator constIt = m_constFrames.crbegin();
    // Skip the last frame in the constant list because it's the first frame
    // in the time dependent list
    if (constIt != m_constFrames.crend()) {
      constIt++;
    }
    for (; constIt != m_constFrames.rend(); constIt++) {
      newTimeDepFrames.push_back(*constIt);
    }

    return Orientations(newRotations, m_times, rotatedAvs, Rotation(1, 0, 0, 0),
                        std::vector<int>(), newTimeDepFrames);
  }


  Orientations operator*(Orientations lhs, const Rotation &rhs) {
    return lhs *= rhs;
  }


  Orientations operator*(const Rotation &lhs, Orientations rhs) {
    return rhs.addConstantRotation(lhs);
  }


  Orientations operator*(Orientations lhs, const Orientations &rhs) {
    return lhs *= rhs;
  }
}
