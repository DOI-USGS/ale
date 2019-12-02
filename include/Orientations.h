#ifndef ALE_ORIENTATIONS_H
#define ALE_ORIENTATIONS_H

#include <vector>

#include "Rotation.h"

namespace ale {
  class Orientations {
  public:
    Orientations() {};
    Orientations(
      const std::vector<Rotation> &rotations,
      const std::vector<double> &times,
      int source,
      int destination,
      const std::vector<std::vector<double>> &avs = std::vector<std::vector<double>>()
    );
    ~Orientations() {};

    std::vector<Rotation> rotations() const;
    std::vector<std::vector<double>> angularVelocities() const;
    std::vector<double> times() const;
    int source() const;
    int destination() const;

    Rotation interpolate(
      double time,
      RotationInterpolation interpType=slerp
    ) const;
    std::vector<double> interpolateAV(double time) const;
    std::vector<double> rotateAt(
      double time,
      const std::vector<double> &vector,
      RotationInterpolation interpType=slerp,
      bool invert=false
    ) const;

  private:
    std::vector<Rotation> m_rotations;
    std::vector<std::vector<double>> m_avs;
    std::vector<double> m_times;
    int m_source;
    int m_destination;
  };
}

#endif
