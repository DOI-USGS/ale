%module interp_utils

%{
  #include "../include/ale/InterpUtils.h"
%}

namespace ale {
  enum RotationInterpolation {
    /// Spherical linear interpolation
    SLERP,
    /// Normalized linear interpolation
    NLERP
  };

  enum PositionInterpolation {
    /// Interpolate using linear interpolation
    LINEAR = 0,
    /// Interpolate using a cubic spline
    SPLINE = 1,
    /// Interpolate using Lagrange polynomials up to 8th order
    LAGRANGE = 2,
  };
}