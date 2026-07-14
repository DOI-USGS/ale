%module ale_c

%feature("autodoc", "3");

%include "std_vector.i"
%include "std_string.i"

%{
  #include <vector>
%}


%include "std_string.i"


%{
  #include "../include/ale/Vectors.h"
  #include "../include/ale/Rotation.h"
  #include "../include/ale/InterpUtils.h"
  #include "../include/ale/States.h"
  #include "../include/ale/Orientations.h"
%}

namespace ale {
  struct Vec3d {
    public:
    double x;
    double y;
    double z;
    Vec3d(const std::vector<double>& vec);
    Vec3d(double x, double y, double z);
  };
}
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(VectorDoubleVector) vector<vector<double>>;
   %template(Vec3dVector) vector<ale::Vec3d>;
}

namespace ale {
  class Rotation {
    Rotation(std::vector<double>& matrix);
    Rotation(double w, double x, double y, double z);
  };
}
namespace std {
   %template(RotationVector) vector<ale::Rotation>;
}

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

%ignore States(const std::vector<double>& ephemTimes, const std::vector<Vec3d>& positions, int refFrame=1);
%ignore States(const std::vector<double>& ephemTimes, const std::vector<State>& states, int refFrame=1);
%ignore States(const std::vector<double>& ephemTimes, const std::vector<Vec3d>& positions, const std::vector<Vec3d>& velocities, int refFrame=1);

%include "../include/ale/States.h"
%include "../include/ale/Orientations.h"