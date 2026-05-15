%module vectors

%include "std_vector.i"

%{
  #include "../include/ale/Vectors.h"
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
   %template(Vec3dVector) vector<ale::Vec3d>;
}