%module rotation

%{
  #include "../include/ale/Rotation.h"
%}

namespace ale {
  class Rotation {
    Rotation(std::vector<double>& matrix);
    Rotation(double w, double x, double y, double z);
  };
}
namespace std {
   %template(RotationVector) vector<ale::Rotation>;
}