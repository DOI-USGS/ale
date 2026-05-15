%module ale_c

%feature("autodoc", "3");

%include "std_vector.i"
%include "std_string.i"

%{
  #include <vector>
%}

namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

%include "States.i"
%include "Orientations.i"