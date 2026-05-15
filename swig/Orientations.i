%module orientations

%include "std_string.i"

%include "Vectors.i"
%include "Rotation.i"
%include "InterpUtils.i"

%{
  #include "../include/ale/Orientations.h"
%}

%include "../include/ale/Orientations.h"