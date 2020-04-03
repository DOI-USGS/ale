#include "ale.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <Python.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace ale {


  // Temporarily moved over from States.cpp. Will be moved into interpUtils in the future.

  /** The following helper functions are used to calculate the reduced states cache and cubic hermite
  to interpolate over it. They were migrated, with minor modifications, from
  Isis::NumericalApproximation **/


  /** Evaluates a cubic hermite at time, interpTime, between the appropriate two points in x. **/
  double evaluateCubicHermite(const double interpTime, const std::vector<double>& derivs,
                              const std::vector<double>& x, const std::vector<double>& y) {
    if( (derivs.size() != x.size()) || (derivs.size() != y.size()) ) {
       throw std::invalid_argument("EvaluateCubicHermite - The size of the first derivative vector does not match the number of (x,y) data points.");
    }

    // Find the interval in which "a" exists
    int lowerIndex = ale::interpolationIndex(x, interpTime);

    double x0, x1, y0, y1, m0, m1;
    // interpTime is contained within the interval (x0,x1)
    x0 = x[lowerIndex];
    x1 = x[lowerIndex+1];
    // the corresponding known y-values for x0 and x1
    y0 = y[lowerIndex];
    y1 = y[lowerIndex+1];
    // the corresponding known tangents (slopes) at (x0,y0) and (x1,y1)
    m0 = derivs[lowerIndex];
    m1 = derivs[lowerIndex+1];

    double h, t;
    h = x1 - x0;
    t = (interpTime - x0) / h;
    return (2 * t * t * t - 3 * t * t + 1) * y0 + (t * t * t - 2 * t * t + t) * h * m0 + (-2 * t * t * t + 3 * t * t) * y1 + (t * t * t - t * t) * h * m1;
  }

  /** Evaluate velocities using a Cubic Hermite Spline at a time a, within some interval in x, **/
 double evaluateCubicHermiteFirstDeriv(const double interpTime, const std::vector<double>& deriv,
                                       const std::vector<double>& times, const std::vector<double>& y) {
    if(deriv.size() != times.size()) {
       throw std::invalid_argument("EvaluateCubicHermiteFirstDeriv - The size of the first derivative vector does not match the number of (x,y) data points.");
    }

    // find the interval in which "interpTime" exists
    int lowerIndex = ale::interpolationIndex(times, interpTime);

    double x0, x1, y0, y1, m0, m1;

    // interpTime is contained within the interval (x0,x1)
    x0 = times[lowerIndex];
    x1 = times[lowerIndex+1];

    // the corresponding known y-values for x0 and x1
    y0 = y[lowerIndex];
    y1 = y[lowerIndex+1];

    // the corresponding known tangents (slopes) at (x0,y0) and (x1,y1)
    m0 = deriv[lowerIndex];
    m1 = deriv[lowerIndex+1];

    double h, t;
    h = x1 - x0;
    t = (interpTime - x0) / h;
    if(h != 0.0) {
      return ((6 * t * t - 6 * t) * y0 + (3 * t * t - 4 * t + 1) * h * m0 + (-6 * t * t + 6 * t) * y1 + (3 * t * t - 2 * t) * h * m1) / h;

    }
    else {
      throw std::invalid_argument("Error in evaluating cubic hermite velocities, values at"
                                  "lower and upper indicies are exactly equal.");
    }
  }

  double lagrangeInterpolate(const std::vector<double>& times,
                             const std::vector<double>& values,
                             double time, int order) {
    // Ensure the times and values have the same length
    if (times.size() != values.size()) {
      throw invalid_argument("Times and values must have the same length.");
    }

    // Get the correct interpolation window
    int index = interpolationIndex(times, time);
    int windowSize = min(index + 1, (int) times.size() - index - 1);
    windowSize = min(windowSize, (int) order / 2);
    int startIndex = index - windowSize + 1;
    int endIndex = index + windowSize + 1;

    // Interpolate
    double result = 0;
    for (int i = startIndex; i < endIndex; i++) {
      double weight = 1;
      double numerator = 1;
      for (int j = startIndex; j < endIndex; j++) {
        if (i == j) {
          continue;
        }
        weight *= times[i] - times[j];
        numerator *= time - times[j];
      }
      result += numerator * values[i] / weight;
    }
    return result;
  }

  double lagrangeInterpolateDerivative(const std::vector<double>& times,
                                       const std::vector<double>& values,
                                       double time, int order) {
    // Ensure the times and values have the same length
    if (times.size() != values.size()) {
      throw invalid_argument("Times and values must have the same length.");
    }

    // Get the correct interpolation window
    int index = interpolationIndex(times, time);
    int windowSize = min(index + 1, (int) times.size() - index - 1);
    windowSize = min(windowSize, (int) order / 2);
    int startIndex = index - windowSize + 1;
    int endIndex = index + windowSize + 1;

    // Interpolate
    double result = 0;
    for (int i = startIndex; i < endIndex; i++) {
      double weight = 1;
      double derivativeWeight = 0;
      double numerator = 1;
      for (int j = startIndex; j < endIndex; j++) {
        if (i == j) {
          continue;
        }
        weight *= times[i] - times[j];
        numerator *= time - times[j];
        derivativeWeight += 1.0 / (time - times[j]);
      }
      result += numerator * values[i] * derivativeWeight / weight;
    }
    return result;
  }

 double interpolate(vector<double> points, vector<double> times, double time, interpolation interp, int d) {
   size_t numPoints = points.size();
   if (numPoints < 2) {
     throw invalid_argument("At least two points must be input to interpolate over.");
   }
   if (points.size() != times.size()) {
     throw invalid_argument("Must have the same number of points as times.");
   }

   int order;
   switch(interp) {
     case LINEAR:
       order = 2;
       break;
     case SPLINE:
       order = 4;
       break;
     case LAGRANGE:
       order = 8;
       break;
     default:
       throw invalid_argument("Invalid derivitive option, must be 0 or 1.");
   }

   double result;
   switch(d) {
     case 0:
       result = lagrangeInterpolate(times, points, time, order);
       break;
     case 1:
       result = lagrangeInterpolateDerivative(times, points, time, order);
       break;
     default:
       throw invalid_argument("Invalid derivitive option, must be 0 or 1.");
   }

   return result;
 }

 std::string getPyTraceback() {
    PyObject* err = PyErr_Occurred();
    if (err != NULL) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyObject *pystr, *module_name, *pyth_module, *pyth_func;
        char *str;
        char *full_backtrace;
        char *error_description;

        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        pystr = PyObject_Str(pvalue);
        str = PyBytes_AS_STRING(PyUnicode_AsUTF8String(pystr));
        error_description = strdup(str);

        // See if we can get a full traceback
        module_name = PyUnicode_FromString("traceback");
        pyth_module = PyImport_Import(module_name);
        Py_DECREF(module_name);

        if (pyth_module == NULL) {
            throw runtime_error("getPyTraceback - Failed to import Python traceback Library");
        }

        pyth_func = PyObject_GetAttrString(pyth_module, "format_exception");
        PyObject *pyth_val;
        pyth_val = PyObject_CallFunctionObjArgs(pyth_func, ptype, pvalue, ptraceback, NULL);

        pystr = PyObject_Str(pyth_val);
        str = PyBytes_AS_STRING(PyUnicode_AsUTF8String(pystr));
        full_backtrace = strdup(str);
        Py_DECREF(pyth_val);

        std::string join_cmd = "trace = ''.join(list(" + std::string(full_backtrace) + "))";
        PyRun_SimpleString(join_cmd.c_str());

        PyObject *evalModule = PyImport_AddModule( (char*)"__main__" );
        PyObject *evalDict = PyModule_GetDict( evalModule );
        PyObject *evalVal = PyDict_GetItemString( evalDict, "trace" );
        full_backtrace = PyBytes_AS_STRING(PyUnicode_AsUTF8String(evalVal));

        return std::string(error_description) + "\n" + std::string(full_backtrace);
    }

    // no traceback to return
    return "";
 }

 std::string loads(std::string filename, std::string props, std::string formatter, bool verbose) {
     static bool first_run = true;
     if(first_run) {
         // Initialize the Python interpreter but only once.
         first_run = !first_run;
         Py_Initialize();
     }

     // Import the file as a Python module.
     PyObject *pModule = PyImport_Import(PyUnicode_FromString("ale"));
     if(!pModule) {
       throw runtime_error("Failed to import ale. Make sure the ale python library is correctly installed.");
     }
     // Create a dictionary for the contents of the module.
     PyObject *pDict = PyModule_GetDict(pModule);

     // Get the add method from the dictionary.
     PyObject *pFunc = PyDict_GetItemString(pDict, "loads");
     if(!pFunc) {
       // import errors do not set a PyError flag, need to use a custom
       // error message instead.
       throw runtime_error("Failed to import ale.loads function from Python."
                           "This Usually indicates an error in the Ale Python Library."
                           "Check if Installed correctly and the function ale.loads exists.");
     }

     // Create a Python tuple to hold the arguments to the method.
     PyObject *pArgs = PyTuple_New(3);
     if(!pArgs) {
       throw runtime_error(getPyTraceback());
     }

     // Set the Python int as the first and second arguments to the method.
     PyObject *pStringFileName = PyUnicode_FromString(filename.c_str());
     PyTuple_SetItem(pArgs, 0, pStringFileName);

     PyObject *pStringProps = PyUnicode_FromString(props.c_str());
     PyTuple_SetItem(pArgs, 1, pStringProps);

     PyObject *pStringFormatter = PyUnicode_FromString(formatter.c_str());
     PyTuple_SetItem(pArgs, 2, pStringFormatter);

     // Call the function with the arguments.
     PyObject* pResult = PyObject_CallObject(pFunc, pArgs);

     if(!pResult) {
        throw invalid_argument("No Valid instrument found for label.");
     }

     PyObject *pResultStr = PyObject_Str(pResult);
     PyObject *temp_bytes = PyUnicode_AsUTF8String(pResultStr); // Owned reference

     if(!temp_bytes){
       throw invalid_argument(getPyTraceback());
     }
     std::string cResult;
     char *temp_str = PyBytes_AS_STRING(temp_bytes); // Borrowed pointer
     cResult = temp_str; // copy into std::string

     Py_DECREF(pResultStr);
     Py_DECREF(pStringFileName);
     Py_DECREF(pStringProps);
     Py_DECREF(pStringFormatter);

     return cResult;
 }

 json load(std::string filename, std::string props, std::string formatter, bool verbose) {
   std::string jsonstr = loads(filename, props, formatter, verbose);
   return json::parse(jsonstr);
 }
}
