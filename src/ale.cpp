#include "ale.h"

#include <nlohmann/json.hpp>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <Python.h>

#include <string>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace ale {

  // Position Data Functions
  vector<double> getPosition(vector<vector<double>> coords, vector<double> times, double time,
                             interpolation interp) {
    // Check that all of the data sizes are okay
    // TODO is there a cleaner way to do this? We're going to have to do this a lot.
    if (coords.size() != 3) {
      throw invalid_argument("Invalid input positions, expected three vectors.");
    }

    // GSL setup
    vector<double> coordinate = {0.0, 0.0, 0.0};

    coordinate = { interpolate(coords[0], times, time, interp, 0),
                   interpolate(coords[1], times, time, interp, 0),
                   interpolate(coords[2], times, time, interp, 0) };

    return coordinate;
  }

  vector<double> getVelocity(vector<vector<double>> coords, vector<double> times,
                             double time, interpolation interp) {
    // Check that all of the data sizes are okay
    // TODO is there a cleaner way to do this? We're going to have to do this a lot.
    if (coords.size() != 3) {
     throw invalid_argument("Invalid input positions, expected three vectors.");
    }

    // GSL setup
    vector<double> coordinate = {0.0, 0.0, 0.0};

    coordinate = { interpolate(coords[0], times, time, interp, 1),
                   interpolate(coords[1], times, time, interp, 1),
                   interpolate(coords[2], times, time, interp, 1) };

    return coordinate;
  }

  // Postion Function Functions
  // vector<double> coeffs = [[cx_0, cx_1, cx_2 ..., cx_n],
  //                          [cy_0, cy_1, cy_2, ... cy_n],
  //                          [cz_0, cz_1, cz_2, ... cz_n]]
  // The equations evaluated by this function are:
  //                x = cx_n * t^n + cx_n-1 * t^(n-1) + ... + cx_0
  //                y = cy_n * t^n + cy_n-1 * t^(n-1) + ... + cy_0
  //                z = cz_n * t^n + cz_n-1 * t^(n-1) + ... + cz_0
  vector<double> getPosition(vector<vector<double>> coeffs, double time) {

    if (coeffs.size() != 3) {
      throw invalid_argument("Invalid input coeffs, expected three vectors.");
    }

    vector<double> coordinate = {0.0, 0.0, 0.0};
    coordinate[0] = evaluatePolynomial(coeffs[0], time, 0); // X
    coordinate[1] = evaluatePolynomial(coeffs[1], time, 0); // Y
    coordinate[2] = evaluatePolynomial(coeffs[2], time, 0); // Z

    return coordinate;
  }


  // Velocity Function
  // Takes the coefficients from the position equation
  vector<double> getVelocity(vector<vector<double>> coeffs, double time) {

    if (coeffs.size() != 3) {
      throw invalid_argument("Invalid input coeffs, expected three vectors.");
    }

    vector<double> coordinate = {0.0, 0.0, 0.0};
    coordinate[0] = evaluatePolynomial(coeffs[0], time, 1); // X
    coordinate[1] = evaluatePolynomial(coeffs[1], time, 1); // Y
    coordinate[2] = evaluatePolynomial(coeffs[2], time, 1); // Z

    return coordinate;
  }


  // Rotation Data Functions
  vector<double> getRotation(vector<vector<double>> rotations,
                             vector<double> times, double time,  interpolation interp) {
    // Check that all of the data sizes are okay
    // TODO is there a cleaner way to do this? We're going to have to do this a lot.
    if (rotations.size() != 4) {
     throw invalid_argument("Invalid input rotations, expected four vectors.");
    }

    // Alot of copying and reassignment becuase conflicting data types
    // probably should rethink our vector situation to guarentee contiguous
    // memory. Should be easy to switch to a contiguous column-major format
    // if we stick with Eigen.
    for (size_t i = 0; i<rotations[0].size(); i++) {
      Eigen::Quaterniond quat(rotations[0][i], rotations[1][i], rotations[2][i], rotations[3][i]);
      quat.normalize();

      rotations[0][i] = quat.w();
      rotations[1][i] = quat.x();
      rotations[2][i] = quat.y();
      rotations[3][i] = quat.z();
    }

    // GSL setup
    vector<double> coordinate = {0.0, 0.0, 0.0, 0.0};

    coordinate = { interpolate(rotations[0], times, time, interp, 0),
                   interpolate(rotations[1], times, time, interp, 0),
                   interpolate(rotations[2], times, time, interp, 0),
                   interpolate(rotations[3], times, time, interp, 0)};

    // Eigen::Map to ensure the array isn't copied, only the pointer is
    Eigen::Map<Eigen::MatrixXd> quat(coordinate.data(), 4, 1);
    quat.normalize();
    return coordinate;
  }

  vector<double> getAngularVelocity(vector<vector<double>> rotations,
                                    vector<double> times, double time,  interpolation interp) {
    // Check that all of the data sizes are okay
    // TODO is there a cleaner way to do this? We're going to have to do this a lot.
    if (rotations.size() != 4) {
     throw invalid_argument("Invalid input rotations, expected four vectors.");
    }

    double data[] = {0,0,0,0};
    for (size_t i = 0; i<rotations[0].size(); i++) {
      Eigen::Quaterniond quat(rotations[0][i], rotations[1][i], rotations[2][i], rotations[3][i]);
      quat.normalize();
      rotations[0][i] = quat.w();
      rotations[1][i] = quat.x();
      rotations[2][i] = quat.y();
      rotations[3][i] = quat.z();
    }

    // GSL setup


    Eigen::Quaterniond quat(interpolate(rotations[0], times, time, interp, 0),
                            interpolate(rotations[1], times, time, interp, 0),
                            interpolate(rotations[2], times, time, interp, 0),
                            interpolate(rotations[3], times, time, interp, 0));
    quat.normalize();

    Eigen::Quaterniond dQuat(interpolate(rotations[0], times, time, interp, 1),
                             interpolate(rotations[1], times, time, interp, 1),
                             interpolate(rotations[2], times, time, interp, 1),
                             interpolate(rotations[3], times, time, interp, 1));

     Eigen::Quaterniond avQuat = quat.conjugate() * dQuat;

     vector<double> coordinate = {-2 * avQuat.x(), -2 * avQuat.y(), -2 * avQuat.z()};
     return coordinate;
  }

  // Rotation Function Functions
  std::vector<double> getRotation(vector<vector<double>> coeffs, double time) {

    if (coeffs.size() != 3) {
      throw invalid_argument("Invalid input coefficients, expected three vectors.");
    }

    vector<double> rotation = {0.0, 0.0, 0.0};

    rotation[0] = evaluatePolynomial(coeffs[0], time, 0); // X
    rotation[1] = evaluatePolynomial(coeffs[1], time, 0); // Y
    rotation[2] = evaluatePolynomial(coeffs[2], time, 0); // Z

    Eigen::Quaterniond quat;
    quat = Eigen::AngleAxisd(rotation[0] * M_PI / 180, Eigen::Vector3d::UnitZ())
                * Eigen::AngleAxisd(rotation[1] * M_PI / 180, Eigen::Vector3d::UnitX())
                * Eigen::AngleAxisd(rotation[2] * M_PI / 180, Eigen::Vector3d::UnitZ());

    quat.normalize();

    vector<double> rotationQ = {quat.w(), quat.x(), quat.y(), quat.z()};
    return rotationQ;
  }

  vector<double> getAngularVelocity(vector<vector<double>> coeffs, double time) {

    if (coeffs.size() != 3) {
      throw invalid_argument("Invalid input coefficients, expected three vectors.");
    }

    double phi = evaluatePolynomial(coeffs[0], time, 0); // X
    double theta = evaluatePolynomial(coeffs[1], time, 0); // Y
    double psi = evaluatePolynomial(coeffs[2], time, 0); // Z

    double phi_dt = evaluatePolynomial(coeffs[0], time, 1);
    double theta_dt = evaluatePolynomial(coeffs[1], time, 1);
    double psi_dt = evaluatePolynomial(coeffs[2], time, 1);

    Eigen::Quaterniond quat1, quat2;
    quat1 = Eigen::AngleAxisd(phi * M_PI / 180, Eigen::Vector3d::UnitZ());
    quat2 =  Eigen::AngleAxisd(theta * M_PI / 180, Eigen::Vector3d::UnitX());

    Eigen::Vector3d velocity =  phi_dt * Eigen::Vector3d::UnitZ();
    velocity += theta_dt * (quat1 *  Eigen::Vector3d::UnitX());
    velocity += psi_dt * (quat1 * quat2 *  Eigen::Vector3d::UnitZ());

    return {velocity[0], velocity[1], velocity[2]};
  }

  // Polynomial evaluation helper function
  // The equation evaluated by this function is:
  //                x = cx_0 + cx_1 * t^(1) + ... + cx_n * t^n
  // The d parameter is for which derivative of the polynomial to compute.
  // Supported options are
  //   0: no derivative
  //   1: first derivative
  //   2: second derivative
  double evaluatePolynomial(vector<double> coeffs, double time, int d){
    if (coeffs.empty()) {
      throw invalid_argument("Invalid input coeffs, must be non-empty.");
    }

    if (d < 0) {
      throw invalid_argument("Invalid derivative degree, must be non-negative.");
    }

    vector<double> derivatives(d + 1);
    gsl_poly_eval_derivs(coeffs.data(), coeffs.size(), time,
                         derivatives.data(), derivatives.size());

    return derivatives.back();
  }

 double interpolate(vector<double> points, vector<double> times, double time, interpolation interp, int d) {
   size_t numPoints = points.size();
   if (numPoints < 2) {
     throw invalid_argument("At least two points must be input to interpolate over.");
   }
   if (points.size() != times.size()) {
     throw invalid_argument("Invalid gsl_interp_type data, must have the same number of points as times.");
   }
   if (time < times.front() || time > times.back()) {
     throw invalid_argument("Invalid gsl_interp_type time, outside of input times.");
   }

   // convert our interp enum into a GSL one,
   // should be easy to add non GSL interp methods here later
   const gsl_interp_type *interp_methods[] = {gsl_interp_linear, gsl_interp_cspline};

   gsl_interp *interpolator = gsl_interp_alloc(interp_methods[interp], numPoints);
   gsl_interp_init(interpolator, &times[0], &points[0], numPoints);
   gsl_interp_accel *acc = gsl_interp_accel_alloc();

   // GSL evaluate
   double result;
   switch(d) {
     case 0:
       result = gsl_interp_eval(interpolator, &times[0], &points[0], time, acc);
       break;
     case 1:
       result = gsl_interp_eval_deriv(interpolator, &times[0], &points[0], time, acc);
       break;
     case 2:
       result = gsl_interp_eval_deriv2(interpolator, &times[0], &points[0], time, acc);
       break;
     default:
       throw invalid_argument("Invalid derivitive option, must be 0, 1 or 2.");
       break;
   }

   // GSL clean up
   gsl_interp_free(interpolator);
   gsl_interp_accel_free(acc);

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
         atexit(Py_Finalize);
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
