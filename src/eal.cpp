#include "eal.h"

#include <json.hpp>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <string>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;
using namespace std;

namespace eal {

  // Parsing the JSON
  json constructStateFromIsd(const string positionRotationData) {
     // Parse the position and rotation data from isd
     json isd = json::parse(positionRotationData);
     json state;

     state["m_w"] = isd.at("w");
     state["m_x"] = isd.at("x");
     state["m_y"] = isd.at("y");
     state["m_z"] = isd.at("z");

     return state;
   }


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

    Eigen::Quaterniond dQuat(interpolate(rotations[0], times, time, interp, 1),
                             interpolate(rotations[1], times, time, interp, 1),
                             interpolate(rotations[2], times, time, interp, 1),
                             interpolate(rotations[3], times, time, interp, 1));

     Eigen::Quaterniond avQuat = quat.conjugate() * dQuat;

     vector<double> coordinate = {-2 * avQuat.x(), -2 * avQuat.y(), -2 * avQuat.z()};
     return coordinate;
  }

  // Rotation Function Functions
  vector<double> getRotation(string from, string to,
                             vector<double> coefficients, double time) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
    return coordinate;
  }

  vector<double> getAngularVelocity(string from, string to,
                                    vector<double> coefficients, double time) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
    return coordinate;
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

}
