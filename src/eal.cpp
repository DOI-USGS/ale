#include "eal.h"

#include <json.hpp>

#include <iostream>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include <string>
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
  vector<double> getPosition(vector<vector<double>> coords, vector<double> times,
                             interpolation interp, double time) {
    // Check that all of the data sizes are okay
    // TODO is there a cleaner way to do this? We're going to have to do this a lot.
    if (coords.size() != 3) {
      throw invalid_argument("Invalid input positions, expected three vectors.");
    }

    // GSL setup
    vector<double> coordinate = {0.0, 0.0, 0.0};
    switch(interp) {
      case linear:
        coordinate = { linearInterpolate(coords[0], times, time),
                       linearInterpolate(coords[1], times, time),
                       linearInterpolate(coords[2], times, time) };
        break;

      case spline:
      coordinate = { splineInterpolate(coords[0], times, time),
                     splineInterpolate(coords[1], times, time),
                     splineInterpolate(coords[2], times, time) };
        break;

      default:
        throw invalid_argument("Invalid interpolation method.");
        break;
    }

    return coordinate;
  }

  vector<double> getVelocity(vector<vector<double>> coords, vector<double> times,
                             interpolation interp, double time) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
    return coordinate;
  }

  // Postion Function Functions
  // vector<double> coeffs = [[cx_n, cx_n-1, cx_n-2, ... cx_0],
  //                          [cy_n, cy_n-1, cy_n-2, ... cy_0],
  //                          [cz_n, cz_n-1, cz_n-2, ... cz_0]]
  // The equations evaluated by this function are:
  //                x = cx_n * t^n + cx_n-1 * t^(n-1) + ... + cx_0
  //                y = cy_n * t^n + cy_n-1 * t^(n-1) + ... + cy_0
  //                z = cz_n * t^n + cz_n-1 * t^(n-1) + ... + cz_0
  vector<double> getPosition(vector<vector<double>> coeffs, double time) {
    if ( time < 0.0) {
      throw invalid_argument("Invalid input time, must be non-negative.");
    }

    if (coeffs.size() != 3) {
      throw invalid_argument("Invalid input coeffs, expected three vectors.");
    }

    // make sure all coeffs sizes are equal, else throw error...
    if ((coeffs[0].size() != coeffs[1].size()) || 
        (coeffs[1].size() != coeffs[2].size())) {
      throw invalid_argument("Invalid input coeffs, each must represent the same degree polynomial.");
    }

    int degree = coeffs[0].size() - 1; 

    vector<double> coordinate = {0.0, 0.0, 0.0};
    
    for (unsigned i=0; i < coeffs[0].size(); i++) {
      coordinate[0] += coeffs[0][i] * pow(time, (degree-i)); // X 
      coordinate[1] += coeffs[1][i] * pow(time, (degree-i)); // Y
      coordinate[2] += coeffs[2][i] * pow(time, (degree-i)); // Z
    }

    return coordinate;
  }


  // Velocity Function Functions
  // vector<double> coeffs = [[cvx_n, cvx_n-1, cvx_n-2, ... cvx_0],
  //                          [cvy_n, cvy_n-1, cvy_n-2, ... cvy_0],
  //                          [cvz_n, cvz_n-1, cvz_n-2, ... cvz_0]]
  // The equations evaluated by this function are:
  //                v_x = cvx_n * t^n + cvx_n-1 * t^(n-1) + ... + cvx_0
  //                v_y = cvy_n * t^n + cvy_n-1 * t^(n-1) + ... + cvy_0
  //                v_z = cvz_n * t^n + cvz_n-1 * t^(n-1) + ... + cvz_0
  vector<double> getVelocity(vector<vector<double>> coeffs, double time) {
    // This is the same calculation, for now? A polynomial is a polynomial -- should this be
    // something else? 
    vector<double> coordinate = getPosition(coeffs, time);
    return coordinate;
  }

  // Rotation Data Functions

  // Rotation Data Functions
  vector<double> getRotation(string from, string to, vector<vector<double>> rotations,
                             vector<double> times, interpolation interp, double time) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
    return coordinate;
  }

  vector<double> getAngularVelocity(string from, string to, vector<vector<double>> rotations,
                                    vector<double> times, interpolation interp, double time) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
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

  // Interpolation helper functions
  double linearInterpolate(vector<double> points, vector<double> times, double time) {
    size_t numPoints = points.size();
    if (numPoints < 2) {
      throw invalid_argument("At least two points must be input to interpolate over.");
    }
    if (points.size() != times.size()) {
      throw invalid_argument("Invalid interpolation data, must have the same number of points as times.");
    }
    if (time < times.front() || time > times.back()) {
      throw invalid_argument("Invalid interpolation time, outside of input times.");
    }

    // GSL setup
    gsl_interp *interpolator = gsl_interp_alloc(gsl_interp_linear, numPoints);
    gsl_interp_init(interpolator, &times[0], &points[0], numPoints);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();

    // GSL evaluate
    double result = gsl_interp_eval(interpolator, &times[0], &points[0], time, acc);

    // GSL clean up
    gsl_interp_free(interpolator);
    gsl_interp_accel_free(acc);

    return result;
  }

  double splineInterpolate(vector<double> points, vector<double> times, double time) {
    size_t numPoints = points.size();
    if (numPoints < 2) {
      throw invalid_argument("Invalid interpolation data, at least two points are required to interpolate.");
    }
    if (points.size() != times.size()) {
      throw invalid_argument("Invalid interpolation data, must have the same number of points as times.");
    }
    if (time < times.front() || time > times.back()) {
      throw invalid_argument("Invalid interpolation time, outside of input times.");
    }

    // GSL setup
    gsl_interp *interpolator = gsl_interp_alloc(gsl_interp_cspline, numPoints);
    gsl_interp_init(interpolator, &times[0], &points[0], numPoints);
    gsl_interp_accel *acc = gsl_interp_accel_alloc();

    // GSL evaluate
    double result = gsl_interp_eval(interpolator, &times[0], &points[0], time, acc);

    // GSL clean up
    gsl_interp_free(interpolator);
    gsl_interp_accel_free(acc);

    return result;
  }

}
