#include "eal.h"

#include <json.hpp>

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

  // Positional Functions

  // Position Data Functions
  vector<double> getPosition(vector<vector<double>> coords, vector<double> times, double time,
                             const gsl_interp_type* interp) {
    // Check that all of the data sizes are okay
    // TODO is there a cleaner way to do this? We're going to have to do this a lot.
    if (coords.size() != 3) {
      throw invalid_argument("Invalid input positions, expected three vectors.");
    }

    // GSL setup
    vector<double> coordinate = {0.0, 0.0, 0.0};

    coordinate = { interpolate(coords[0], times, time, interp, none),
                   interpolate(coords[1], times, time, interp, none),
                   interpolate(coords[2], times, time, interp, none) };

    return coordinate;
  }

  vector<double> getVelocity(vector<vector<double>> coords, vector<double> times,
                             double time, const gsl_interp_type* interp) {
    // Check that all of the data sizes are okay
    // TODO is there a cleaner way to do this? We're going to have to do this a lot.
    if (coords.size() != 3) {
     throw invalid_argument("Invalid input positions, expected three vectors.");
    }

    // GSL setup
    vector<double> coordinate = {0.0, 0.0, 0.0};

    coordinate = { interpolate(coords[0], times, time, interp, first),
                   interpolate(coords[1], times, time, interp, first),
                   interpolate(coords[2], times, time, interp, first) };

    return coordinate;
  }

  // Postion Function Functions
  vector<double> getPosition(vector<double> coeffs, double time) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
    return coordinate;
  }

  vector<double> getVelocity(vector<double> coeffs, double time) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
    return coordinate;
  }

  // Rotation Data Functions

  // Rotation Data Functions
  vector<double> getRotation(string from, string to, vector<vector<double>> rotations,
                             vector<double> times, double time,  const gsl_interp_type* interp) {
    vector<double> coordinate = {0.0, 0.0, 0.0};
    return coordinate;
  }

  vector<double> getAngularVelocity(string from, string to, vector<vector<double>> rotations,
                                    vector<double> times, double time,  const gsl_interp_type* interp) {
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


 double interpolate(vector<double> points, vector<double> times, double time, const gsl_interp_type* interp, derivation d) {
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

   gsl_interp *interpolator = gsl_interp_alloc(interp, numPoints);
   gsl_interp_init(interpolator, &times[0], &points[0], numPoints);
   gsl_interp_accel *acc = gsl_interp_accel_alloc();

   // GSL evaluate
   double result;
   switch(d) {
     case none:
       result = gsl_interp_eval(interpolator, &times[0], &points[0], time, acc);
       break;
     case first:
       result = gsl_interp_eval_deriv(interpolator, &times[0], &points[0], time, acc);
       break;
     case second:
       result = gsl_interp_eval_deriv2(interpolator, &times[0], &points[0], time, acc);
       break;
     default:
       throw invalid_argument("Invalid Derivation option.");
       break;
   }

   // GSL clean up
   gsl_interp_free(interpolator);
   gsl_interp_accel_free(acc);

   return result;
 }

}
