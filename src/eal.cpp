#include "eal.h"

#include <json.hpp>

#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include <string>

using json = nlohmann::json;

using namespace eal;


// Parsing the JSON
json constructStateFromIsd(const std::string positionRotationData) {
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
vector<double> getPosition(vector<vector<double>> coords, vector<double> times,
                           interpolation interp, double time) {
  // Check that all of the data sizes are okay
  // TODO is there a cleaner way to do this? We're going to have to do this a lot.
  if (coords.size() < 3) {
    //exception
  }
  size_t numPoints = times.size();
  if (numPoints == 0) {
    //exception
  }
  if (coords[0].size() != numPoints ||
      coords[1].size() != numPoints ||
      coords[2].size() != numPoints) {
    //exception
  }

  // GSL setup
  vector<double> coordinate = {0.0, 0.0, 0.0};
  gsl_interp *xInterpolator = nullptr;
  gsl_interp *yInterpolator = nullptr;
  gsl_interp *zInterpolator = nullptr;
  switch(interp) {
    case linear:
      xInterpolator = gsl_interp_alloc(gsl_interp_linear, numPoints);
      yInterpolator = gsl_interp_alloc(gsl_interp_linear, numPoints);
      zInterpolator = gsl_interp_alloc(gsl_interp_linear, numPoints);
      break;

    case spline:
      xInterpolator = gsl_interp_alloc(gsl_interp_cspline, numPoints);
      yInterpolator = gsl_interp_alloc(gsl_interp_cspline, numPoints);
      zInterpolator = gsl_interp_alloc(gsl_interp_cspline, numPoints);
      break;

    default:
      //exception
      return coordinate;
      break;
  }
  gsl_interp_init(xInterpolator, &times[0], &coords[0][0], numPoints);
  gsl_interp_init(yInterpolator, &times[0], &coords[1][0], numPoints);
  gsl_interp_init(zInterpolator, &times[0], &coords[2][0], numPoints);
  gsl_interp_accel *xAcc = gsl_interp_accel_alloc();
  gsl_interp_accel *yAcc = gsl_interp_accel_alloc();
  gsl_interp_accel *zAcc = gsl_interp_accel_alloc();

  // Actually evaluate
  coordinate = { gsl_interp_eval(xInterpolator, &times[0], &coords[0][0], time, xAcc),
                 gsl_interp_eval(yInterpolator, &times[0], &coords[1][0], time, yAcc),
                 gsl_interp_eval(zInterpolator, &times[0], &coords[2][0], time, zAcc) };

  // GSL clean up
  gsl_interp_free(xInterpolator);
  gsl_interp_free(yInterpolator);
  gsl_interp_free(zInterpolator);
  gsl_interp_accel_free(xAcc);
  gsl_interp_accel_free(yAcc);
  gsl_interp_accel_free(zAcc);

  return coordinate;
}

vector<double> getVelocity(vector<vector<double>> coords, vector<double> times,
                           interpolation interp, double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
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
vector<double> getRotation(string from, string to, vector<double> coefficients,
                           double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}

vector<double> getAngularVelocity(string from, string to, vector<double> coefficients,
                                  double time) {
  vector<double> coordinate = {0.0, 0.0, 0.0};
  return coordinate;
}
