#ifndef ALE_INCLUDE_ALE_H
#define ALE_INCLUDE_ALE_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include <gsl/gsl_interp.h>
 
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace ale {

  /// Interpolation enum for defining different methods of interpolation
  enum interpolation {
    /// Interpolate using linear interpolation
    linear,
    /// Interpolate using spline interpolation
    spline
  };

  /**
   *@brief Get the position of the spacecraft at a given time based on a set of coordinates, and their associated times
   *@param coords A vector of double vectors of coordinates
   *@param times A double vector of times
   *@param time Time to observe the spacecraft's position at
   *@param interp Interpolation type
   *@return A vector double of the spacecrafts position
   */
  std::vector<double> getPosition(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, interpolation interp);
  /**
   *@brief Get the velocity of the spacecraft at a given time based on a set of coordinates, and their associated times
   *@param coords A vector of double vectors of coordinates
   *@param times A double vector of times
   *@param time Time to observe the spacecraft's velocity at
   *@param interp Interpolation type
   *@return A vector double of the spacecrafts velocity
   */
  std::vector<double> getVelocity(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, const interpolation interp);
  /**
   *@brief Get the position of the spacecraft at a given time based on a derived function from a set of coeffcients
   *@param coeffs A vector of double vector of coeffcients
   *@param time Time to observe the spacecraft's position at
   *@return A vector double of the spacecrafts position
   */
  std::vector<double> getPosition(std::vector<std::vector<double>> coeffs, double time);

  /**
   *@brief Get the velocity of the spacecraft at a given time based on a derived function from a set of coeffcients
   *@param coeffs A vector of double vector of coeffcients
   *@param time Time to observe the spacecraft's velocity at
   *@return A vector double of the spacecrafts velocity
   */
  std::vector<double> getVelocity(std::vector<std::vector<double>> coeffs, double time);

  /**
   *@brief Get the rotation of the spacecraft at a given time based on a set of rotations, and their associated times
   *@param rotations A vector of double vector of rotations
   *@param times A double vector of times
   *@param time Time to observe the spacecraft's rotation at
   *@param interp Interpolation type
   *@return A vector double of the spacecrafts rotation
   */
  std::vector<double> getRotation(std::vector<std::vector<double>> rotations,
                                  std::vector<double> times,
                                  double time, interpolation interp);

  /**
   *@brief Get the angular velocity of the spacecraft at a given time based on a set of rotations, and their associated times
   *@param rotations A vector of double vector of rotations
   *@param times A double vector of times
   *@param time Time to observe the spacecraft's angular velocity at
   *@param interp Interpolation type
   *@return A vector double of the spacecrafts angular velocity
   */
  std::vector<double> getAngularVelocity(std::vector<std::vector<double>> rotations,
                                         std::vector<double> times,
                                         double time, interpolation interp);

   /**
    *@brief Get the rotation of the spacecraft at a given time based on a derived function from a set of coeffcients
    *@param coeffs A vector of double vector of coeffcients
    *@param time Time to observe the spacecraft's rotation at
    *@return A vector double of the spacecrafts rotation
    */
  std::vector<double> getRotation(std::vector<std::vector<double>> coeffs, double time);

  /**
   *@brief Get the angular velocity of the spacecraft at a given time based on a derived function from a set of coeffcients
   *@param coeffs A vector of double vector of coeffcients
   *@param time Time to observe the spacecraft's angular velocity at
   *@return A vector double of the spacecrafts angular velocity
   */
  std::vector<double> getAngularVelocity(std::vector<std::vector<double>> coeffs, double time);

  /**
   *@brief Generates a derivatives in respect to time from a polynomial constructed using the given coeffcients, time, and derivation number
   *@param coeffs A double vector of coefficients can be any number of coefficients
   *@param time Time to use when deriving
   *@param d The order of the derivative to generate (Currently supports 0, 1, and 2)
   *@return Evalutation of the given polynomial as a double
   */
  double evaluatePolynomial(std::vector<double> coeffs, double time, int d);

  /**
   *@brief Interpolates the spacecrafts position along a path generated from a set of points,
                times, and a time of observation
   *@param points A double vector of points
   *@param times A double vector of times
   *@param time A double to use as the time of observation
   *@param interp An interpolation enum dictating what type of interpolation to use
   *@param d The order of the derivative to generate when interpolating
                      (Currently supports 0, 1, and 2)
   *@return
   */
  double interpolate(std::vector<double> points, std::vector<double> times, double time, interpolation interp, int d);
  std::string loads(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);

  json load(std::string filename, std::string props="", std::string formatter="usgscsm", bool verbose=true);


}

#endif // ALE_H
