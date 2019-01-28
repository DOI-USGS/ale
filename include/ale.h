#ifndef ALE_INCLUDE_ALE_H
#define ALE_INCLUDE_ALE_H
/// @file

#include <json.hpp>
#include <string>
#include <vector>

#include <gsl/gsl_interp.h>

namespace ale {

  enum interpolation {
    linear,
    spline
  };

  /**
   *@brief Get the position of the spacecraft at a given time based on a set of coordinates, and their associated times
   *@param coords a vector of double vectors of coordinates
   *@param times a double vector of times
   *@param time time to observe the spacecraft's position at
   *@param interp interpolation type
   */
  std::vector<double> getPosition(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, interpolation interp);
  /**
   *@brief Get the velocity of the spacecraft at a given time based on a set of coordinates, and their associated times
   *@param coords a vector of double vectors of coordinates
   *@param times a double vector of times
   *@param time time to observe the spacecraft's velocity at
   *@param interp interpolation type
   */
  std::vector<double> getVelocity(std::vector<std::vector<double>> coords,
                                  std::vector<double> times,
                                  double time, const interpolation interp);
  /**
   *@brief Get the position of the spacecraft at a given time based on a derived function from a set of coeffcients
   *@param coeffs a vector of double vector of coeffcients
   *@param time time to observe the spacecraft's position at
   */
  std::vector<double> getPosition(std::vector<std::vector<double>> coeffs, double time);

  /**
   *@brief Get the velocity of the spacecraft at a given time based on a derived function from a set of coeffcients
   *@param coeffs a vector of double vector of coeffcients
   *@param time time to observe the spacecraft's velocity at
   */
  std::vector<double> getVelocity(std::vector<std::vector<double>> coeffs, double time);

  /**
   *@brief Get the rotation of the spacecraft at a given time based on a set of rotations, and their associated times
   *@param rotations a vector of double vector of rotations
   *@param times a double vector of times
   *@param time time to observe the spacecraft's rotation at
   *@param interp interpolation type
   */
  std::vector<double> getRotation(std::vector<std::vector<double>> rotations,
                                  std::vector<double> times,
                                  double time, interpolation interp);

  /**
   *@brief Get the angular velocity of the spacecraft at a given time based on a set of rotations, and their associated times
   *@param rotations a vector of double vector of rotations
   *@param times a double vector of times
   *@param time time to observe the spacecraft's angular velocity at
   *@param interp interpolation type
   */
  std::vector<double> getAngularVelocity(std::vector<std::vector<double>> rotations,
                                         std::vector<double> times,
                                         double time, interpolation interp);

   /**
    *@brief Get the rotation of the spacecraft at a given time based on a derived function from a set of coeffcients
    *@param coeffs a vector of double vector of coeffcients
    *@param time time to observe the spacecraft's rotation at
    */
  std::vector<double> getRotation(std::vector<std::vector<double>> coeffs, double time);

  /**
   *@brief Get the angular velocity of the spacecraft at a given time based on a derived function from a set of coeffcients
   *@param coeffs a vector of double vector of coeffcients
   *@param time time to observe the spacecraft's angular velocity at
   */
  std::vector<double> getAngularVelocity(std::vector<std::vector<double>> coeffs, double time);

  double evaluatePolynomial(std::vector<double> coeffs, double time, int d);
  double interpolate(std::vector<double> points, std::vector<double> times, double time, interpolation interp, int d);
  std::string load(std::string filename);
}

#endif // ALE_H
