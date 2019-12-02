#include "States.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>
#include <cmath>

#define DBL_EPSILON 2.2204460492503131e-16

namespace ale {
  bool InsideDomain(const double a, std::vector<double> x) {
    try {
      if(a + DBL_EPSILON < *min_element(x.begin(), x.end())) {
        return false;
      }
      if(a - DBL_EPSILON > *max_element(x.begin(), x.end())) {
        return false;
      }
    }
    catch(...) { // catch exception from DomainMinimum(), DomainMaximum()
//      throw IException(e,
//                       e.errorType(),
//                       "InsideDomain() - Unable to compute domain boundaries",
//                       _FILEINFO_);
    }
    return true;
  }


  // copoied and modified from Numerical Approximation
  int FindIntervalLowerIndex(const double a, std::vector<double> x) {
    if(ale::InsideDomain(a, x)) {
      // find the interval in which "a" exists
      std::vector<double>::iterator pos;
      // find position in vector that is greater than or equal to "a"
      pos = upper_bound(x.begin(), x.end(), a);
      int upperIndex = 0;
      if(pos != x.end()) {
        upperIndex = distance(x.begin(), pos);
      }
      else {
        upperIndex = x.size() - 1;
      }
      return upperIndex - 1;
    }
    else if((a + DBL_EPSILON) < *min_element(x.begin(), x.end())) {
      return 0;
    }
    else {
      return x.size() - 2;
    }
  }


  double EvaluateCubicHermite(const double a, std::vector<double> derivs, std::vector<double> x, std::vector<double> y) {
    //  algorithm was found at en.wikipedia.org/wiki/Cubic_Hermite_spline
    //  it seems to produce same answers, as the NumericalAnalysis book

    if(derivs.size() != x.size() || derivs.size() != y.size()) {
//      ReportException(IException::User, "EvaluateCubicHermite()",
  //                    "Invalid arguments. The size of the first derivative vector does not match the number of (x,y) data points.",
    //                  _FILEINFO_);
    }
    // find the interval in which "a" exists
    int lowerIndex = ale::FindIntervalLowerIndex(a, x);

    // we know that "a" is within the domain since this is verified in
    // Evaluate() before this method is called, thus n <= Size()
    if(a == x[lowerIndex]) {
      return y[lowerIndex];
    }
    if(a == x[lowerIndex+1]) {
      return y[lowerIndex+1];
    }

    double x0, x1, y0, y1, m0, m1;
    // a is contained within the interval (x0,x1)
    x0 = x[lowerIndex];
    x1 = x[lowerIndex+1];
    // the corresponding known y-values for x0 and x1
    y0 = y[lowerIndex];
    y1 = y[lowerIndex+1];
    // the corresponding known tangents (slopes) at (x0,y0) and (x1,y1)
    m0 = derivs[lowerIndex];
    m1 = derivs[lowerIndex+1];

    //  following algorithm found at en.wikipedia.org/wiki/Cubic_Hermite_spline
    //  seems to produce same answers, is it faster?

    double h, t;
    h = x1 - x0;
    t = (a - x0) / h;
    return (2 * t * t * t - 3 * t * t + 1) * y0 + (t * t * t - 2 * t * t + t) * h * m0 + (-2 * t * t * t + 3 * t * t) * y1 + (t * t * t - t * t) * h * m1;
  }

 double EvaluateCubicHermiteFirstDeriv(const double a, std::vector<double> deriv, std::vector<double> x, std::vector<double> y) {
    if(deriv.size() != x.size()) {
//      ReportException(IException::User, "EvaluateCubicHermiteFirstDeriv()",
//                      "Invalid arguments. The size of the first derivative vector does not match the number of (x,y) data points.",
//                      _FILEINFO_);
    }
    // find the interval in which "a" exists
    int lowerIndex = ale::FindIntervalLowerIndex(a, x);

    // we know that "a" is within the domain since this is verified in
    // Evaluate() before this method is called, thus n <= Size()
    if(a == x[lowerIndex]) {
      return deriv[lowerIndex];
    }
    if(a == x[lowerIndex+1]) {
      return deriv[lowerIndex+1];
    }

    double x0, x1, y0, y1, m0, m1;
    // a is contained within the interval (x0,x1)
    x0 = x[lowerIndex];
    x1 = x[lowerIndex+1];
    // the corresponding known y-values for x0 and x1
    y0 = y[lowerIndex];
    y1 = y[lowerIndex+1];
    // the corresponding known tangents (slopes) at (x0,y0) and (x1,y1)
    m0 = deriv[lowerIndex];
    m1 = deriv[lowerIndex+1];

    double h, t;
    h = x1 - x0;
    t = (a - x0) / h;
    if(h != 0.) {
      return ((6 * t * t - 6 * t) * y0 + (3 * t * t - 4 * t + 1) * h * m0 + (-6 * t * t + 6 * t) * y1 + (3 * t * t - 2 * t) * h * m1) / h;
    }
    else {
      return 0;  // Should never happen
    }
  }
  
   double interpolateState(std::vector<double> points, std::vector<double> times, double time, interpolation interp, int d) {
   size_t numPoints = points.size();
   if (numPoints < 2) {
     throw std::invalid_argument("At least two points must be input to interpolate over.");
   }
   if (points.size() != times.size()) {
     throw std::invalid_argument("Invalid gsl_interp_type data, must have the same number of points as times.");
   }
   if (time < times.front() || time > times.back()) {
     throw std::invalid_argument("Invalid gsl_interp_type time, outside of input times.");
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
       throw std::invalid_argument("Invalid derivitive option, must be 0, 1 or 2.");
       break;
   }

   // GSL clean up
   gsl_interp_free(interpolator);
   gsl_interp_accel_free(acc);

   return result;
 }

  States::States() : refFrame(0) {
    ale::Vec3d position = {0.0, 0.0, 0.0};
    ale::Vec3d velocity = {0.0, 0.0, 0.0};
    states.push_back(ale::State(position, velocity)); 
    
    std::vector<double> ephems = {0.0};
    ephemTimes = ephems; 
    minimizedCache = false;
  }

  States::States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, int refFrame) :
  ephemTimes(ephemTimes), refFrame(refFrame) {
    // Construct State vector from position and velocity vectors

    // if time isn't the same length, also error out
    ale::Vec3d velocities = {0.0, 0.0, 0.0};
    for (ale::Vec3d position : positions) { // if sizes aren't equal, error out.
      states.push_back(ale::State(position, velocities)); 
    }
    minimizedCache=false;
  }


  States::States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, 
                   std::vector<ale::Vec3d> velocities, int refFrame) : 
  ephemTimes(ephemTimes), refFrame(refFrame) {
    // if time isn't the same length, also error out
    for (int i=0; i < positions.size() ;i++) { // if sizes aren't equal, error out.
      states.push_back(ale::State(positions[i], velocities[i])); 
    }
    minimizedCache=false;
  }

  States::States(std::vector<double> ephemTimes, std::vector<ale::State> states, int refFrame) :
  ephemTimes(ephemTimes), states(states), refFrame(refFrame), minimizedCache(false) {}

  // Default Destructor 
  States::~States() {}

  // Getters 

  //! Returns state vectors (6-element positions&velocities) 
  std::vector<ale::State> States::getStates() const {
    return states; 
  }

  std::vector<ale::Vec3d> States::getPositions() const {
    // extract positions from state vector
    std::vector<ale::Vec3d> positions; 
    
    for(ale::State state : states) {
        ale::Vec3d position = state.position;
        positions.push_back(position);
    }
    return positions; 
  }


  std::vector<ale::Vec3d> States::getVelocities() const {
    // extract velocities from state vector
    std::vector<ale::Vec3d> velocities; 
    
    for(ale::State state : states) {
        ale::Vec3d velocity = state.velocity;
        velocities.push_back(velocity);
    }
    return velocities; 
  }


  std::vector<double> States::getTimes() const {
    return ephemTimes; 
  }

  //! Returns reference frame as NAIF ID
  int States::getReferenceFrame() const {
    return refFrame; 
  }

  // Interpolate state, position, velocity 
  ale::State States::getState(double time, ale::interpolation interp) const {
    std::vector<double> xs, ys, zs, vxs, vys, vzs; 
    for (ale::State state : states) {
      ale::Vec3d position = state.position;
      double x = position.x;
      double y = position.y;
      double z = position.z;
      xs.push_back(x);
      ys.push_back(y);
      zs.push_back(z);
      ale::Vec3d velocity = state.velocity;
      double vx = velocity.x;
      double vy = velocity.y;
      double vz = velocity.z;
      vxs.push_back(vx);
      vys.push_back(vy);
      vzs.push_back(vz);
    }

    if (!minimizedCache) {
      ale::Vec3d position = {interpolateState(xs,  ephemTimes, time, interp, 0),
                             interpolateState(ys,  ephemTimes, time, interp, 0),
                             interpolateState(zs,  ephemTimes, time, interp, 0)};

      ale::Vec3d velocity = {interpolateState(vxs, ephemTimes, time, interp, 0),
                             interpolateState(vys, ephemTimes, time, interp, 0),
                             interpolateState(vzs, ephemTimes, time, interp, 0)};

      return ale::State(position, velocity);
    }

    // We have a minimized cache
    double baseTime = (ephemTimes.at(0) + ephemTimes.at(ephemTimes.size() - 1)) / 2.;
    double timeScale = 1.0;

    std::vector<double> scaledEphemTimes; 
    for(unsigned int i = 0; i < ephemTimes.size(); i++) {
      scaledEphemTimes.push_back((ephemTimes[i] - baseTime) / timeScale);
    }
    double sTime = (time - baseTime) / timeScale;
    ale::Vec3d position; 
    position.x = ale::EvaluateCubicHermite(sTime, vxs, scaledEphemTimes, xs);
    position.y = ale::EvaluateCubicHermite(sTime, vys, scaledEphemTimes, ys);
    position.z = ale::EvaluateCubicHermite(sTime, vzs, scaledEphemTimes, zs);

    ale::Vec3d velocity;
    velocity.x = ale::EvaluateCubicHermiteFirstDeriv(sTime, vxs, scaledEphemTimes, xs);
    velocity.y = ale::EvaluateCubicHermiteFirstDeriv(sTime, vys, scaledEphemTimes, ys);
    velocity.z = ale::EvaluateCubicHermiteFirstDeriv(sTime, vzs, scaledEphemTimes, zs);
    return ale::State(position, velocity);
  }

  ale::Vec3d States::getPosition(double time, ale::interpolation interp) const {
    ale::State interpState = getState(time, interp); 
    return interpState.position;
  }

  ale::Vec3d States::getVelocity(double time, ale::interpolation interp) const {
    ale::State interpState = getState(time, interp); 
    return interpState.velocity;
  }

   double States::getStartTime() {
     return ephemTimes[0];
   }

   double States::getStopTime() {
     int len = ephemTimes.size(); 
     return ephemTimes[len -1];
   }
      
   // Cache reduction
   // alternative names reduceLookup, reduceTable, reduceLookupTable, reduceState, reduceStateCache
  void States::minimizeCache(double tolerance) { // default tolerance? 
    if (states.size() == 1) {
//      print("error: cache size is only 1, no need to minimize");
      return;
    }
    if (false) {//!hasVelocity()) {
//      print("error: can only minimize cache if velocities are provided");
      return; 
    }

    // Compute scaled time to use for fitting. 
    double baseTime = (ephemTimes.at(0) + ephemTimes.at(ephemTimes.size() - 1)) / 2.;
    double timeScale = 1.0;

    // find current size of cache
    int currentSize = ephemTimes.size() - 1;

    // create 3 starting values for the new table
    std::vector <int> inputIndices;
    inputIndices.push_back(0);
    inputIndices.push_back(currentSize / 2);
    inputIndices.push_back(currentSize);

    // find all indices needed to make a hermite table within the appropriate tolerance
    std::vector <int> indexList = HermiteIndices(tolerance, inputIndices, baseTime, timeScale);
/*    for (int i=0; i < indexList.size(); i++) {
      std::cout << "index: " << indexList[i] << std::endl; 
    }*/
    // Remove all lines from cache vectors that are not in the index list
    for(int i = currentSize; i >= 0; i--) {
      if(!std::binary_search(indexList.begin(), indexList.end(), i)) {
        states.erase(states.begin() + i);
        ephemTimes.erase(ephemTimes.begin() + i);
      }
    }
    minimizedCache=true;
   }

std::vector<int> States::HermiteIndices(double tolerance, std::vector <int> indexList, 
                                                 double baseTime, double timeScale) {
    unsigned int n = indexList.size();
    double sTime;

    std::vector<double> x, y, z, vx, vy, vz, scaledEphemTimes; 
    for(unsigned int i = 0; i < indexList.size(); i++) {
      scaledEphemTimes.push_back((ephemTimes[indexList[i]] - baseTime) / timeScale);
      x.push_back(states[indexList[i]].position.x);
      y.push_back(states[indexList[i]].position.y);
      z.push_back(states[indexList[i]].position.z);
      vx.push_back(states[i].velocity.x);
      vy.push_back(states[i].velocity.y);
      vz.push_back(states[i].velocity.z); 
    }

    // loop through the saved indices from the end
    for(unsigned int i = indexList.size() - 1; i > 0; i--) {
      double xerror = 0;
      double yerror = 0;
      double zerror = 0;

      // check every value of the original kernel values within interval
      for(int line = indexList[i-1] + 1; line < indexList[i]; line++) {
        sTime = (ephemTimes[line] - baseTime) / timeScale;

        // find the errors at each value
        xerror = fabs(ale::EvaluateCubicHermite(sTime, vx, scaledEphemTimes, x) - states[line].position.x);
        yerror = fabs(ale::EvaluateCubicHermite(sTime, vy, scaledEphemTimes, y) - states[line].position.y);
        zerror = fabs(ale::EvaluateCubicHermite(sTime, vz, scaledEphemTimes, z) - states[line].position.z);
        
//        std::cout << "xerror: " << xerror << std::endl;
        if(xerror > tolerance || yerror > tolerance || zerror > tolerance) {
          // if any error is greater than tolerance, no need to continue looking, break
          break;
        }
      }

      if(xerror < tolerance && yerror < tolerance && zerror < tolerance) {
        // if errors are less than tolerance after looping interval, no new point is necessary
        continue;
      }
      else {
        // if any error is greater than tolerance, add midpoint of interval to indexList vector
        indexList.push_back((indexList[i] + indexList[i-1]) / 2);
      }
    }

    if(indexList.size() > n) {
      std::sort(indexList.begin(), indexList.end());
      indexList = HermiteIndices(tolerance, indexList, baseTime, timeScale);
    }
    return indexList;
  }


}


