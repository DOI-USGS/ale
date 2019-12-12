#include "States.h"

#include <iostream>
#include <algorithm>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>
#include <cmath>
#include <float.h>

namespace ale {
  
  /** The following helper functions are used to calculate the reduced states cache and cubic hermite 
  to interpolate over it. They were migrated, with minor modifications, from 
  Isis::NumericalApproximation **/

  /** Determines the lower index for the interpolation interval. */
  int interpolationIndex(const std::vector<double> &times, double interpTime) {
    if (times.size() < 2){
      throw std::invalid_argument("There must be at least two times.");
    }
    auto nextTimeIt = std::upper_bound(times.begin(), times.end(), interpTime);
    if (nextTimeIt == times.end()) {
      --nextTimeIt;
    }
    if (nextTimeIt != times.begin()) {
      --nextTimeIt;
    }
    return std::distance(times.begin(), nextTimeIt);
  }


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
                                       const std::vector<double>& x, const std::vector<double>& y) {
    if(deriv.size() != x.size()) {
       throw std::invalid_argument("EvaluateCubicHermiteFirstDeriv - The size of the first derivative vector does not match the number of (x,y) data points.");
    }

    // find the interval in which "a" exists
    int lowerIndex = ale::interpolationIndex(x, interpTime);

    double x0, x1, y0, y1, m0, m1;

    // interpTime is contained within the interval (x0,x1)
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
    t = (interpTime - x0) / h;
    if(h != 0.0) {
      return ((6 * t * t - 6 * t) * y0 + (3 * t * t - 4 * t + 1) * h * m0 + (-6 * t * t + 6 * t) * y1 + (3 * t * t - 2 * t) * h * m1) / h;
    }
    else {
      throw std::invalid_argument("Error in evaluating cubic hermite velocities, values at"
                                  "lower and upper indicies are exactly equal.");
    }
  }
  
  // Stadard default gsl interpolation to use if we haven't yet reduced the cache.
  // Times must be sorted in order of least to greatest
   double interpolateState(const std::vector<double>& points, const std::vector<double>& times, 
                           double time, interpolation interp, int d) {
   size_t numPoints = points.size();
   if (numPoints < 2) {
     throw std::invalid_argument("At least two points must be input to interpolate over.");
   }
   if (points.size() != times.size()) {
     throw std::invalid_argument("Invalid gsl_interp_type data, must have the same number of points as times.");
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

  // States Class
  
  // Empty constructor
   States::States() : m_refFrame(0) {
    ale::Vec3d position = {0.0, 0.0, 0.0};
    ale::Vec3d velocity = {0.0, 0.0, 0.0};
    m_states.push_back(ale::State(position, velocity)); 
    
    std::vector<double> ephems = {0.0};
    m_ephemTimes = ephems; 
  }


  States::States(const std::vector<double>& ephemTimes, const std::vector<ale::Vec3d>& positions, 
                 int refFrame) :
    m_ephemTimes(ephemTimes), m_refFrame(refFrame) {
    // Construct State vector from position and velocity vectors
    
    if (positions.size() != ephemTimes.size()) {
      throw std::invalid_argument("Length of times must match number of positions"); 
    }
    
    ale::Vec3d velocities = {0.0, 0.0, 0.0};
    for (ale::Vec3d position : positions) {
      m_states.push_back(ale::State(position, velocities)); 
    }
  }


  States::States(const std::vector<double>& ephemTimes, const std::vector<ale::Vec3d>& positions, 
                 const std::vector<ale::Vec3d>& velocities, int refFrame) : 
    m_ephemTimes(ephemTimes), m_refFrame(refFrame) {
    
    if ((positions.size() != ephemTimes.size())||(ephemTimes.size() != velocities.size())) {
      throw std::invalid_argument("Length of times must match number of positions and velocities."); 
    }
    
    for (int i=0; i < positions.size() ;i++) {
      m_states.push_back(ale::State(positions[i], velocities[i])); 
    }
  }


  States::States(const std::vector<double>& ephemTimes, const std::vector<ale::State>& states, 
                 int refFrame) :
  m_ephemTimes(ephemTimes), m_states(states), m_refFrame(refFrame) {
    if (states.size() != ephemTimes.size()) {
      throw std::invalid_argument("Length of times must match number of states."); 
    }
  }

  // Default Destructor 
  States::~States() {}

  // Getters 
  std::vector<ale::State> States::getStates() const {
    return m_states; 
  }

  std::vector<ale::Vec3d> States::getPositions() const {
    // extract positions from state vector
    std::vector<ale::Vec3d> positions; 
    
    for(ale::State state : m_states) {
        positions.push_back(state.position);
    }
    return positions; 
  }


  std::vector<ale::Vec3d> States::getVelocities() const {
    // extract velocities from state vector
    std::vector<ale::Vec3d> velocities; 
    
    for(ale::State state : m_states) {
        velocities.push_back(state.velocity);
    }
    return velocities; 
  }


  std::vector<double> States::getTimes() const {
    return m_ephemTimes; 
  }


  int States::getReferenceFrame() const {
    return m_refFrame; 
  }


  bool States::hasVelocity() const {
    std::vector<ale::Vec3d> velocities = getVelocities(); 
    bool allZero = std::all_of(velocities.begin(), velocities.end(), [](ale::Vec3d vec) 
                               { return vec.x==0.0 && vec.y==0.0 && vec.z==0.0; });
    return !allZero;
  }


  ale::State States::getState(double time, ale::interpolation interp) const {
    std::vector<double> xs, ys, zs, vxs, vys, vzs; 
    for (ale::State state : m_states) {
      xs.push_back(state.position.x);
      ys.push_back(state.position.y);
      zs.push_back(state.position.z);
      vxs.push_back(state.velocity.x);
      vys.push_back(state.velocity.y);
      vzs.push_back(state.velocity.z);
    }

    ale::Vec3d position, velocity; 

    if (interp == LINEAR || (interp == SPLINE && !hasVelocity())) {
      position = {interpolateState(xs,  m_ephemTimes, time, interp, 0),
                  interpolateState(ys,  m_ephemTimes, time, interp, 0),
                  interpolateState(zs,  m_ephemTimes, time, interp, 0)};

      velocity = {interpolateState(vxs, m_ephemTimes, time, interp, 0),
                  interpolateState(vys, m_ephemTimes, time, interp, 0),
                  interpolateState(vzs, m_ephemTimes, time, interp, 0)};
    }
    else if (interp == SPLINE && hasVelocity()){
      // Do hermite spline if velocities are available 
      double baseTime = (m_ephemTimes.at(0) + m_ephemTimes.at(m_ephemTimes.size() - 1)) / 2.;
      double timeScale = 1.0;

      std::vector<double> scaledEphemTimes; 
      for(unsigned int i = 0; i < m_ephemTimes.size(); i++) {
        scaledEphemTimes.push_back((m_ephemTimes[i] - baseTime) / timeScale);
      }

      double sTime = (time - baseTime) / timeScale;
      position.x = ale::evaluateCubicHermite(sTime, vxs, scaledEphemTimes, xs);
      position.y = ale::evaluateCubicHermite(sTime, vys, scaledEphemTimes, ys);
      position.z = ale::evaluateCubicHermite(sTime, vzs, scaledEphemTimes, zs);

      velocity.x = ale::evaluateCubicHermiteFirstDeriv(sTime, vxs, scaledEphemTimes, xs);
      velocity.y = ale::evaluateCubicHermiteFirstDeriv(sTime, vys, scaledEphemTimes, ys);
      velocity.z = ale::evaluateCubicHermiteFirstDeriv(sTime, vzs, scaledEphemTimes, zs);
    }
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
    return m_ephemTimes[0];
  }


  double States::getStopTime() {
    int len = m_ephemTimes.size(); 
    return m_ephemTimes.back();
  }
      

  States States::minimizeCache(double tolerance) {
    if (m_states.size() <= 2) {
      throw std::invalid_argument("Cache size is 2, cannot minimize.");
    }
    if (!hasVelocity()) {
      throw std::invalid_argument("The cache can only be minimized if velocity is provided.");
    }

    // Compute scaled time to use for fitting. 
    double baseTime = (m_ephemTimes.at(0) + m_ephemTimes.back())/ 2.0;
    double timeScale = 1.0;

    // Find current size of m_states
    int currentSize = m_ephemTimes.size() - 1;

    // Create 3 starting values for the new size-minimized cache
    std::vector <int> inputIndices;
    inputIndices.push_back(0);
    inputIndices.push_back(currentSize / 2);
    inputIndices.push_back(currentSize);

    // find all indices needed to make a hermite table within the appropriate tolerance
    std::vector <int> indexList = hermiteIndices(tolerance, inputIndices, baseTime, timeScale);

    // Update m_states and m_ephemTimes to only save the necessary indicies in the index list
    std::vector<ale::State> tempStates; 
    std::vector<double> tempTimes; 

    for(int i : indexList) {
      tempStates.push_back(m_states[i]);
      tempTimes.push_back(m_ephemTimes[i]);
    }
    return States(tempTimes, tempStates, m_refFrame); 
   }


  std::vector<int> States::hermiteIndices(double tolerance, std::vector<int> indexList, 
                                                 double baseTime, double timeScale) {
    unsigned int n = indexList.size();
    double sTime;

    std::vector<double> x, y, z, vx, vy, vz, scaledEphemTimes; 
    for(unsigned int i = 0; i < indexList.size(); i++) {
      scaledEphemTimes.push_back((m_ephemTimes[indexList[i]] - baseTime) / timeScale);
      x.push_back(m_states[indexList[i]].position.x);
      y.push_back(m_states[indexList[i]].position.y);
      z.push_back(m_states[indexList[i]].position.z);
      vx.push_back(m_states[i].velocity.x);
      vy.push_back(m_states[i].velocity.y);
      vz.push_back(m_states[i].velocity.z); 
    }

    // loop through the saved indices from the end
    for(unsigned int i = indexList.size() - 1; i > 0; i--) {
      double xerror = 0;
      double yerror = 0;
      double zerror = 0;

      // check every value of the original kernel values within interval
      for(int line = indexList[i-1] + 1; line < indexList[i]; line++) {
        sTime = (m_ephemTimes[line] - baseTime) / timeScale;

        // find the errors at each value
        xerror = fabs(ale::evaluateCubicHermite(sTime, vx, scaledEphemTimes, x) - m_states[line].position.x);
        yerror = fabs(ale::evaluateCubicHermite(sTime, vy, scaledEphemTimes, y) - m_states[line].position.y);
        zerror = fabs(ale::evaluateCubicHermite(sTime, vz, scaledEphemTimes, z) - m_states[line].position.z);
        
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
      indexList = hermiteIndices(tolerance, indexList, baseTime, timeScale);
    }
    return indexList;
  }
}


