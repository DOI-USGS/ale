#include "States.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>
#include <cmath>
#include <float.h>

namespace ale {
  
  // The following helper functions are used to calculate the reduced m_states cache and cubic hermite 
  // to interpolate over it. They were migrated (with minor modifications from 
  // Isis::NumericalApproximation
  
  // Checks to see if an individual value, a, is within the doman specified by x
  bool InsideDomain(const double a, std::vector<double> x) {
    if(a + DBL_EPSILON < *min_element(x.begin(), x.end())) {
      return false;
    }
    if(a - DBL_EPSILON > *max_element(x.begin(), x.end())) {
      return false;
    }
    return true;
  }

  /* Find the index of the x-value in the data set that is just
   * below the input value, a. If a is below the domain minimum,
   * the method returns 0 as the lower index.  If a is above the
   * domain maximum, it returns the second to last index of the
   * data set, Size()-2, as the lower index.*/
  int FindIntervalLowerIndex(const double a, std::vector<double> x) {
    // find the interval in which "a" exists
    if(ale::InsideDomain(a, x)) {
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

  // Evaluates a cubic hermite at time, a, between two points in x. 
  double EvaluateCubicHermite(const double a, std::vector<double> derivs, std::vector<double> x, std::vector<double> y) {
    if( (derivs.size() != x.size()) || (derivs.size() != y.size()) ) {
//      ReportException(IException::User, "EvaluateCubicHermite()",
//                      "Invalid arguments. The size of the first derivative vector does not match the number of (x,y) data points.",
//                      _FILEINFO_);
    }

    // Find the interval in which "a" exists
    int lowerIndex = ale::FindIntervalLowerIndex(a, x);

    // If the time is exactly equal to either endpoint of the interval,
    // return the value at the appropriate endpoint
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

    double h, t;
    h = x1 - x0;
    t = (a - x0) / h;
    return (2 * t * t * t - 3 * t * t + 1) * y0 + (t * t * t - 2 * t * t + t) * h * m0 + (-2 * t * t * t + 3 * t * t) * y1 + (t * t * t - t * t) * h * m1;
  }

  // Evaluate velocities using a Cubic Hermite Spline at a time a, within some interval in x, 
 double EvaluateCubicHermiteFirstDeriv(const double a, std::vector<double> deriv, std::vector<double> x, std::vector<double> y) {
    if(deriv.size() != x.size()) {
//      ReportException(IException::User, "EvaluateCubicHermiteFirstDeriv()",
//                      "Invalid arguments. The size of the first derivative vector does not match the number of (x,y) data points.",
//                      _FILEINFO_);
    }
    // find the interval in which "a" exists
    int lowerIndex = ale::FindIntervalLowerIndex(a, x);

    // if a is exactly equal to the time at either endpoint, return the 
    // value at the appropriate endpoint
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
    if(h != 0.0) {
      return ((6 * t * t - 6 * t) * y0 + (3 * t * t - 4 * t + 1) * h * m0 + (-6 * t * t + 6 * t) * y1 + (3 * t * t - 2 * t) * h * m1) / h;
    }
    else {
      return 0;  // Should never happen
    }
  }
  
  // Stadard default gsl interpolation to use if we haven't yet reduced the cache.
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

  // Empty constructor
   States::States() : m_refFrame(0) {
    ale::Vec3d position = {0.0, 0.0, 0.0};
    ale::Vec3d velocity = {0.0, 0.0, 0.0};
    m_states.push_back(ale::State(position, velocity)); 
    
    std::vector<double> ephems = {0.0};
    m_ephemTimes = ephems; 
    m_minimizedCache = false;
  }

  States::States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, int refFrame) :
  m_ephemTimes(ephemTimes), m_refFrame(refFrame) {
    // Construct State vector from position and velocity vectors

    // if time isn't the same length, also error out
    ale::Vec3d velocities = {0.0, 0.0, 0.0};
    for (ale::Vec3d position : positions) { // if sizes aren't equal, error out.
      m_states.push_back(ale::State(position, velocities)); 
    }
    m_minimizedCache=false;
  }


  States::States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, 
                   std::vector<ale::Vec3d> velocities, int refFrame) : 
  m_ephemTimes(ephemTimes), m_refFrame(refFrame) {
    // if time isn't the same length, also error out
    for (int i=0; i < positions.size() ;i++) { // if sizes aren't equal, error out.
      m_states.push_back(ale::State(positions[i], velocities[i])); 
    }
    m_minimizedCache=false;
  }

  States::States(std::vector<double> ephemTimes, std::vector<ale::State> states, int refFrame) :
  m_ephemTimes(ephemTimes), m_states(states), m_refFrame(refFrame), m_minimizedCache(false) {}

  // Default Destructor 
  States::~States() {}

  // Getters 

  //! Returns state vectors (6-element positions&velocities) 
  std::vector<ale::State> States::getStates() const {
    return m_states; 
  }

  //! Returns the current positions inside the m_states object
  std::vector<ale::Vec3d> States::getPositions() const {
    // extract positions from state vector
    std::vector<ale::Vec3d> positions; 
    
    for(ale::State state : m_states) {
        ale::Vec3d position = state.position;
        positions.push_back(position);
    }
    return positions; 
  }


  //! Returns the current velocities inside the m_states object
  std::vector<ale::Vec3d> States::getVelocities() const {
    // extract velocities from state vector
    std::vector<ale::Vec3d> velocities; 
    
    for(ale::State state : m_states) {
        ale::Vec3d velocity = state.velocity;
        velocities.push_back(velocity);
    }
    return velocities; 
  }


  //! Returns the current times in the m_states object
  std::vector<double> States::getTimes() const {
    return m_ephemTimes; 
  }

  //! Returns reference frame as NAIF ID
  int States::getReferenceFrame() const {
    return m_refFrame; 
  }

  //! Returns true if cache has been minimized
  bool States::hasMinimizedCache() const {
    return m_minimizedCache;
  }

  //! Returns true if velocities have been set
  bool States::hasVelocity() const {
    std::vector<ale::Vec3d> velocities = getVelocities(); 
    bool allZero = std::all_of(velocities.begin(), velocities.end(), [](ale::Vec3d vec) { return vec.x==0.0 && vec.y==0.0 && vec.z==0.0; });
    return true;
  }


  //! Returns a single state by interpolating state, position, velocity.
  //! If the Cache has been minimized, a cubic hermite is used to interpolate the 
  //! position and velocity over the reduced cache. 
  //! If not, a standard gsl interpolation will be odne.
  ale::State States::getState(double time, ale::interpolation interp) const {
    std::vector<double> xs, ys, zs, vxs, vys, vzs; 
    for (ale::State state : m_states) {
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

    ale::Vec3d position, velocity; 

    if (!hasMinimizedCache()) {
      position = {interpolateState(xs,  m_ephemTimes, time, interp, 0),
                  interpolateState(ys,  m_ephemTimes, time, interp, 0),
                  interpolateState(zs,  m_ephemTimes, time, interp, 0)};

      velocity = {interpolateState(vxs, m_ephemTimes, time, interp, 0),
                  interpolateState(vys, m_ephemTimes, time, interp, 0),
                  interpolateState(vzs, m_ephemTimes, time, interp, 0)};
    }
    else {
      // We have a minimized cache
      double baseTime = (m_ephemTimes.at(0) + m_ephemTimes.at(m_ephemTimes.size() - 1)) / 2.;
      double timeScale = 1.0;

      std::vector<double> scaledm_ephemTimes; 
      for(unsigned int i = 0; i < m_ephemTimes.size(); i++) {
        scaledm_ephemTimes.push_back((m_ephemTimes[i] - baseTime) / timeScale);
      }

      double sTime = (time - baseTime) / timeScale;
      position.x = ale::EvaluateCubicHermite(sTime, vxs, scaledm_ephemTimes, xs);
      position.y = ale::EvaluateCubicHermite(sTime, vys, scaledm_ephemTimes, ys);
      position.z = ale::EvaluateCubicHermite(sTime, vzs, scaledm_ephemTimes, zs);

      velocity.x = ale::EvaluateCubicHermiteFirstDeriv(sTime, vxs, scaledm_ephemTimes, xs);
      velocity.y = ale::EvaluateCubicHermiteFirstDeriv(sTime, vys, scaledm_ephemTimes, ys);
      velocity.z = ale::EvaluateCubicHermiteFirstDeriv(sTime, vzs, scaledm_ephemTimes, zs);
    }
    return ale::State(position, velocity);
  }

  // Gets a position at a single time. Operates the same way as getState()
  ale::Vec3d States::getPosition(double time, ale::interpolation interp) const {
    ale::State interpState = getState(time, interp); 
    return interpState.position;
  }

  // Gets a velocity at a single time. Operates the same way as getState()
  ale::Vec3d States::getVelocity(double time, ale::interpolation interp) const {
    ale::State interpState = getState(time, interp); 
    return interpState.velocity;
  }


  // Returns the first ephemeris time
  double States::getStartTime() {
    return m_ephemTimes[0];
  }


  // Returns the last ephemeris time
  double States::getStopTime() {
    int len = m_ephemTimes.size(); 
    return m_ephemTimes[len -1];
  }
      
  // Perform a cache reduction. 
  // After running this, getm_states(), getPositions(), and getVelocities() will return
  // caches of reduced size, and getState(), getPosition(), and getVelocity() will
  // returns values interpolated over the reduced cache using a cubic hermite spline
  void States::minimizeCache(double tolerance) { // default tolerance? 
    if (m_states.size() == 1) {
//      print("error: cache size is only 1, no need to minimize");
      return;
    }
    if (!hasVelocity()) {
//      print("error: can only minimize cache if velocities are provided");
      return; 
    }

    // Compute scaled time to use for fitting. 
    double baseTime = (m_ephemTimes.at(0) + m_ephemTimes.at(m_ephemTimes.size() - 1)) / 2.0;
    double timeScale = 1.0;

    // Find current size of m_states
    int currentSize = m_ephemTimes.size() - 1;

    // Create 3 starting values for the new size-minimized cache
    std::vector <int> inputIndices;
    inputIndices.push_back(0);
    inputIndices.push_back(currentSize / 2);
    inputIndices.push_back(currentSize);

    // find all indices needed to make a hermite table within the appropriate tolerance
    std::vector <int> indexList = HermiteIndices(tolerance, inputIndices, baseTime, timeScale);

    // Remove all lines from cache vectors that are not in the index list
    for(int i = currentSize; i >= 0; i--) {
      if(!std::binary_search(indexList.begin(), indexList.end(), i)) {
        m_states.erase(m_states.begin() + i);
        m_ephemTimes.erase(m_ephemTimes.begin() + i);
      }
    }
    m_minimizedCache=true;
   }

  
  // Calculates the points (indicies) which need to be kept for the hermite spline to interpolate
  // between to mantain a maximum error of tolerance.
  std::vector<int> States::HermiteIndices(double tolerance, std::vector <int> indexList, 
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
        xerror = fabs(ale::EvaluateCubicHermite(sTime, vx, scaledEphemTimes, x) - m_states[line].position.x);
        yerror = fabs(ale::EvaluateCubicHermite(sTime, vy, scaledEphemTimes, y) - m_states[line].position.y);
        zerror = fabs(ale::EvaluateCubicHermite(sTime, vz, scaledEphemTimes, z) - m_states[line].position.z);
        
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


