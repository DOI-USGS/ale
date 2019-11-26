#include "States.h"

#include <stdexcept> 
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_poly.h>

namespace ale {

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
  }

  States::States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, int refFrame) :
  ephemTimes(ephemTimes), refFrame(refFrame) {
    // Construct State vector from position and velocity vectors

    // if time isn't the same length, also error out
    ale::Vec3d velocities = {0.0, 0.0, 0.0};
    for (ale::Vec3d position : positions) { // if sizes aren't equal, error out.
      states.push_back(ale::State(position, velocities)); 
    }
  }


  States::States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, 
                   std::vector<ale::Vec3d> velocities, int refFrame) : 
  ephemTimes(ephemTimes), refFrame(refFrame) {
    // if time isn't the same length, also error out
    for (int i=0; i < positions.size() ;i++) { // if sizes aren't equal, error out.
      states.push_back(ale::State(positions[i], velocities[i])); 
    }
  }

  States::States(std::vector<double> ephemTimes, std::vector<ale::State> states, int refFrame) :
  ephemTimes(ephemTimes), states(states), refFrame(refFrame) {}

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

    for(ale::State state : states) {
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

    // GSL setup
//    vector<double> coordinate = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//
//    coordinate = { interpolate(xs,  ephemTimes, time, interp, 0),
//                   interpolate(ys,  ephemTimes, time, interp, 0),
//                   interpolate(zs,  ephemTimes, time, interp, 0),
//                   interpolate(vxs, ephemTimes, time, interp, 0),
//                   interpolate(vys, ephemTimes, time, interp, 0),
//                   interpolate(vzs, ephemTimes, time, interp, 0),
//                  };

//    return coordinate; -- as a different getState option


    ale::Vec3d position = {interpolateState(xs,  ephemTimes, time, interp, 0),
                           interpolateState(ys,  ephemTimes, time, interp, 0),
                           interpolateState(zs,  ephemTimes, time, interp, 0)};

    ale::Vec3d velocity = {interpolateState(vxs, ephemTimes, time, interp, 0),
                           interpolateState(vys, ephemTimes, time, interp, 0),
                           interpolateState(vzs, ephemTimes, time, interp, 0)};

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
   void minimizeCache(double tolerance=0.1) { // default tolerance? 
     // write this function
   }
}


