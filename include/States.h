#ifndef ALE_STATES_H
#define ALE_STATES_H

#include<vector>
#include <gsl/gsl_interp.h>
 
namespace ale {

enum interpolation {
  /// Interpolate using linear interpolation
  linear,
  /// Interpolate using spline interpolation
  spline
};

// Consider converting to a class if useful or adding units and/or ref frame
/** A 3D cartesian vector */
struct Vec3d {
  double x;
  double y;
  double z;

  Vec3d(double x, double y, double z) : x(x), y(y), z(z) {};
  Vec3d() : x(0.0), y(0.0), z(0.0) {};
};

// Consider converting to a class if useful or adding units and/or ref frame
/** A state vector with position and velocity*/
struct State {
  Vec3d position;
  Vec3d velocity;

  //State(double x, double y, double z, double vx, double vy, double vz) : x(x), y(y), z(z), 
  //    vx(vx), vy(vy), vz(vz) {};
  State(ale::Vec3d position, ale::Vec3d velocity) : position(position), velocity(velocity) {}; 
  State() {};
};

class States {
    public:
      // Constructors
      States();
      States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, int refFrame=1);
      States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, 
             std::vector<ale::Vec3d> velocities, int refFrame=1);
      States(std::vector<double> ephemTimes, std::vector<ale::State> states, int refFrame=1); 

      ~States();

      // Getters
      std::vector<ale::State> getStates() const; //! Returns state vectors (6-element positions&velocities) 
      std::vector<ale::Vec3d> getPositions() const; 
      std::vector<ale::Vec3d> getVelocities() const;
      // getOriginalVelocities / positions or state? pre-reduction? 
      std::vector<double> getTimes() const;
      int getReferenceFrame() const; //! Returns reference frame as NAIF ID

      // Interpolate state, position, velocity 
      ale::State getState(double time, ale::interpolation interp=linear) const;
      ale::Vec3d getPosition(double time, ale::interpolation interp=linear) const;
      ale::Vec3d getVelocity(double time, ale::interpolation interp=linear) const; // needed? 

      // Get start and stop time of cache 
      double getStartTime(); 
      double getStopTime(); 
      
      // Cache reduction
      // alternative names reduceLookup, reduceTable, reduceLookupTable, reduceState, reduceStateCache
      void minimizeCache(double tolerance=0.1); // default tolerance? 

    private:
      std::vector<ale::State> states; //! Represent at states internally to keep pos, vel together
      std::vector<double> ephemTimes; //! Time in seconds (since _____ ) 
      int refFrame;  //! Naif IDs for reference frames 
  };

}

#endif

