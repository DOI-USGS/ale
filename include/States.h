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

/** A 3D cartesian vector */
struct Vec3d {
  double x;
  double y;
  double z;

  Vec3d(double x, double y, double z) : x(x), y(y), z(z) {};
  Vec3d() : x(0.0), y(0.0), z(0.0) {};
};

/** A state vector with position and velocity*/
struct State {
  Vec3d position;
  Vec3d velocity;

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
      std::vector<double> getTimes() const;
      int getReferenceFrame() const; //! Returns reference frame as NAIF ID
      bool hasMinimizedCache() const; //! Returns true if the cache has been minimized
      bool hasVelocity() const; //! Returns true if any velocities have been provided

      // Interpolate state, position, velocity 
      ale::State getState(double time, ale::interpolation interp=linear) const;
      ale::Vec3d getPosition(double time, ale::interpolation interp=linear) const;
      ale::Vec3d getVelocity(double time, ale::interpolation interp=linear) const;

      // Get start and stop time of cache 
      double getStartTime(); 
      double getStopTime(); 
      
      // Cache reduction
      void minimizeCache(double tolerance=0.01);
      std::vector<int> HermiteIndices(double tolerance, std::vector <int> indexList, 
                                                 double baseTime, double timeScale);
    private:
      std::vector<ale::State> m_states; //! Represent at states internally to keep pos, vel together
      std::vector<double> m_ephemTimes; //! Time in seconds
      int m_refFrame;  //! Naif IDs for reference frames 
      bool m_minimizedCache; 
  };
}

#endif

