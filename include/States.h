#ifndef ALE_STATES_H
#define ALE_STATES_H

#include<vector>
#include <stdexcept>
#include <gsl/gsl_interp.h>
 
namespace ale {

enum interpolation {
  /// Interpolate using linear interpolation
  LINEAR,
  /// Interpolate using spline interpolation
  SPLINE
};

/** A 3D cartesian vector */
struct Vec3d {
  double x;
  double y;
  double z;

  // Accepts an {x,y,z} vector
  Vec3d(std::vector<double> vec) {
    if (vec.size() != 3) {
      throw std::invalid_argument("Input vector must have 3 entries.");
    }
    x = vec[0];
    y = vec[1];
    z = vec[2];
  }; 

  Vec3d(double x, double y, double z) : x(x), y(y), z(z) {};
  Vec3d() : x(0.0), y(0.0), z(0.0) {};
};

/** A state vector with position and velocity*/
struct State {
  Vec3d position;
  Vec3d velocity;

  // Accepts a {x, y, z, vx, vy, vz} vector
  State(std::vector<double> vec) {
    if (vec.size() != 6) {
      throw std::invalid_argument("Input vector must have 6 entries.");
    }
    position = {vec[0], vec[1], vec[2]};
    velocity = {vec[3], vec[4], vec[5]};
  };

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

//    States(std::vector<double> ephemTimes, std::vector<ale::Vec3d> positions, 
//           std::vector<ale::Vec3d> velocities = {ale::Vec3d(0.0, 0.0, 0.0)}, int refFrame=1);
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
    ale::State getState(double time, ale::interpolation interp=LINEAR) const;
    ale::Vec3d getPosition(double time, ale::interpolation interp=LINEAR) const;
    ale::Vec3d getVelocity(double time, ale::interpolation interp=LINEAR) const;
    
    // Get start and stop time of cache 
    double getStartTime(); 
    double getStopTime(); 
    
    // Cache reduction
    void minimizeCache(double tolerance=0.01);

  private:
    std::vector<int> HermiteIndices(double tolerance, std::vector <int> indexList, 
                                    double baseTime, double timeScale);
    std::vector<ale::State> m_states; //! Represent at states internally to keep pos, vel together
    std::vector<double> m_ephemTimes; //! Time in seconds
    int m_refFrame;  //! Naif IDs for reference frames 
    bool m_minimizedCache; 
  };
}

#endif

