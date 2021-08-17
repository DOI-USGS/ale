#ifndef ALE_STATES_H
#define ALE_STATES_H

#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>


#include "ale/Vectors.h"
#include "ale/InterpUtils.h"

namespace ale {
  /** A state vector with position and velocity*/
  struct State {
    Vec3d position;
    Vec3d velocity;

    // Creates a state from a {x, y, z, vx, vy, vz} vector
    State(const std::vector<double>& vec) {
      if (vec.size() != 6) {
        throw std::invalid_argument("Input vector must have 6 entries.");
      }
      position = {vec[0], vec[1], vec[2]};
      velocity = {vec[3], vec[4], vec[5]};
    };

    // Creates a state with only a position
    State(Vec3d position) : position(position) {
      velocity = {std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::quiet_NaN()};
    };

    // Creates a state with a position and velocity
    State(Vec3d position, Vec3d velocity) : position(position), velocity(velocity) {};

    // Creates an un-initialized state
    State() {};

    // If the velocity for the state has been initialized
    bool hasVelocity() const {
      return !(std::isnan(velocity.x) || std::isnan(velocity.y) || std::isnan(velocity.z));
    }
  };

  class States {
    public:
      // Constructors
      /**
       * Creates an empty States object
       */
      States();

      /**
       * Creates a States object from a set of times and positions
       */
      States(const std::vector<double>& ephemTimes, const std::vector<Vec3d>& positions,
             int refFrame=1);

      /**
       * Creates a States object from a set of times and positions
       */
      States(const std::vector<double>& ephemTimes, const std::vector<std::vector<double>>& positions,
             int refFrame=1);

      /**
       * Creates a States object from a set of times, positions, and velocities
       */
      States(const std::vector<double>& ephemTimes, const std::vector<Vec3d>& positions,
             const std::vector<Vec3d>& velocities, int refFrame=1);

      /**
       * Creates a States object from a set of times, states
       */
      States(const std::vector<double>& ephemTimes, const std::vector<State>& states,
             int refFrame=1);

      ~States();

      // Getters
      std::vector<State> getStates() const; //!< Returns state vectors (6-element positions&velocities)
      std::vector<Vec3d> getPositions() const; //!< Returns the current positions
      std::vector<Vec3d> getVelocities() const; //!< Returns the current velocities
      std::vector<double> getTimes() const; //!< Returns the current times
      int getReferenceFrame() const; //!< Returns reference frame as NAIF ID
      bool hasVelocity() const; //!< Returns true if any velocities have been provided

      /**
       * Returns a single state by interpolating state.
       * If the Cache has been minimized, a cubic hermite is used to interpolate the
       * position and velocity over the reduced cache.
       * If not, a standard lagrange interpolation will be done.
       *
       * @param time Time to get a value at
       * @param interp Interpolation type to use. Will be ignored if cache is minimized.
       *
       * @return The interpolated state
       */
      State getState(double time, PositionInterpolation interp=LINEAR) const;

      /** Gets a position at a single time. Operates the same way as getState() **/
      Vec3d getPosition(double time, PositionInterpolation interp=LINEAR) const;

      /** Gets a velocity at a single time. Operates the same way as getState() **/
      Vec3d getVelocity(double time, PositionInterpolation interp=LINEAR) const;

      /** Returns the first ephemeris time **/
      double getStartTime();

      /** Returns the last ephemeris time **/
      double getStopTime();

      /**
       * Perform a cache reduction. After running this, getStates(), getPositions(),
       * and getVelocities() will return vectors of reduced size, and getState(),
       * getPosition(), and getVelocity() will
       * returns values interpolated over the reduced cache using a cubic hermite spline
       *
       * Adapted from Isis::SpicePosition::reduceCache().
       *
       * @param tolerance Maximum error between hermite approximation and original value.
       *
       * @return A new set of states that has been downsized.
       */
      States minimizeCache(double tolerance=0.01);

    private:

      /**
       * Calculates the points (indicies) which need to be kept for the hermite spline to
       * interpolate between to mantain a maximum error of tolerance.
       *
       * Adapted from Isis::SpicePosition::HermiteIndices.
       *
       * @param tolerance Maximum error between hermite approximation and original value.
       * @param indexList The list of indicies that need to be kept.
       * @param baseTime Scaled base time for fit
       * @param timeScale Time scale for fit.
       *
       * @return The indices that should be kept to downsize the set of States
       */
      std::vector<int> hermiteIndices(double tolerance, std::vector <int> indexList,
                                      double baseTime, double timeScale);
      std::vector<State> m_states; //!< The internal states cache
      std::vector<double> m_ephemTimes; //!< The times for the states cache
      int m_refFrame;  //!< Naif ID for the reference frame the states are in
    };
}

#endif
