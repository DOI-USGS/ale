#include "ale/States.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <float.h>

namespace ale {

  // Empty constructor
  States::States() : m_refFrame(0) {
    m_states = {};
    m_ephemTimes = {};
  }


  States::States(const std::vector<double>& ephemTimes, const std::vector<Vec3d>& positions,
                 int refFrame) :
    m_ephemTimes(ephemTimes), m_refFrame(refFrame) {
    // Construct State vector from position and velocity vectors
    if (positions.size() != ephemTimes.size()) {
      throw std::invalid_argument("Length of times must match number of positions");
    }

    for (Vec3d position : positions) {
      m_states.push_back(State(position));
    }
  }

  States::States(const std::vector<double>& ephemTimes, const std::vector<std::vector<double>>& positions,
                 int refFrame) :
    m_ephemTimes(ephemTimes), m_refFrame(refFrame) {

    // Construct State vector from position and velocity vectors
    if (positions.size() != ephemTimes.size()) {
      throw std::invalid_argument("Length of times must match number of positions");
    }

    for (Vec3d position : positions) {
      m_states.push_back(State(position));
    }
  }

  States::States(const std::vector<double>& ephemTimes, const std::vector<Vec3d>& positions,
                 const std::vector<Vec3d>& velocities, int refFrame) :
    m_ephemTimes(ephemTimes), m_refFrame(refFrame) {

    if ((positions.size() != ephemTimes.size())||(ephemTimes.size() != velocities.size())) {
      throw std::invalid_argument("Length of times must match number of positions and velocities.");
    }

    for (int i=0; i < positions.size() ;i++) {
      m_states.push_back(State(positions[i], velocities[i]));
    }
  }


  States::States(const std::vector<double>& ephemTimes, const std::vector<State>& states,
                 int refFrame) :
  m_ephemTimes(ephemTimes), m_states(states), m_refFrame(refFrame) {
    if (states.size() != ephemTimes.size()) {
      throw std::invalid_argument("Length of times must match number of states.");
    }
  }

  // Default Destructor
  States::~States() {}

  // Getters
  std::vector<State> States::getStates() const {
    return m_states;
  }

  std::vector<Vec3d> States::getPositions() const {
    // extract positions from state vector
    std::vector<Vec3d> positions;

    for(State state : m_states) {
        positions.push_back(state.position);
    }
    return positions;
  }


  std::vector<Vec3d> States::getVelocities() const {
    // extract velocities from state vector
    std::vector<Vec3d> velocities;

    for(State state : m_states) {
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
    bool allVelocity = std::all_of(m_states.begin(), m_states.end(), [](State vec)
                                   { return vec.hasVelocity(); });
    return allVelocity;
  }


  State States::getState(double time, PositionInterpolation interp) const {


    // If time is in times, don't need to interpolate!
    auto candidate_time = std::lower_bound(m_ephemTimes.begin(), m_ephemTimes.end(), time);

    if ( (candidate_time != m_ephemTimes.end()) && (*candidate_time == time) ) {
      int index = std::distance(m_ephemTimes.begin(), candidate_time);
      return m_states[index];
    }
  
    if (m_ephemTimes.size() > 1) {
      int lowerBound = interpolationIndex(m_ephemTimes, time); 
      // try to copy the surrounding 8 points as that's the most possibly needed
      int interpStart = std::max(0, lowerBound - 3);
      int interpStop = std::min(lowerBound + 4, (int) m_ephemTimes.size() - 1);

      State state;
      std::vector<double> xs, ys, zs, vxs, vys, vzs, interpTimes;

      for (int i = interpStart; i <= interpStop; i++) {
        state = m_states[i];
        interpTimes.push_back(m_ephemTimes[i]);
        xs.push_back(state.position.x);
        ys.push_back(state.position.y);
        zs.push_back(state.position.z);
        vxs.push_back(state.velocity.x);
        vys.push_back(state.velocity.y);
        vzs.push_back(state.velocity.z);
      }

      Vec3d position, velocity;
      if ( interp == LINEAR || (interp == SPLINE && !hasVelocity())) {
        position = {interpolate(xs,  interpTimes, time, interp, 0),
                    interpolate(ys,  interpTimes, time, interp, 0),
                    interpolate(zs,  interpTimes, time, interp, 0)};

        velocity = {interpolate(xs, interpTimes, time, interp, 1),
                    interpolate(ys, interpTimes, time, interp, 1),
                    interpolate(zs, interpTimes, time, interp, 1)};
      }
      else if (interp == SPLINE && hasVelocity()){
        // Do hermite spline if velocities are available
        double baseTime = (interpTimes.front() + interpTimes.back()) / 2;
        std::vector<double> scaledEphemTimes;
        for(unsigned int i = 0; i < interpTimes.size(); i++) {
          scaledEphemTimes.push_back(interpTimes[i] - baseTime);
        }
        double sTime = time - baseTime;
        position.x = evaluateCubicHermite(sTime, vxs, scaledEphemTimes, xs);
        position.y = evaluateCubicHermite(sTime, vys, scaledEphemTimes, ys);
        position.z = evaluateCubicHermite(sTime, vzs, scaledEphemTimes, zs);

        velocity.x = evaluateCubicHermiteFirstDeriv(sTime, vxs, scaledEphemTimes, xs);
        velocity.y = evaluateCubicHermiteFirstDeriv(sTime, vys, scaledEphemTimes, ys);
        velocity.z = evaluateCubicHermiteFirstDeriv(sTime, vzs, scaledEphemTimes, zs);
      }
      return State(position, velocity);
    }
    else if (hasVelocity()) {
      // Here we have: 1 state (1 time, 1 position, 1 velocity)
      // x_f = x_i + v * (t_f - t-i)
      Vec3d position = m_states[0].position + m_states[0].velocity*(time - m_ephemTimes[0]);
      Vec3d velocity = m_states[0].velocity;
      return State(position, velocity);
    }
    else { // Here we have: only 1 time and 1 state, so just return the only state.
      return m_states[0];
    }
  }


  Vec3d States::getPosition(double time, PositionInterpolation interp) const {
    State interpState = getState(time, interp);
    return interpState.position;
  }


  Vec3d States::getVelocity(double time, PositionInterpolation interp) const {
    State interpState = getState(time, interp);
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
    std::vector<State> tempStates;
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
        xerror = fabs(evaluateCubicHermite(sTime, vx, scaledEphemTimes, x) - m_states[line].position.x);
        yerror = fabs(evaluateCubicHermite(sTime, vy, scaledEphemTimes, y) - m_states[line].position.y);
        zerror = fabs(evaluateCubicHermite(sTime, vz, scaledEphemTimes, z) - m_states[line].position.z);

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
