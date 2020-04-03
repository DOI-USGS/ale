#ifndef ALE_UTIL_H
#define ALE_UTIL_H

#include <string>
#include <nlohmann/json.hpp>

#include "Isd.h"
#include "Distortion.h"

namespace ale {

  /** A 3D cartesian vector */
  struct Vec3d {
    double x;
    double y;
    double z;

    // Accepts an {x,y,z} vector
    Vec3d(const std::vector<double>& vec) {
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

  double getMinHeight(nlohmann::json isd);
  std::string getSensorModelName(nlohmann::json isd);
  std::string getImageId(nlohmann::json isd);
  std::string getSensorName(nlohmann::json isd);
  std::string getPlatformName(nlohmann::json isd);
  std::string getLogFile(nlohmann::json isd);
  int getTotalLines(nlohmann::json isd);
  int getTotalSamples(nlohmann::json isd);
  double getStartingTime(nlohmann::json isd);
  double getCenterTime(nlohmann::json isd);
  std::vector<std::vector<double>> getLineScanRate(nlohmann::json isd);
  int getSampleSumming(nlohmann::json isd);
  int getLineSumming(nlohmann::json isd);
  double getFocalLength(nlohmann::json isd);
  double getFocalLengthUncertainty(nlohmann::json isd);
  std::vector<double> getFocal2PixelLines(nlohmann::json isd);
  std::vector<double> getFocal2PixelSamples(nlohmann::json isd);
  double getDetectorCenterLine(nlohmann::json isd);
  double getDetectorCenterSample(nlohmann::json isd);
  double getDetectorStartingLine(nlohmann::json isd);
  double getDetectorStartingSample(nlohmann::json isd);
  double getMinHeight(nlohmann::json isd);
  double getMaxHeight(nlohmann::json isd);
  double getSemiMajorRadius(nlohmann::json isd);
  double getSemiMinorRadius(nlohmann::json isd);
  DistortionType getDistortionModel(nlohmann::json isd);
  std::vector<double> getDistortionCoeffs(nlohmann::json isd);
  std::vector<double> getSunPositions(nlohmann::json isd);
  std::vector<double> getSensorPositions(nlohmann::json isd);
  std::vector<double> getSensorVelocities(nlohmann::json isd);
  std::vector<double> getSensorOrientations(nlohmann::json isd);
}

#endif
