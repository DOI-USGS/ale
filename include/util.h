#ifndef ALE_UTIL_H
#define ALE_UTIL_H

#include <string>
#include <nlohmann/json.hpp>

#include "isd.h"
#include "distortion.h"

namespace ale {
  using json = nlohmann::json;

  double getMinHeight(nlohmann::json isd);
  std::string getSensorModelName(json isd);
  std::string getImageId(json isd);
  std::string getSensorName(json isd);
  std::string getPlatformName(json isd);
  std::string getLogFile(nlohmann::json isd);
  int getTotalLines(json isd);
  int getTotalSamples(json isd);
  double getStartingTime(json isd);
  double getCenterTime(json isd);
  std::vector<std::vector<double>> getLineScanRate(json isd);
  int getSampleSumming(json isd);
  int getLineSumming(json isd);
  double getFocalLength(json isd);
  double getFocalLengthUncertainty(json isd);
  std::vector<double> getFocal2PixelLines(json isd);
  std::vector<double> getFocal2PixelSamples(json isd);
  double getDetectorCenterLine(json isd);
  double getDetectorCenterSample(json isd);
  double getDetectorStartingLine(json isd);
  double getDetectorStartingSample(json isd);
  double getMinHeight(json isd);
  double getMaxHeight(json isd);
  double getSemiMajorRadius(json isd);
  double getSemiMinorRadius(json isd);
  DistortionType getDistortionModel(json isd);
  std::vector<double> getDistortionCoeffs(json isd);
  std::vector<double> getSunPositions(json isd);
  std::vector<double> getSensorPositions(json isd);
  std::vector<double> getSensorVelocities(json isd);
  std::vector<double> getSensorOrientations(json isd);
}

#endif
