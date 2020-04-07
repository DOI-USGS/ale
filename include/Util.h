#ifndef ALE_UTIL_H
#define ALE_UTIL_H

#include <string>
#include <nlohmann/json.hpp>

#include "InterpUtils.h"
#include "Distortion.h"
#include "States.h"
#include "Orientations.h"
#include "Vectors.h"

namespace ale {
  using json = nlohmann::json;

  template<typename T>
  std::vector<T> getJsonArray(json obj) {
    std::vector<T> positions;
    try {
      for (auto &location : obj) {
        positions.push_back(location.get<T>());
      }
    } catch (...) {
      throw std::runtime_error("Could not parse the json array.");
    }
    return positions;
  }


  interpolation getInterpolationMethod(json isd);
  double getMinHeight(nlohmann::json isd);
  std::string getSensorModelName(json isd);
  std::string getImageId(json isd);
  std::string getSensorName(json isd);
  std::string getPlatformName(json isd);
  std::string getLogFile(nlohmann::json isd);
  std::string getIsisCameraVersion(json isd);
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
  
  std::vector<double> getJsonDoubleArray(json obj);
  std::vector<Vec3d> getJsonVec3dArray(json obj);
  std::vector<Rotation> getJsonQuatArray(json obj);

  States getInstrumentPosition(json isd);
  States getSunPosition(json isd);

  Orientations getBodyRotation(json isd);
  Orientations getInstrumentPointing(json isd);
}

#endif
