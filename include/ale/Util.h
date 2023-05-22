#ifndef ALE_UTIL_H
#define ALE_UTIL_H

#include <string>
#include <nlohmann/json.hpp>

#include "ale/InterpUtils.h"
#include "ale/Distortion.h"
#include "ale/States.h"
#include "ale/Orientations.h"
#include "ale/Vectors.h"

namespace ale {

  template<typename T>
  std::vector<T> getJsonArray(nlohmann::json obj) {
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


  PositionInterpolation getInterpolationMethod(nlohmann::json isd);
  double getMinHeight(nlohmann::json isd);
  std::string getSensorModelName(nlohmann::json isd);
  std::string getImageId(nlohmann::json isd);
  std::string getSensorName(nlohmann::json isd);
  std::string getPlatformName(nlohmann::json isd);
  std::string getLogFile(nlohmann::json isd);
  std::string getIsisCameraVersion(nlohmann::json isd);
  std::string getProjection(nlohmann::json isd);
  
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
  std::vector<double> getGeoTransform(nlohmann::json isd);
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

  std::vector<double> getJsonDoubleArray(nlohmann::json obj);
  std::vector<Vec3d> getJsonVec3dArray(nlohmann::json obj);
  std::vector<Rotation> getJsonQuatArray(nlohmann::json obj);

  States getInstrumentPosition(nlohmann::json isd);
  States getSunPosition(nlohmann::json isd);

  Orientations getBodyRotation(nlohmann::json isd);
  Orientations getInstrumentPointing(nlohmann::json isd);
}

#endif
