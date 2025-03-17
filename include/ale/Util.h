#ifndef ALE_UTIL_H
#define ALE_UTIL_H

#include <highfive/highfive.hpp>
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

  template<typename T>
  std::vector<std::vector<T>> getMultiSpectralArray(HighFive::DataSet ds){
    std::vector<std::vector<T>> data;
    try {
      ds.read(data);
    } catch (...) {
      throw std::runtime_error("Could not parse the hdf5 dataset");
    }
    return data;
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


  std::vector<std::string> getSensorModelName(HighFive::File hdf5);
  std::vector<std::string> getImageId(HighFive::File hdf5);
  std::vector<std::string> getPlatformName(HighFive::File hdf5);
  std::vector<std::string> getSensorName(HighFive::File hdf5);
  std::vector<unsigned int> getTotalLines(HighFive::File hdf5);
  std::vector<unsigned int> getTotalSamples(HighFive::File hdf5);
  std::vector<double> getStartingTime(HighFive::File hdf5);
  std::vector<double> getCenterTime(HighFive::File hdf5);
  std::vector<std::vector<std::vector<double>>> getLineScanRate(HighFive::File hdf5);
  std::vector<double> getSampleSumming(HighFive::File hdf5);
  std::vector<double> getLineSumming(HighFive::File hdf5);
  std::vector<double> getFocalLength(HighFive::File hdf5);
  std::vector<double> getFocalLengthUncertainty(HighFive::File hdf5);
  std::vector<std::vector<double>> getFocal2PixelLines(HighFive::File hdf5);
  std::vector<std::vector<double>> getFocal2PixelSamples(HighFive::File hdf5);
  std::vector<double> getDetectorCenterLine(HighFive::File hdf5);
  std::vector<double> getDetectorCenterSample(HighFive::File hdf5);
  std::vector<double> getDetectorStartingLine(HighFive::File hdf5);
  std::vector<double> getDetectorStartingSample(HighFive::File hdf5);
  std::vector<double> getMinHeight(HighFive::File hdf5);
  std::vector<double> getMaxHeight(HighFive::File hdf5);
  std::vector<double> getSemiMajorRadius(HighFive::File hdf5);
  std::vector<double> getSemiMinorRadius(HighFive::File hdf5);
  std::vector<DistortionType> getDistortionModel(HighFive::File hdf5);
  std::vector<std::vector<double>> getDistortionCoeffs(HighFive::File hdf5);
  std::vector<PositionInterpolation> getInterpolationMethod(HighFive::File hdf5);
  std::vector<Orientations> getBodyRotation(HighFive::File hdf5);
  std::vector<std::vector<Vec3d>> getVec3dArray(HighFive::DataSet ds);
  std::vector<std::vector<Rotation>> getQuatArray(HighFive::DataSet ds);

  std::vector<States> getInstrumentPosition(HighFive::File hdf5);
  std::vector<States> getSunPosition(HighFive::File hdf5);
  std::vector<Orientations> getInstrumentPointing(HighFive::File hdf5);
}

#endif
