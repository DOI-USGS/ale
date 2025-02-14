#include <stdexcept>
#include <algorithm>
#include <iostream>

#include "ale/Util.h"

using json = nlohmann::json;

namespace ale {

bool iequals(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(),
                      b.begin(),
                      [](char a, char b) {
                          return tolower(a) == tolower(b);
                      });
}


std::string getSensorModelName(json isd) {
  std::string name = "";
  try {
    name = isd.at("name_model");
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor model name.");
  }
  return name;
}

std::string getImageId(json isd) {
  std::string id = "";
  try {
    id = isd.at("image_identifier");
  } catch (...) {
    throw std::runtime_error("Could not parse the image identifier.");
  }
  return id;
}


std::vector<double> getGeoTransform(json isd) {
  std::vector<double> transform = {};
  try {
    transform = isd.at("geotransform").get<std::vector<double>>();
  } catch (std::exception &e) {
    std::string originalError = e.what();
    std::string msg = "Could not parse the geo_transform. ERROR: \n" + originalError;
    throw std::runtime_error(msg);
  }
  return transform;
}


std::string getProjection(json isd) {
  std::string projection_string = "";
  try {
    projection_string = isd.at("projection");
  } catch (...) {
    throw std::runtime_error("Could not parse the projection string.");
  }
  return projection_string;
}


std::string getSensorName(json isd) {
  std::string name = "";
  try {
    name = isd.at("name_sensor");
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor name.");
  }
  return name;
}

std::string getIsisCameraVersion(json isd) {
  std::string name = "";
  try {
    name = isd.at("IsisCameraVersion");
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor name.");
  }
  return name;
}


std::string getPlatformName(json isd) {
  std::string name = "";
  try {
    name = isd.at("name_platform");
  } catch (...) {
    throw std::runtime_error("Could not parse the platform name.");
  }
  return name;
}

std::string getLogFile(json isd) {
  std::string file = "";
  try {
    file = isd.at("log_file");
  } catch (...) {
    throw std::runtime_error("Could not parse the log filename.");
  }
  return file;
}

int getTotalLines(json isd) {
  int lines = 0;
  try {
    lines = isd.at("image_lines");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the number of lines in the image.");
  }
  return lines;
}

int getTotalSamples(json isd) {
  int samples = 0;
  try {
    samples = isd.at("image_samples");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the number of samples in the image.");
  }
  return samples;
}

double getStartingTime(json isd) {
  double time = 0.0;
  try {
    time = isd.at("starting_ephemeris_time");
  } catch (...) {
    throw std::runtime_error("Could not parse the image start time.");
  }
  return time;
}

double getCenterTime(json isd) {
  double time = 0.0;
  try {
    time = isd.at("center_ephemeris_time");
  } catch (...) {
    throw std::runtime_error("Could not parse the center image time.");
  }
  return time;
}

PositionInterpolation getInterpolationMethod(json isd) {
  std::string interpMethod = "linear";
  try {
    interpMethod = isd.at("interpolation_method");

    if (iequals(interpMethod, "linear")) {
      return PositionInterpolation::LINEAR;
    }
    else if (iequals(interpMethod, "spline")){
      return PositionInterpolation::SPLINE;
    }
    else if (iequals(interpMethod, "lagrange")) {
      return PositionInterpolation::LAGRANGE;
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the interpolation method.");
  }

  return PositionInterpolation::LINEAR;
}

std::vector<std::vector<double>> getLineScanRate(json isd) {
  std::vector<std::vector<double>> lines;
  try {
    for (auto &scanRate : isd.at("line_scan_rate")) {
      lines.push_back(scanRate.get<std::vector<double>>());
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the line scan rate from the isd.");
  }
  return lines;
}


int getSampleSumming(json isd) {
  int summing = 0;
  try {
    summing = isd.at("detector_sample_summing");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the sample direction detector pixel summing.");
  }
  return summing;
}

int getLineSumming(json isd) {
  int summing = 0;
  try {
    summing = isd.at("detector_line_summing");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the line direction detector pixel summing.");
  }
  return summing;
}

double getFocalLength(json isd) {
  double length = 0.0;
  try {
    length = isd.at("focal_length_model").at("focal_length");
  } catch (...) {
    throw std::runtime_error("Could not parse the focal length.");
  }
  return length;
}

double getFocalLengthUncertainty(json isd) {
  double uncertainty = 1.0;
  try {
    uncertainty = isd.at("focal_length_model").value("focal_uncertainty", uncertainty);
  } catch (...) {
    throw std::runtime_error("Could not parse the focal length uncertainty.");
  }
  return uncertainty;
}

std::vector<double> getFocal2PixelLines(json isd) {
  std::vector<double> transformation;
  try {
    transformation = isd.at("focal2pixel_lines").get<std::vector<double>>();
  } catch (...) {
    throw std::runtime_error("Could not parse the focal plane coordinate to "
                             "detector lines transformation.");
  }
  return transformation;
}

std::vector<double> getFocal2PixelSamples(json isd) {
  std::vector<double> transformation;
  try {
    transformation = isd.at("focal2pixel_samples").get<std::vector<double>>();
  } catch (...) {
    throw std::runtime_error("Could not parse the focal plane coordinate to "
                             "detector samples transformation.");
  }
  return transformation;
}

double getDetectorCenterLine(json isd) {
  double line;
  try {
    line = isd.at("detector_center").at("line");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector center line.");
  }
  return line;
}

double getDetectorCenterSample(json isd) {
  double sample;
  try {
    sample = isd.at("detector_center").at("sample");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector center sample.");
  }
  return sample;
}

double getDetectorStartingLine(json isd) {
  double line;
  try {
    line = isd.at("starting_detector_line");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector starting line.");
  }
  return line;
}

double getDetectorStartingSample(json isd) {
  double sample;
  try {
    sample = isd.at("starting_detector_sample");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector starting sample.");
  }
  return sample;
}

double getMinHeight(json isd) {
  double height = 0.0;
  try {
    json referenceHeight = isd.at("reference_height");
    json minHeight = referenceHeight.at("minheight");
    height = minHeight.get<double>();
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the minimum height above the reference ellipsoid.");
  }
  return height;
}

double getMaxHeight(json isd) {
  double height = 0.0;
  try {
    json referenceHeight = isd.at("reference_height");
    json maxHeight = referenceHeight.at("maxheight");

    height = maxHeight.get<double>();
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the maximum height above the reference ellipsoid.");
  }
  return height;
}

double getSemiMajorRadius(json isd) {
  double radius = 0.0;
  try {
    json radii = isd.at("radii");
    json semiMajor = radii.at("semimajor");
    radius = semiMajor.get<double>();

  } catch (...) {
    throw std::runtime_error(
        "Could not parse the reference ellipsoid semimajor radius.");
  }
  return radius;
}

double getSemiMinorRadius(json isd) {
  double radius = 0.0;
  try {
    json radii = isd.at("radii");
    json semiMinor = radii.at("semiminor");
    radius = semiMinor.get<double>();

  } catch (...) {
    throw std::runtime_error(
        "Could not parse the reference ellipsoid semiminor radius.");
  }
  return radius;
}

// Converts the distortion model name from the ISD (string) to the enumeration
// type. Defaults to transverse
DistortionType getDistortionModel(json isd) {
  try {
    json distortion_subset = isd.at("optical_distortion");

    json::iterator it = distortion_subset.begin();

    std::string distortion = (std::string)it.key();

    if (distortion.compare("transverse") == 0) {
      return DistortionType::TRANSVERSE;
    } else if (distortion.compare("radial") == 0) {
      return DistortionType::RADIAL;
    } else if (distortion.compare("kaguyalism") == 0) {
      return DistortionType::KAGUYALISM;
    } else if (distortion.compare("dawnfc") == 0) {
      return DistortionType::DAWNFC;
    } else if (distortion.compare("lrolrocnac") == 0) {
      return DistortionType::LROLROCNAC;
    } else if (distortion.compare("cahvor") == 0) {
      return DistortionType::CAHVOR;
    } else if (distortion.compare("lunarorbiter") == 0) {
      return DistortionType::LUNARORBITER;
    } else if (distortion.compare("radtan") == 0) {
      return DistortionType::RADTAN;
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the distortion model.");
  }
  return DistortionType::TRANSVERSE;
}

std::vector<double> getDistortionCoeffs(json isd) {
  std::vector<double> coefficients;

  DistortionType distortion = getDistortionModel(isd);

  switch (distortion) {
    case DistortionType::TRANSVERSE: {
      try {
        std::vector<double> coefficientsX, coefficientsY;

        coefficientsX = isd.at("optical_distortion")
                            .at("transverse")
                            .at("x")
                            .get<std::vector<double>>();
        coefficientsX.resize(10, 0.0);

        coefficientsY = isd.at("optical_distortion")
                            .at("transverse")
                            .at("y")
                            .get<std::vector<double>>();
        coefficientsY.resize(10, 0.0);

        coefficients = coefficientsX;

        coefficients.insert(coefficients.end(), coefficientsY.begin(),
                            coefficientsY.end());
        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse a set of transverse distortion model coefficients.");
        coefficients = std::vector<double>(20, 0.0);
        coefficients[1] = 1.0;
        coefficients[12] = 1.0;
      }
    } break;
    case DistortionType::RADIAL: {
      try {
        coefficients = isd.at("optical_distortion")
                          .at("radial")
                          .at("coefficients")
                          .get<std::vector<double>>();

        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the radial distortion model coefficients.");
        coefficients = std::vector<double>(3, 0.0);
      }
    } break;
    case DistortionType::KAGUYALISM: {
      try {

        std::vector<double> coefficientsX = isd.at("optical_distortion")
                                                .at("kaguyalism")
                                                .at("x")
                                                .get<std::vector<double>>();
        std::vector<double> coefficientsY = isd.at("optical_distortion")
                                                .at("kaguyalism")
                                                .at("y")
                                                .get<std::vector<double>>();
        double boresightX = isd.at("optical_distortion")
                                .at("kaguyalism")
                                .at("boresight_x")
                                .get<double>();
        double boresightY = isd.at("optical_distortion")
                                .at("kaguyalism")
                                .at("boresight_y")
                                .get<double>();

        coefficientsX.resize(4, 0.0);
        coefficientsY.resize(4, 0.0);
        coefficientsX.insert(coefficientsX.begin(), boresightX);
        coefficientsY.insert(coefficientsY.begin(), boresightY);
        coefficientsX.insert(coefficientsX.end(), coefficientsY.begin(),
                            coefficientsY.end());

        return coefficientsX;
      } catch (...) {
        throw std::runtime_error("Could not parse a set of Kaguya LISM "
                                "distortion model coefficients.");
        coefficients = std::vector<double>(8, 0.0);
      }
    } break;
    case DistortionType::DAWNFC: {
      try {
        coefficients = isd.at("optical_distortion")
                          .at("dawnfc")
                          .at("coefficients")
                          .get<std::vector<double>>();

        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the dawn radial distortion model coefficients.");
        coefficients = std::vector<double>(1, 0.0);
      }
    } break;
    case DistortionType::LROLROCNAC: {
      try {
        coefficients = isd.at("optical_distortion")
                          .at("lrolrocnac")
                          .at("coefficients")
                          .get<std::vector<double>>();
        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the lrolrocnac distortion model coefficients.");
        coefficients = std::vector<double>(1, 0.0);
      }
    } break;
    case DistortionType::CAHVOR:
    {
      try
      {
        coefficients = isd.at("optical_distortion")
                          .at("cahvor")
                          .at("coefficients")
                          .get<std::vector<double>>();

        return coefficients;
      }
      catch (...)
      {
        throw std::runtime_error(
            "Could not parse the cahvor distortion model coefficients.");
        coefficients = std::vector<double>(5, 0.0);
      }
    } break;
    case DistortionType::LUNARORBITER:
    {
      try
      {
        double perspectiveX = isd.at("optical_distortion")
                                 .at("lunarorbiter")
                                 .at("perspective_x")
                                 .get<double>();
        double perspectiveY = isd.at("optical_distortion")
                                 .at("lunarorbiter")
                                 .at("perspective_y")
                                 .get<double>();
        double centerPointX = isd.at("optical_distortion")
                                 .at("lunarorbiter")
                                 .at("center_point_x")
                                 .get<double>();
        double centerPointY = isd.at("optical_distortion")
                                 .at("lunarorbiter")
                                 .at("center_point_y")
                                 .get<double>();

        coefficients = {perspectiveX, perspectiveY, centerPointX, centerPointY};
        return coefficients;
      }
      catch (...)
      {
        throw std::runtime_error(
          "Could not parse the Lunar Orbiter distortion model coefficients.");
        coefficients = std::vector<double>(4, 0.0);
      }
    } break;
    case DistortionType::RADTAN: {
      try {
        coefficients = isd.at("optical_distortion")
                          .at("radtan")
                          .at("coefficients")
                          .get<std::vector<double>>();

        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the radtan distortion model coefficients.");
        coefficients = std::vector<double>(5, 0.0);
      }
    } break;
  }
  throw std::runtime_error(
      "Could not parse the distortion model coefficients.");

  return coefficients;
}

std::vector<Vec3d> getJsonVec3dArray(json obj) {
  std::vector<Vec3d> positions;
  try {
    for (auto &location : obj) {
      Vec3d vec(location[0].get<double>(),location[1].get<double>(), location[2].get<double>() );
      positions.push_back(vec);
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the 3D vector array.");
  }
  return positions;
}


std::vector<Rotation> getJsonQuatArray(json obj) {
  std::vector<Rotation> quats;
  try {
    for (auto &location : obj) {
      Rotation vec(location[0].get<double>(),location[1].get<double>(), location[2].get<double>(), location[3].get<double>() );
      quats.push_back(vec);
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the quaternion json object.");
  }
  return quats;
}


States getInstrumentPosition(json isd) {
  try {
    json ipos = isd.at("instrument_position");
    std::vector<Vec3d> positions = getJsonVec3dArray(ipos.at("positions"));
    std::vector<double> times = getJsonArray<double>(ipos.at("ephemeris_times"));
    int refFrame = ipos.at("reference_frame").get<int>();

    bool hasVelocities = ipos.find("velocities") != ipos.end();

    if (hasVelocities) {
      std::vector<Vec3d> velocities = getJsonVec3dArray(ipos.at("velocities"));
      return States(times, positions, velocities, refFrame);
    }

    return States(times, positions, refFrame);
  } catch (...) {
    throw std::runtime_error("Could not parse the instrument position");
  }
}


States getSunPosition(json isd) {
  try {
    json spos = isd.at("sun_position");
    std::vector<Vec3d> positions = getJsonVec3dArray(spos.at("positions"));
    std::vector<double> times = getJsonArray<double>(spos.at("ephemeris_times"));
    int refFrame = spos.at("reference_frame").get<int>();
    bool hasVelocities = spos.find("velocities") != spos.end();

    if (hasVelocities) {
      std::vector<Vec3d> velocities = getJsonVec3dArray(spos.at("velocities"));
      return States(times, positions, velocities, refFrame);
    }

    return States(times, positions, refFrame);

  } catch (...) {
    throw std::runtime_error("Could not parse the sun position");
  }
}

Orientations getInstrumentPointing(json isd) {
  try {
    json pointing = isd.at("instrument_pointing");

    std::vector<Rotation> rotations = getJsonQuatArray(pointing.at("quaternions"));
    std::vector<double> times = getJsonArray<double>(pointing.at("ephemeris_times"));
    std::vector<Vec3d> velocities = getJsonVec3dArray(pointing.at("angular_velocities"));

    std::vector<int> constFrames;
    if (pointing.find("constant_frames") != pointing.end()){
      constFrames  = getJsonArray<int>(pointing.at("constant_frames"));
    }

    std::vector<int> timeDepFrames;
    if (pointing.find("time_dependent_frames") != pointing.end()){
      timeDepFrames = getJsonArray<int>(pointing.at("time_dependent_frames"));
    }

    std::vector<double> rotArray = {1,0,0,0,1,0,0,0,1};
    if (pointing.find("time_dependent_frames") != pointing.end()){
      rotArray = getJsonArray<double>(pointing.at("constant_rotation"));
    }

    Rotation constRot(rotArray);

    Orientations orientation(rotations, times, velocities, constRot, constFrames, timeDepFrames);

    return orientation;

  } catch (...) {
    throw std::runtime_error("Could not parse the instrument pointing");
  }
}

Orientations getBodyRotation(json isd) {
  try {
    json bodrot = isd.at("body_rotation");
    std::vector<Rotation> rotations = getJsonQuatArray(bodrot.at("quaternions"));
    std::vector<double> times = getJsonArray<double>(bodrot.at("ephemeris_times"));
    std::vector<Vec3d> velocities = getJsonVec3dArray(bodrot.at("angular_velocities"));

    std::vector<int> constFrames;
    if (bodrot.find("constant_frames") != bodrot.end()){
      constFrames  = getJsonArray<int>(bodrot.at("constant_frames"));
    }

    std::vector<int> timeDepFrames;
    if (bodrot.find("time_dependent_frames") != bodrot.end()){
      timeDepFrames = getJsonArray<int>(bodrot.at("time_dependent_frames"));
    }

    std::vector<double> rotArray = {1,0,0,0,1,0,0,0,1};
    if (bodrot.find("constant_rotation") != bodrot.end()){
      rotArray = getJsonArray<double>(bodrot.at("constant_rotation"));
    }

    Rotation constRot(rotArray);

    Orientations orientation(rotations, times, velocities, constRot, constFrames, timeDepFrames);
    return orientation;

  } catch (...) {
    throw std::runtime_error("Could not parse the body rotation");
  }
}


std::vector<std::string> getSensorModelName(HighFive::File hdf5) {
  std::vector<std::string> name;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("name_model");
    ds.read(name);
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor model name.");
  }
  return name;
}

std::vector<std::string> getImageId(HighFive::File hdf5) {
  std::vector<std::string> id;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("image_identifier");
    ds.read(id);
  } catch (...) {
    throw std::runtime_error("Could not parse the image identifier.");
  }
  return id;
}


std::vector<std::vector<double>> getGeoTransform(HighFive::File hdf5) {
  std::vector<std::vector<double>> transform;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("geotransform");
    ds.read(transform);
  } catch (std::exception &e) {
    std::string originalError = e.what();
    std::string msg = "Could not parse the geo_transform. ERROR: \n" + originalError;
    throw std::runtime_error(msg);
  }
  return transform;
}


std::vector<std::string> getProjection(HighFive::File hdf5) {
  std::vector<std::string> projection;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("projection");
    ds.read(projection);
  } catch (...) {
    throw std::runtime_error("Could not parse the projection string.");
  }
  return projection;
}


std::vector<std::string> getSensorName(HighFive::File hdf5) {
  std::vector<std::string> name;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("name_sensor");
    ds.read(name);
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor name.");
  }
  return name;
}

std::vector<std::string> getIsisCameraVersion(HighFive::File hdf5) {
  std::vector<std::string> version;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("IsisCameraVersion");
    ds.read(version);
  } catch (...) {
    throw std::runtime_error("Could not parse the IsisCameraVersion.");
  }
  return version;
}


std::vector<std::string> getPlatformName(HighFive::File hdf5) {
  std::vector<std::string> platformName;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("name_platform");
    ds.read(platformName);
  } catch (...) {
    throw std::runtime_error("Could not parse the platform name.");
  }
  return platformName;
}

std::vector<std::string> getLogFile(HighFive::File hdf5) {
  std::vector<std::string> logFile;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("log_file");
    ds.read(logFile);
  } catch (...) {
    throw std::runtime_error("Could not parse the log filename.");
  }
  return logFile;
}

std::vector<unsigned int> getTotalLines(HighFive::File hdf5) {
  std::vector<unsigned int> lines;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("image_lines");
    ds.read(lines);
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the number of lines in the image.");
  }
  return lines;
}

std::vector<unsigned int> getTotalSamples(HighFive::File hdf5) {
  std::vector<unsigned int> samples;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("image_samples");
    ds.read(samples);
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the number of samples in the image.");
  }
  return samples;
}

std::vector<double> getStartingTime(HighFive::File hdf5) {
  std::vector<double> time;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("starting_ephemeris_time");
    ds.read(time);
  } catch (...) {
    throw std::runtime_error("Could not parse the image start time.");
  }
  return time;
}

std::vector<double> getCenterTime(HighFive::File hdf5) {
  std::vector<double> time;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("center_ephemeris_time");
    ds.read(time);
  } catch (...) {
    throw std::runtime_error("Could not parse the center image time.");
  }
  return time;
}

std::vector<PositionInterpolation> getInterpolationMethod(HighFive::File hdf5) {
  std::vector<std::string> interpMethodStrings;
  std::vector<PositionInterpolation> interpMethod;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("interpolation_method");
    ds.read(interpMethodStrings);

    for (auto &it : interpMethodStrings){
      if (iequals(it, "linear")) {
        interpMethod.push_back(PositionInterpolation::LINEAR);
      }
      else if (iequals(it, "spline")){
        interpMethod.push_back(PositionInterpolation::SPLINE);
      }
      else if (iequals(it, "lagrange")) {
        interpMethod.push_back(PositionInterpolation::LAGRANGE);
      }
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the interpolation method.");
  }

  return interpMethod;
}

std::vector<std::vector<std::vector<double>>> getLineScanRate(HighFive::File hdf5) {
  std::vector<std::vector<std::vector<double>>> lines;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("line_scan_rate");
    ds.read(lines);
  } catch (...) {
    throw std::runtime_error("Could not parse the line scan rate from the isd.");
  }
  return lines;
}


std::vector<double> getSampleSumming(HighFive::File hdf5) {
  std::vector<double> summing;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("detector_sample_summing");
    ds.read(summing);
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the sample direction detector pixel summing.");
  }
  return summing;
}

std::vector<double> getLineSumming(HighFive::File hdf5) {
  std::vector<double> summing;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("detector_line_summing");
    ds.read(summing);
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the line direction detector pixel summing.");
  }
  return summing;
}

std::vector<double> getFocalLength(HighFive::File hdf5) {
  std::vector<double> focalLength;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("focal_length_model/focal_length");
    ds.read(focalLength);
  } catch (...) {
    throw std::runtime_error("Could not parse the focal length.");
  }
  return focalLength;
}

std::vector<double> getFocalLengthUncertainty(HighFive::File hdf5) {
  std::vector<double> focalUncertainty;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("focal_length_model/focal_uncertainty");
    ds.read(focalUncertainty);
  } catch (...) {
    throw std::runtime_error("Could not parse the focal length uncertainty.");
  }
  return focalUncertainty;
}

std::vector<std::vector<double>> getFocal2PixelLines(HighFive::File hdf5) {
  std::vector<std::vector<double>> transformation;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("focal2pixel_lines");
    ds.read(transformation);
  } catch (...) {
    throw std::runtime_error("Could not parse the focal plane coordinate to "
                             "detector lines transformation.");
  }
  return transformation;
}

std::vector<std::vector<double>> getFocal2PixelSamples(HighFive::File hdf5) {
  std::vector<std::vector<double>> transformation;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("focal2pixel_samples");
    ds.read(transformation);
  } catch (...) {
    throw std::runtime_error("Could not parse the focal plane coordinate to "
                             "detector samples transformation.");
  }
  return transformation;
}

std::vector<double> getDetectorCenterLine(HighFive::File hdf5) {
  std::vector<double> line;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("detector_center/line");
    ds.read(line);
  } catch (...) {
    throw std::runtime_error("Could not parse the detector center line.");
  }
  return line;
}

std::vector<double> getDetectorCenterSample(HighFive::File hdf5) {
  std::vector<double> sample;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("detector_center/sample");
    ds.read(sample);
  } catch (...) {
    throw std::runtime_error("Could not parse the detector center sample.");
  }
  return sample;
}

std::vector<double> getDetectorStartingLine(HighFive::File hdf5) {
  std::vector<double> line;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("starting_detector_line");
    ds.read(line);
  } catch (...) {
    throw std::runtime_error("Could not parse the detector starting line.");
  }
  return line;
}

std::vector<double> getDetectorStartingSample(HighFive::File hdf5) {
  std::vector<double> sample;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("starting_detector_sample");
    ds.read(sample);
  } catch (...) {
    throw std::runtime_error("Could not parse the detector starting sample.");
  }
  return sample;
}

std::vector<double> getMinHeight(HighFive::File hdf5) {
  std::vector<double> height;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("reference_height/minheight");
    ds.read(height);
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the minimum height above the reference ellipsoid.");
  }
  return height;
}

std::vector<double> getMaxHeight(HighFive::File hdf5) {
  std::vector<double> height;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("reference_height/maxheight");
    ds.read(height);
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the maximum height above the reference ellipsoid.");
  }
  return height;
}

std::vector<double> getSemiMajorRadius(HighFive::File hdf5) {
  std::vector<double> radius;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("radii/semimajor");
    ds.read(radius);
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the reference ellipsoid semimajor radius.");
  }
  return radius;
}

std::vector<double> getSemiMinorRadius(HighFive::File hdf5) {
  std::vector<double> radius;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("radii/semiminor");
    ds.read(radius);

  } catch (...) {
    throw std::runtime_error(
        "Could not parse the reference ellipsoid semiminor radius.");
  }
  return radius;
}

// Converts the distortion model name from the ISD (string) to the enumeration
// type.
std::vector<DistortionType> getDistortionModel(HighFive::File hdf5) {
  std::vector<DistortionType> distortions;
  try {
    HighFive::DataSet ds = hdf5.getDataSet("optical_distortion");
    std::vector<std::string> distortionStrings;
    ds.read(distortionStrings);

    for (auto &dist : distortionStrings){
      if (dist.compare("transverse") == 0) {
        distortions.push_back(DistortionType::TRANSVERSE);
      } else if (dist.compare("radial") == 0) {
        distortions.push_back(DistortionType::RADIAL);
      } else if (dist.compare("kaguyalism") == 0) {
        distortions.push_back(DistortionType::KAGUYALISM);
      } else if (dist.compare("dawnfc") == 0) {
        distortions.push_back(DistortionType::DAWNFC);
      } else if (dist.compare("lrolrocnac") == 0) {
        distortions.push_back(DistortionType::LROLROCNAC);
      } else if (dist.compare("cahvor") == 0) {
        distortions.push_back(DistortionType::CAHVOR);
      } else if (dist.compare("lunarorbiter") == 0) {
        distortions.push_back(DistortionType::LUNARORBITER);
      } else if (dist.compare("radtan") == 0) {
        distortions.push_back(DistortionType::RADTAN);
      }
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the distortion model.");
  }
  return distortions;
}

std::vector<std::vector<double>> getDistortionCoeffs(HighFive::File hdf5) {
  std::vector<std::vector<double>> coefficients;

  std::vector<DistortionType> distortion = getDistortionModel(hdf5);

  //TODO handle different distortion types per band...?
  switch (distortion[0]) {
    case DistortionType::TRANSVERSE: {
      try {
        std::vector<double> coefficientsX, coefficientsY;

        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/transverse/x");
        ds.read(coefficientsX);

        ds = hdf5.getDataSet("optical_distortion/transverse/y");
        ds.read(coefficientsY);

        //coefficients = coefficientsX;

        for (size_t i = 0; i < coefficientsX.size(); i++) {
            // Combine the corresponding vectors from vec1 and vec2
            std::vector<double> combinedCoeff;
            combinedCoeff.push_back(coefficientsX[i]);
            combinedCoeff.push_back(coefficientsY[i]);
            //combinedCoeff.insert(combinedCoeff.end(), coefficientsY[i].begin(), coefficientsY[i].end());
            
            // Add the combined vector to the result
            coefficients.push_back(combinedCoeff);
        }

        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse a set of transverse distortion model coefficients.");
        /* TODO ???
        coefficients = std::vector<double>(20, 0.0);
        coefficients[1] = 1.0;
        coefficients[12] = 1.0;
        */
      }
    } break;
    case DistortionType::RADIAL: {
      try {
        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/radial/coefficients");
        ds.read(coefficients);
        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the radial distortion model coefficients.");
        /* TODO ???
        coefficients = std::vector<double>(3, 0.0);
        */
      }
    } break;
    case DistortionType::KAGUYALISM: {
      try {
        std::vector<std::vector<double>> coefficientsX, coefficientsY;
        std::vector<double> boresightX, boresightY;
        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/kaguyalism/x");
        ds.read(coefficientsX);
        ds = hdf5.getDataSet("optical_distortion/kaguyalism/y");
        ds.read(coefficientsY);

        ds = hdf5.getDataSet("optical_distortion/kaguyalism/boresight_x");
        ds.read(boresightX);
        ds = hdf5.getDataSet("optical_distortion/kaguyalism/boresight_y");
        ds.read(boresightY);

        for (int i = 0 ; i < coefficientsX.size(); i++){
          coefficientsX[i].insert(coefficientsX[i].begin(), boresightX[i]);
          coefficientsY[i].insert(coefficientsY[i].begin(), boresightY[i]);
          coefficientsX[i].insert(coefficientsX[i].end(), coefficientsY[i].begin(),
                              coefficientsY[i].end());
        }

        return coefficientsX;
      } catch (...) {
        throw std::runtime_error("Could not parse a set of Kaguya LISM "
                                "distortion model coefficients.");
        /* TODO
        coefficients = std::vector<double>(8, 0.0);
        */
      }
    } break;
    case DistortionType::DAWNFC: {
      try {
        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/dawnfc/coefficients");
        ds.read(coefficients);
        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the dawn radial distortion model coefficients.");
        /* TODO
        coefficients = std::vector<double>(1, 0.0);
        */
      }
    } break;
    case DistortionType::LROLROCNAC: {
      try {
        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/lrolrocnac/coefficients");
        ds.read(coefficients);
        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the lrolrocnac distortion model coefficients.");
        /* TODO
        coefficients = std::vector<double>(1, 0.0);
        */
      }
    } break;
    case DistortionType::CAHVOR:
    {
      try
      {
        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/cahvor/coefficients");
        ds.read(coefficients);
        return coefficients;
      }
      catch (...)
      {
        throw std::runtime_error(
            "Could not parse the cahvor distortion model coefficients.");
        /* TODO
        coefficients = std::vector<double>(5, 0.0);
        */
      }
    } break;
    case DistortionType::LUNARORBITER:
    {
      try
      {
        std::vector<double> perspectiveX, perspectiveY, centerPointX, centerPointY;
        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/lunarorbiter/perspective_x");
        ds.read(perspectiveX);
        ds = hdf5.getDataSet("optical_distortion/lunarorbiter/perspective_y");
        ds.read(perspectiveY);
        ds = hdf5.getDataSet("optical_distortion/lunarorbiter/center_point_x");
        ds.read(centerPointX);
        ds = hdf5.getDataSet("optical_distortion/lunarorbiter/center_point_y");
        ds.read(centerPointY);

        for (int i = 0 ; i < perspectiveX.size(); i++){
          coefficients.push_back({perspectiveX[i],perspectiveY[i],centerPointX[i],centerPointY[i]});
        }

        return coefficients;
      }
      catch (...)
      {
        throw std::runtime_error(
          "Could not parse the Lunar Orbiter distortion model coefficients.");
        /* TODO
        coefficients = std::vector<double>(4, 0.0);
        */
      }
    } break;
    case DistortionType::RADTAN: {
      try {
        HighFive::DataSet ds = hdf5.getDataSet("optical_distortion/radtan/coefficients");
        ds.read(coefficients);
        return coefficients;
      } catch (...) {
        throw std::runtime_error(
            "Could not parse the radtan distortion model coefficients.");
        /* TODO
        coefficients = std::vector<double>(5, 0.0);
        */
      }
    } break;
  }
  throw std::runtime_error(
      "Could not parse the distortion model coefficients.");

  return coefficients;
}

std::vector<std::vector<Vec3d>> getVec3dArray(HighFive::DataSet ds) {
  std::vector<std::vector<Vec3d>> positions;
  // yikes.
  // Each band in the multispectral isd has a 2d vector [band][position][x,y,z]
  std::vector<std::vector<std::array<double,3>>> data;
  try {
    ds.read(data);
    for(auto &band : data){
      // Create a vector of 3d coords for this band
      std::vector<Vec3d> bandVec;
      for (auto &item: band) {
        // Push all the 3d coords in this band into one vector
        Vec3d vec(item[0],item[1],item[2]);
        bandVec.push_back(vec);
      }
      // Add the vector to the multispectral vector
      positions.push_back(bandVec);
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the 3D vector array.");
  }
  return positions;
}


std::vector<std::vector<Rotation>> getQuatArray(HighFive::DataSet ds) {
  std::vector<std::vector<Rotation>> quats;
  std::vector<std::vector<std::array<double, 4>>> data;
  try {
    ds.read(data);
    for (auto &band : data){
      std::vector<Rotation> bandQuats;
      for (auto &row : band) {
        Rotation vec(row[0], row[1], row[2], row[3]);
        bandQuats.push_back(vec);
      }
      quats.push_back(bandQuats);
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the quaternion hdf5 object.");
  }
  return quats;
}


std::vector<States> getInstrumentPosition(HighFive::File hdf5) {
  try {
    std::vector<States> statesVec;
    HighFive::DataSet ds = hdf5.getDataSet("instrument_position/positions");
    std::vector<std::vector<Vec3d>> positions = getVec3dArray(ds);

    ds = hdf5.getDataSet("instrument_position/ephemeris_times");
    std::vector<std::vector<double>> times = getMultiSpectralArray<double>(ds);

    ds = hdf5.getDataSet("instrument_position/reference_frame");
    std::vector<int> refFrames;
    ds.read(refFrames);

    try{
      ds = hdf5.getDataSet("instrument_position/velocities");
      std::vector<std::vector<Vec3d>> velocities = getVec3dArray(ds);
      for (int i = 0; i < positions.size(); i++){
        statesVec.push_back(States(times[i], positions[i], velocities[i], refFrames[i]));
      }
      return statesVec;
    }catch(const HighFive::Exception&){
      // Intentionally left blank.  If no velocities, move on.
    }

    for (int i = 0; i < positions.size(); i++){
      statesVec.push_back(States(times[i], positions[i], refFrames[i]));
    }

    return statesVec;
  } catch (...) {
    throw std::runtime_error("Could not parse the instrument position");
  }
}


std::vector<States> getSunPosition(HighFive::File hdf5) {
  try {
    std::vector<States> statesVec;
    HighFive::DataSet ds = hdf5.getDataSet("sun_position/positions");
    std::vector<std::vector<Vec3d>> positions = getVec3dArray(ds);

    ds = hdf5.getDataSet("sun_position/ephemeris_times");
    std::vector<std::vector<double>> times = getMultiSpectralArray<double>(ds);


    ds = hdf5.getDataSet("sun_position/reference_frame");
    std::vector<int> refFrames;
    ds.read(refFrames);

    // If there are velocities, read them
    try{
      ds = hdf5.getDataSet("sun_position/velocities");
      std::vector<std::vector<Vec3d>> velocities = getVec3dArray(ds);
      for (int i = 0; i < positions.size(); i++){
        statesVec.push_back(States(times[i], positions[i], velocities[i], refFrames[i]));
      }
      return statesVec;
    }catch (const HighFive::Exception&){
      // Intentionally left blank.  If there are no velocities, just move on.
    }
    
    for (int i = 0; i < positions.size(); i++){
      statesVec.push_back(States(times[i], positions[i], refFrames[i]));
    }
    return statesVec;

  } catch (...) {
    throw std::runtime_error("Could not parse the sun position");
  }
}

std::vector<Orientations> getInstrumentPointing(HighFive::File hdf5) {
  try {
    HighFive::DataSet ds = hdf5.getDataSet("instrument_pointing/quaternions");
    std::vector<std::vector<Rotation>> rotations = getQuatArray(ds);

    ds = hdf5.getDataSet("instrument_pointing/ephemeris_times");
    std::vector<std::vector<double>> times = getMultiSpectralArray<double>(ds);

    ds = hdf5.getDataSet("instrument_pointing/angular_velocities");
    std::vector<std::vector<Vec3d>> velocities = getVec3dArray(ds);

    std::vector<std::vector<int>> constFrames;
    try{
      ds = hdf5.getDataSet("instrument_pointing/constant_frames");
      constFrames  = getMultiSpectralArray<int>(ds);
    }catch(const HighFive::Exception&){
      //Intentionally left blank.  If there are no constant frames, move on.
    }

    std::vector<std::vector<int>> timeDepFrames;
    try{
      ds = hdf5.getDataSet("instrument_pointing/time_dependent_frames");
      timeDepFrames= getMultiSpectralArray<int>(ds);
    }catch(const HighFive::Exception&){
      //Intentionally left blank.  If there are no time dependent frames, move on.
    }

    std::vector<std::vector<double>> rotArray(timeDepFrames.size(), {1,0,0,0,1,0,0,0,1});
    try{
      ds = hdf5.getDataSet("instrument_pointing/constant_rotation");
      rotArray = getMultiSpectralArray<double>(ds);
    }catch(const HighFive::Exception&){
      //Intentionally left blank.  If there are no constant rotations, move on.
    }

    std::vector<Rotation> constRots;
    for (auto &rots : rotArray){
      Rotation constRot(rots);
      constRots.push_back(constRot);
    }

    
    std::vector<Orientations> orientationsVec;
    for (int i = 0 ; i < rotations.size(); i++){
      orientationsVec.push_back(Orientations(rotations[i], times[i], velocities[i], constRots[i], constFrames[i], timeDepFrames[i]));
    }

    return orientationsVec;

  } catch (...) {
    throw std::runtime_error("Could not parse the instrument pointing");
  }
}

std::vector<Orientations> getBodyRotation(HighFive::File hdf5) {
  try {
    HighFive::DataSet ds = hdf5.getDataSet("body_rotation/quaternions");
    std::vector<std::vector<Rotation>> rotations = getQuatArray(ds);

    ds = hdf5.getDataSet("body_rotation/ephemeris_times");
    std::vector<std::vector<double>> times = getMultiSpectralArray<double>(ds);
    ds = hdf5.getDataSet("body_rotation/angular_velocities");
    std::vector<std::vector<Vec3d>> velocities = getVec3dArray(ds);

    std::vector<std::vector<int>> constFrames;
    try{
      ds = hdf5.getDataSet("body_rotation/constant_frames");
      constFrames  = getMultiSpectralArray<int>(ds);
    }catch(const HighFive::Exception&){
      //Intentionally left blank.  If no const frames, move on.
    }

    std::vector<std::vector<int>> timeDepFrames;
    try{
      ds = hdf5.getDataSet("body_rotation/time_dependent_frames");
      timeDepFrames = getMultiSpectralArray<int>(ds);
    }catch(const HighFive::Exception&){
      //Intentionally left blank.  If no time dependent frames, move on.
    }

    std::vector<std::vector<double>> rotArrays(rotations.size(), {1,0,0,0,1,0,0,0,1});
    try{
      ds = hdf5.getDataSet("body_rotation/constant_rotation");
      rotArrays = getMultiSpectralArray<double>(ds);
    }catch(const HighFive::Exception&){
      //Intentionally left blank.  If no constant rotations, move on.
    }

    std::vector<Rotation> constRots;
    for (auto &rot : rotArrays){
      Rotation constRot(rot);
      constRots.push_back(constRot);
    }

    std::vector<Orientations> orientationsVec;
    for (size_t i = 0 ; i < rotations.size(); i++){
      Orientations orientation(rotations[i], times[i], velocities[i], constRots[i], constFrames[i], timeDepFrames[i]);
      orientationsVec.push_back(orientation);
    }
    return orientationsVec;

  } catch (...) {
    throw std::runtime_error("Could not parse the body rotation");
  }
}

}
