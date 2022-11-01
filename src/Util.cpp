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
    json distoriton_subset = isd.at("optical_distortion");

    json::iterator it = distoriton_subset.begin();

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
  }
  break;
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

}
