#include <stdexcept>
#include <algorithm>

#include "ale.h"
#include "Util.h"


bool iequals(const string& a, const string& b)
{
    return std::equal(a.begin(), a.end(),
                      b.begin(), b.end(),
                      [](char a, char b) {
                          return tolower(a) == tolower(b);
                      });
}


std::string ale::getSensorModelName(json isd) {
  std::string name = "";
  try {
    name = isd.at("name_model");
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor model name.");
  }
  return name;
}

std::string ale::getImageId(json isd) {
  std::string id = "";
  try {
    id = isd.at("image_identifier");
  } catch (...) {
    throw std::runtime_error("Could not parse the image identifier.");
  }
  return id;
}

std::string ale::getSensorName(json isd) {
  std::string name = "";
  try {
    name = isd.at("name_sensor");
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor name.");
  }
  return name;
}

std::string ale::getIsisCameraVersion(json isd) {
  std::string name = "";
  try {
    name = isd.at("IsisCameraVersion");
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor name.");
  }
  return name;
}


std::string ale::getPlatformName(json isd) {
  std::string name = "";
  try {
    name = isd.at("name_platform");
  } catch (...) {
    throw std::runtime_error("Could not parse the platform name.");
  }
  return name;
}

std::string ale::getLogFile(nlohmann::json isd) {
  std::string file = "";
  try {
    file = isd.at("log_file");
  } catch (...) {
    throw std::runtime_error("Could not parse the log filename.");
  }
  return file;
}

int ale::getTotalLines(json isd) {
  int lines = 0;
  try {
    lines = isd.at("image_lines");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the number of lines in the image.");
  }
  return lines;
}

int ale::getTotalSamples(json isd) {
  int samples = 0;
  try {
    samples = isd.at("image_samples");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the number of samples in the image.");
  }
  return samples;
}

double ale::getStartingTime(json isd) {
  double time = 0.0;
  try {
    time = isd.at("starting_ephemeris_time");
  } catch (...) {
    throw std::runtime_error("Could not parse the image start time.");
  }
  return time;
}

double ale::getCenterTime(json isd) {
  double time = 0.0;
  try {
    time = isd.at("center_ephemeris_time");
  } catch (...) {
    throw std::runtime_error("Could not parse the center image time.");
  }
  return time;
}

ale::interpolation ale::getInterpolationMethod(json isd) {
  std::string interpoMethod = "linear";
  try {
    interpMethod = isd.at("interpolation_method");
     
    if (iequals(interpMethod, "linear")) {
      return ale::interpolation::LINEAR;
    }
    else if (iequals(interpMethod, "spline")){ 
      return ale::interpolation::SPLINE;
    } 
    else if (iequals(interpMethod, "lagrange")) {
      // return ale::interpolation::LAGRANGE;
      // temporary stand-in  
      throw "Not implemented"; 
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the interpolation method.");
  }
  return time;
}

std::vector<std::vector<double>> ale::getLineScanRate(json isd) {
  std::vector<std::vector<double>> lines;
  try {
    for (auto &scanRate : isd.at("line_scan_rate")) {
      lines.push_back(scanRate.get<std::vector<double>>());
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the integration start lines in "
                             "the integration time table.");
  }
  return lines;
}


int ale::getSampleSumming(json isd) {
  int summing = 0;
  try {
    summing = isd.at("detector_sample_summing");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the sample direction detector pixel summing.");
  }
  return summing;
}

int ale::getLineSumming(json isd) {
  int summing = 0;
  try {
    summing = isd.at("detector_line_summing");
  } catch (...) {
    throw std::runtime_error(
        "Could not parse the line direction detector pixel summing.");
  }
  return summing;
}

double ale::getFocalLength(json isd) {
  double length = 0.0;
  try {
    length = isd.at("focal_length_model").at("focal_length");
  } catch (...) {
    throw std::runtime_error("Could not parse the focal length.");
  }
  return length;
}

double ale::getFocalLengthUncertainty(json isd) {
  double uncertainty = 1.0;
  try {
    uncertainty = isd.at("focal_length_model").value("focal_uncertainty", uncertainty);
  } catch (...) {
    throw std::runtime_error("Could not parse the focal length uncertainty.");
  }
  return uncertainty;
}

std::vector<double> ale::getFocal2PixelLines(json isd) {
  std::vector<double> transformation;
  try {
    transformation = isd.at("focal2pixel_lines").get<std::vector<double>>();
  } catch (...) {
    throw std::runtime_error("Could not parse the focal plane coordinate to "
                             "detector lines transformation.");
  }
  return transformation;
}

std::vector<double> ale::getFocal2PixelSamples(json isd) {
  std::vector<double> transformation;
  try {
    transformation = isd.at("focal2pixel_samples").get<std::vector<double>>();
  } catch (...) {
    throw std::runtime_error("Could not parse the focal plane coordinate to "
                             "detector samples transformation.");
  }
  return transformation;
}

double ale::getDetectorCenterLine(json isd) {
  double line;
  try {
    line = isd.at("detector_center").at("line");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector center line.");
  }
  return line;
}

double ale::getDetectorCenterSample(json isd) {
  double sample;
  try {
    sample = isd.at("detector_center").at("sample");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector center sample.");
  }
  return sample;
}

double ale::getDetectorStartingLine(json isd) {
  double line;
  try {
    line = isd.at("starting_detector_line");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector starting line.");
  }
  return line;
}

double ale::getDetectorStartingSample(json isd) {
  double sample;
  try {
    sample = isd.at("starting_detector_sample");
  } catch (...) {
    throw std::runtime_error("Could not parse the detector starting sample.");
  }
  return sample;
}

double ale::getMinHeight(json isd) {
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

double ale::getMaxHeight(json isd) {
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

double ale::getSemiMajorRadius(json isd) {
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

double ale::getSemiMinorRadius(json isd) {
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
ale::DistortionType ale::getDistortionModel(json isd) {
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
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the distortion model.");
  }
  return DistortionType::TRANSVERSE;
}

std::vector<double> ale::getDistortionCoeffs(json isd) {
  std::vector<double> coefficients;

  ale::DistortionType distortion = getDistortionModel(isd);

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
  }
  throw std::runtime_error(
      "Could not parse the distortion model coefficients.");

  return coefficients;
}

std::vector<double> ale::getSunPositions(json isd) {
  std::vector<double> positions;
  try {
    json jayson = isd.at("sun_position");
    for (auto &location : jayson.at("positions")) {
      positions.push_back(location[0].get<double>());
      positions.push_back(location[1].get<double>());
      positions.push_back(location[2].get<double>());
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the sun positions.");
  }
  return positions;
}

std::vector<double> ale::getSensorPositions(json isd) {
  std::vector<double> positions;
  try {
    json jayson = isd.at("sensor_position");
    for (auto &location : jayson.at("positions")) {
      positions.push_back(location[0].get<double>());
      positions.push_back(location[1].get<double>());
      positions.push_back(location[2].get<double>());
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor positions.");
  }
  return positions;
}

std::vector<double> ale::getSensorVelocities(json isd) {
  std::vector<double> velocities;
  try {
    json jayson = isd.at("sensor_position");
    for (auto &velocity : jayson.at("velocities")) {
      velocities.push_back(velocity[0].get<double>());
      velocities.push_back(velocity[1].get<double>());
      velocities.push_back(velocity[2].get<double>());
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor velocities.");
  }
  return velocities;
}

std::vector<double> ale::getSensorOrientations(json isd) {
  std::vector<double> quaternions;
  try {
    for (auto &quaternion : isd.at("sensor_orientation").at("quaternions")) {
      quaternions.push_back(quaternion[0]);
      quaternions.push_back(quaternion[1]);
      quaternions.push_back(quaternion[2]);
      quaternions.push_back(quaternion[3]);
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor orientations.");
  }
  return quaternions;
}


std::vector<double> ale::getJsonDouble1Array(json obj) {
  std::vector<double> positions;
  try {
    for (auto &location : obj) {
      positions.push_back(location.get<double>());
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor positions.");
  }
  return positions;
}


std::vector<Vec3d> ale::getJsonVec3dArray(json obj) {
  std::vector<Vec3d> positions;
  try {
    for (auto &location : obj) {
      Vec3d vec(location[0].get<double>(),location[1].get<double>(), location[2].get<double>() );
      positions.append(vec);
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the sensor positions.");
  }
  return positions;
}


ale::States getInstrumentPosition(json isd) {
  try {
    std::vector<Vec3d> positions = getJsonVec3dArray(isd.at("InstrumentPosition").at("Positions"));
    std::vector<double> times = getJsonDouble1Array(isd.at("InstrumentPosition").at("EphemerisTimes")); 
     
    json frames = json.at("InstrumentPointing").at("TimeDependantFrames");
    int reFrame = frames.at(frames.size()-1).get<int>();

    bool hasVelocities = isd.at("InstrumentPosition").find("Velocities") != isd.at("InstrumentPosition").end();
    if (hasVelocities) {
      std::vector<Vec3d> velocities = getJsonVec3dArray(isd.at("InstrumentPositions").at("Velocities")); 
      States states(times, positions, velocties, refFrame);  
      return states;
    }
    else {
      States states(times, positions, refFrame);
      return states; 
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the instrument position");
  }
}


ale::States getSunPosition(json isd) {
  try {
    json sp = isd.at("SunPosition");
    std::vector<Vec3d> positions = getJsonVec3dArray(isd.at("SunPosition").at("Positions"));
    std::vector<double> times = getJsonDouble1Array(isd.at("SunPosition").at("EphemerisTimes")); 
     
    json frames = json.at("SunPointing").at("TimeDependantFrames");
    int reFrame = frames.at(frames.size()-1).get<int>();

    bool hasVelocities = isd.at("SunPosition").find("Velocities") != isd.at("SunPosition").end();
    if (hasVelocities) {
      std::vector<Vec3d> velocities = getJsonVec3dArray(isd.at("InstrumentPositions").at("Velocities")); 
      States states(times, positions, velocties, refFrame);  
      return states;
    }
    else {
      States states(times, positions, refFrame);
      return states; 
    }
  } catch (...) {
    throw std::runtime_error("Could not parse the instrument position");
  }
}
