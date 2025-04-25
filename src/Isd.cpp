
#include "ale/Isd.h"
#include "ale/Util.h"

using json = nlohmann::json;

ale::Isd::Isd(std::string isd_file) {
  json isd = json::parse(isd_file);

  usgscsm_name_model = getSensorModelName(isd);
  image_id = getImageId(isd);
  name_platform = getPlatformName(isd);
  name_sensor = getSensorName(isd);

  image_lines = getTotalLines(isd);
  image_samples = getTotalSamples(isd);

  starting_ephemeris_time = getStartingTime(isd);
  center_ephemeris_time = getCenterTime(isd);

  try {
    line_scan_rate = getLineScanRate(isd);
  } catch (...) {
    // Framers do not deal with line scan rates
    // This is assuming that we may be dealing with a framer rather than
    // a malformed ISD
    line_scan_rate = std::vector<std::vector<double>>();
  }

  detector_sample_summing = getSampleSumming(isd);
  detector_line_summing = getLineSumming(isd);

  focal_length = getFocalLength(isd);
  focal_uncertainty = getFocalLengthUncertainty(isd);

  focal2pixel_line = getFocal2PixelLines(isd);
  focal2pixel_sample = getFocal2PixelSamples(isd);

  detector_center_line = getDetectorCenterLine(isd);
  detector_center_sample = getDetectorCenterSample(isd);

  starting_detector_line = getDetectorStartingLine(isd);
  starting_detector_sample = getDetectorStartingSample(isd);

  min_reference_height = getMinHeight(isd);
  max_reference_height = getMaxHeight(isd);

  semi_major = getSemiMajorRadius(isd);
  semi_minor = getSemiMinorRadius(isd);

  distortion_model = getDistortionModel(isd);
  distortion_coefficients = getDistortionCoeffs(isd);

  interpMethod = getInterpolationMethod(isd);

  inst_pos = getInstrumentPosition(isd);
  sun_pos = getSunPosition(isd);

  inst_pointing = getInstrumentPointing(isd);
  body_rotation = getBodyRotation(isd);
 }



ale::MultiSpectralIsd::MultiSpectralIsd(HighFive::File hdf5) {
  usgscsm_name_model = getSensorModelName(hdf5);
  image_id = getImageId(hdf5);
  name_platform = getPlatformName(hdf5);
  name_sensor = getSensorName(hdf5);

  image_lines = getTotalLines(hdf5);
  image_samples = getTotalSamples(hdf5);

  starting_ephemeris_time = getStartingTime(hdf5);
  center_ephemeris_time = getCenterTime(hdf5);

  try {
    line_scan_rate = getLineScanRate(hdf5);
  } catch (...) {
    // Framers do not deal with line scan rates
    // This is assuming that we may be dealing with a framer rather than
    // a malformed ISD
    line_scan_rate = std::vector<std::vector<std::vector<double>>>();
  }

  detector_sample_summing = getSampleSumming(hdf5);
  detector_line_summing = getLineSumming(hdf5);

  focal_length = getFocalLength(hdf5);
  focal_uncertainty = getFocalLengthUncertainty(hdf5);

  focal2pixel_line = getFocal2PixelLines(hdf5);
  focal2pixel_sample = getFocal2PixelSamples(hdf5);

  detector_center_line = getDetectorCenterLine(hdf5);
  detector_center_sample = getDetectorCenterSample(hdf5);

  starting_detector_line = getDetectorStartingLine(hdf5);
  starting_detector_sample = getDetectorStartingSample(hdf5);

  min_reference_height = getMinHeight(hdf5);
  max_reference_height = getMaxHeight(hdf5);

  semi_major = getSemiMajorRadius(hdf5);
  semi_minor = getSemiMinorRadius(hdf5);

  distortion_model = getDistortionModel(hdf5);
  distortion_coefficients = getDistortionCoeffs(hdf5);

  interpMethod = getInterpolationMethod(hdf5);

  inst_pos = getInstrumentPosition(hdf5);
  sun_pos = getSunPosition(hdf5);

  inst_pointing = getInstrumentPointing(hdf5);
  body_rotation = getBodyRotation(hdf5);
}
