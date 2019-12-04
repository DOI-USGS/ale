#include "Isd.h"
#include "Util.h"


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

  line_scan_rate = getLineScanRate(isd);

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
}
