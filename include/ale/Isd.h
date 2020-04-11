#ifndef ALE_ISD_H
#define ALE_ISD_H

#include <string>
#include <vector>
#include <map>

#include <nlohmann/json.hpp>

#include "ale/Distortion.h"
#include "ale/Rotation.h"
#include "ale/States.h"
#include "ale/Orientations.h"

namespace ale {

  class Isd {
    public:

    Isd(std::string);

    std::string usgscsm_name_model;
    std::string name_platform;
    std::string image_id;
    std::string name_sensor;

    double semi_major;
    double semi_minor;

    double detector_sample_summing;
    double detector_line_summing;

    double focal_length;
    double focal_uncertainty;

    double detector_center_line;
    double detector_center_sample;

    // should probably be ints
    double starting_detector_line;
    double starting_detector_sample;

    std::vector<double> focal2pixel_line;
    std::vector<double> focal2pixel_sample;

    // maybe change
    DistortionType distortion_model;
    std::vector<double> distortion_coefficients;

    unsigned int image_lines;
    unsigned int image_samples;

    double max_reference_height;
    double min_reference_height;

    std::vector<std::vector<double>> line_scan_rate;

    double starting_ephemeris_time;
    double center_ephemeris_time;

    nlohmann::json naif_keywords;

    PositionInterpolation interpMethod;

    States inst_pos;
    States sun_pos;

    Orientations inst_pointing;
    Orientations body_rotation;
  };
}

#endif
