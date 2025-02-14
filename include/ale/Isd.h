#ifndef ALE_ISD_H
#define ALE_ISD_H

#include <string>
#include <vector>
#include <map>

#include <highfive/highfive.hpp>
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
  

  class MultiSpectralIsd {
    public:

    MultiSpectralIsd(HighFive::File);

    std::vector<std::string> usgscsm_name_model;
    std::vector<std::string> name_platform;
    std::vector<std::string> image_id;
    std::vector<std::string> name_sensor;

    std::vector<double> semi_major;
    std::vector<double> semi_minor;

    std::vector<double> detector_sample_summing;
    std::vector<double> detector_line_summing;

    std::vector<double> focal_length;
    std::vector<double> focal_uncertainty;

    std::vector<double> detector_center_line;
    std::vector<double> detector_center_sample;

    // should probably be ints
    std::vector<double> starting_detector_line;
    std::vector<double> starting_detector_sample;

    std::vector<std::vector<double>> focal2pixel_line;
    std::vector<std::vector<double>> focal2pixel_sample;

    // maybe change
    std::vector<DistortionType> distortion_model;
    std::vector<std::vector<double>> distortion_coefficients;

    std::vector<unsigned int> image_lines;
    std::vector<unsigned int> image_samples;

    std::vector<double> max_reference_height;
    std::vector<double> min_reference_height;

    std::vector<std::vector<std::vector<double>>> line_scan_rate;

    std::vector<double> starting_ephemeris_time;
    std::vector<double> center_ephemeris_time;

    std::vector<nlohmann::json> naif_keywords;

    std::vector<PositionInterpolation> interpMethod;

    std::vector<States> inst_pos;
    std::vector<States> sun_pos;

    std::vector<Orientations> inst_pointing;
    std::vector<Orientations> body_rotation;
  };
}


#endif
