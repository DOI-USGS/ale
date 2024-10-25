#ifndef ALE_INCLUDE_ALE_H
#define ALE_INCLUDE_ALE_H

#include <nlohmann/json.hpp>

#include <string>

namespace ale {
  /**
   * Load all of the metadata for an image into an ISD string.
   * This method calls the Python driver structure in ALE to load all
   * of the metadata for an image into an ISD string. See the Python
   * loads method for how this is implemented on the Python side.
   *
   * @param filename The filename of the image to load metadata for
   * @param props A JSON formatted properties string to pass to the Python drivers.
   *              Users can specify certain properties that the drivers will use.
   *              Currently kernels and nadir properties are allowed. See the
   *              data_naif driver mix-in for details.
   * @param formatter A string specifying the format of the output ISD string.
   *                  Currently supported formatters are isis, usgscsm, and ale.
   *                  The isis and usgscsm formatters will be deprecated in the future.
   * @param verbose A flag to output what the load function is attempting to do.
   *                If set to true, information about the drivers load attempts
   *                to use will be output to standard out.
   * @param onlyIsisSpice A flag the forces the load function to only use IsisSpice
   *                      drivers
   * @param onlyNaifSpice A flag the forces the load function to only use NaifSpice
   *                      drivers
   *
   * @returns A string containing a JSON formatted ISD for the image.
   */
  std::string loads(std::string filename, std::string props="", std::string formatter="ale", int indent=2, bool verbose=true, bool onlyIsisSpice=false, bool onlyNaifSpice=false);

  /**
   * Load all of the metadata for an image into a JSON ISD.
   * This method is a convenience wrapper around the loads method that parses the
   * string output of loads into a JSON object.
   *
   * @param filename The filename of the image to load metadata for
   * @param props A JSON formatted properties string to pass to the Python drivers.
   *              Users can specify certain properties that the drivers will use.
   *              Currently kernels and nadir properties are allowed. See the
   *              data_naif driver mix-in for details.
   * @param formatter A string specifying the format of the output ISD string.
   *                  Currently supported formatters are isis, usgscsm, and ale.
   *                  The isis and usgscsm formatters will be deprecated in the future.
   * @param verbose A flag to output what the load function is attempting to do.
   *                If set to true, information about the drivers load attempts
   *                to use will be output to standard out.
   * @param onlyIsisSpice A flag the forces the load function to only use IsisSpice
   *                      drivers
   * @param onlyIsisSpice A flag the forces the load function to only use NaifSpice
   *                      drivers
   *
   * @returns A string containing a JSON formatted ISD for the image.
   */
  nlohmann::json load(std::string filename, std::string props="", std::string formatter="ale", bool verbose=true, bool onlyIsisSpice=false, bool onlyNaifSpice=false);
}

#endif // ALE_H
