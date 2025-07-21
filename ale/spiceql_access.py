import logging
import math
import os
import requests

import numpy as np

from ale import util
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(message)s", level=logging.INFO)

spiceql_url = os.environ.get('SPICEQL_REST_URL')
log_level = os.environ.get('ALESPICEQL_LOG_LEVEL')
if log_level is not None and log_level.lower() == "debug":
    logger.setLevel(logging.DEBUG)

try:
    import pyspiceql
except ImportError:
    logger.info("Optional package 'pyspiceql' was not imported, " \
                "install 'spiceql' to use this feature. " \
                "NOTE: SpiceQL is not supported on Windows.") 

def stringify_web_args(function_args):
    """
    Takes a dictionary of args and converts them into web acceptable strings

    Parameters
    ----------
    function_args : dict

    Returns
    -------
    clean_function_args : dict
    """
    clean_function_args = function_args
    for key, value in function_args.items():
        if isinstance(value, np.ndarray):
            clean_function_args[key] = str(value.tolist())
        if isinstance(value, list):
            clean_function_args[key] = str(value)
        if isinstance(value, bool):
            clean_function_args[key] = str(value)

    return clean_function_args

def check_response(response):
    """
    Checks that a response from the spice server returned correctly

    Parameters
    ----------
    : obj
      Request response object
    """
    
    response.raise_for_status()

    if response.status_code != 200:
        raise requests.HTTPError(f"Received code {response.status_code} from spice server, with error: {response.json()}")

    if response.json()["statusCode"] != 200:
        raise requests.HTTPError(f"Received code {response.json()['statusCode']} from spice server, with error: {response.json()['body']}")


def spiceql_call(function_name = "", function_args = {}, use_web=False):
    """
    Interface to SpiceQL (Spice Query Library) for both Offline and Online use

    This function will access the value passed through props defined as `web`. This
    value determines the access pattern for spice data. When set to Online, you will
    access the SpiceQL service provided through the USGS Astro AWS platform. This service
    performs kernel and data aquisition. If set to Offline, you will access locally loaded
    kernels, and SpiceQL will do no searching for you.

    Parameters
    ----------
    functions_name : str
                        String defineing the function to call, properly exposed SpiceQL
                        functions should map 1-to-1 with endpoints on the service
                    
    function_args : dict
                    Dictionary of arguments used by the function

    use_web : bool
              Boolean value to either use the USGS web service when set to True
              or local data when set to False

    Returns : any
                Any return from a SpiceQL function
    """
    logger.debug(f"Calling {function_name} with args: {function_args}")
    if use_web == False:
        function_args["useWeb"] = False
        func = getattr(pyspiceql, function_name)
        ret = func(**function_args)[0]
        return ret
    
    if spiceql_url:
        url = spiceql_url
    else:
        url = "https://astrogeology.usgs.gov/apis/spiceql/latest/"

    url += function_name
    headers = {
        'accept': '*/*',
        'Content-Type': 'application/json'
    }

    # Convert any args being passed over the wire to strings
    clean_function_args = stringify_web_args(function_args)
    logger.debug("Args: " + str(clean_function_args))

    if function_name == "getTargetStates":
        post_body = str(clean_function_args).replace("\'", "\"")
        logger.debug("getTargetStates POST Payload: " + post_body)
        response = requests.post(url, data=post_body, headers=headers, verify=False)
    else:
        response = requests.get(url, params=clean_function_args, headers=headers, verify=False)
    check_response(response)

    logger.debug(f"Request URL={str(response.url)}")
    logger.debug(f"Kernels={str(response.json()['body']['kernels'])}")
    logger.debug(f"Data={str(response.json()['body']['return'])}")
    return response.json()["body"]["return"]

def get_ephem_data(times, function_name, batch_size=300, web=False, function_args={}):
    """
    This function provides access to ephemeris data aquisition in spiceql. 
    For the web service there is a limited number of times that can be
    requested at once due to URL size limits. This limit is ~400 times.
    This function is used to chunk up the requests to be submitted all at
    once when accessing the web service.

    When accessing local data, the function queries all of the times at once.

    Parameters
    ----------
    times : list
            List of ephemeris times to get data for

    function_name : str
                    The name of the spiceql function to run. This can be
                    either getTargetOrientations or getTargetStates

    batch_size : int
                 Number of times to request to the web services at once.
                 Where the number of request is times / batch_size

    web : bool
          Boolean value to either use the USGS web service when set to True
          or local data when set to False

    kwargs : dict
             Arguments to be passed to the spiceql function

    Returns
    -------
    results : list
              Returns a list of ephemeris data to the user as a list
              of lists. Where each element corrisponds to the time
              requested in times.
    """
    valid_functions = ["getTargetOrientations", "getTargetStates"]
    if function_name not in valid_functions:
        raise ValueError(f"The function name {function_name} is not supported "
                          "by this function please pass one of the following: " + str(valid_functions))
    if not web:
      func_args = {**function_args}
      func_args["ets"] = times
      ephemeris_data = spiceql_call(function_name, func_args, web)
      return ephemeris_data
    
    batches = math.ceil(len(times) / batch_size)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for i in range(1, batches+1):
            batch_times = times[(i - 1) * batch_size: i * batch_size]
            func_args = {**function_args}
            func_args["ets"] = batch_times
            futures.append(executor.submit(spiceql_call, function_name, func_args, web))

        results = []
        for i in range(0, len(futures)):
            results.append(futures[i].result())


    flat_results = []
    for i in results:
        flat_results.extend(i)
    return flat_results
