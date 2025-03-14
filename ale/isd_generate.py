#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create ISD .json file from image or label file."""

# This is free and unencumbered software released into the public domain.
#
# The authors of ale do not claim copyright on the contents of this file.
# For more details about the LICENSE terms and the AUTHORS, you will
# find files of those names at the top level of this repository.
#
# SPDX-License-Identifier: CC0-1.0

import argparse
import concurrent.futures
import logging
import os
import pvl
from pathlib import Path, PurePath
import sys
import json
import ale
import brotli
import json
from ale.drivers import AleJsonEncoder

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-k", "--kernel",
        type=Path,
        help="Typically this is an optional metakernel file, care should be "
             "taken by the user that it is applicable to all of the input "
             "files.  It can also be a single ISIS cube, which is sometimes "
             "needed if the input file is a label file."
    )
    parser.add_argument(
        "--max_workers",
        default=None,
        type=int,
        help="If more than one file is provided to work on, this program "
             "will engage multiprocessing to parallelize the work.  This "
             "multiprocessing will default to the number of processors on the "
             "machine.  If you want to throttle this to use less resources on "
             "your machine, indicate the number of processors you want to use."
    )
    parser.add_argument(
        "-o", "--out",
        type=Path,
        help="Optional output file.  If not specified, this will be set to "
             "the input filename with its final suffix replaced with .json. "
             "If multiple input files are provided, this option will be ignored "
             "and the default strategy of replacing their final suffix with "
             ".json will be used to generate the output file paths."
    )
    parser.add_argument(
        "--semimajor", "-a", "-r", "--radius",
        required="--semiminor" in sys.argv or "-b" in sys.argv,
        type=float,
        default=None,
        help="Optional spherical radius (m) override.  Setting "
             " '--semimajor 3396190.0' "
             "will override both semi-major and semi-minor radius values with the same value.  "
             "An ellipsoid can be defined if '--semiminor' is also sent.  "
             "If not specified, the default radius values "
             "(e.g.; from NAIF kernels or the ISIS Cube) will be used.  "
             "When is a semimajor specification needed? Beyond a specialized need, it is common "
             "that planetary bodies are defined as a triaxial body.  "
             "In most of these cases, the IAU WGCCRE report recommends the use of a "
             "best-fit sphere for a derived map product.  "
             "For current IAU spherical recommendations see: "
             "https://doi.org/10.1007/s10569-017-9805-5 or "
             "http://voparis-vespa-crs.obspm.fr:8080/web/ ."
             "Make sure radius values are in meters (not kilometers)."
    )
    parser.add_argument(
        "--semiminor", "-b",
        type=float,
        default=None,
        help="Optional semi-minor radius (m) override. When using this parameter, you must also define the semi-major radius. For example: "
             " '--semimajor 3396190.0 --semiminor 3376200.0' "
             "will override the semi-major and semi-minor radii to define an ellipsoid.  "
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display information as program runs."
    )
    parser.add_argument(
        "-c", "--compress",
        action="store_true",
        help="Output a compressed isd json file with .br file extension. "
             "Ale uses the brotli compression algorithm. "
             "To decompress an isd file run: python -c \"import ale.isd_generate as isdg; isdg.decompress_json('/path/to/isd.br')\""
    )
    parser.add_argument(
        "-i", "--only_isis_spice",
        action="store_true",
        help="Only use drivers that read from spiceinit'd ISIS cubes"
    )
    parser.add_argument(
        "-n", "--only_naif_spice",
        action="store_true",
        help="Only use drivers that generate fresh spice data"
    )
    parser.add_argument(
        "-l", "--local",
        action="store_true",
        help="Generate local spice data, an isd that is unaware of itself relative to "
             "target body. This is largely used for landed/rover data."
    )
    parser.add_argument(
        "-N", "--nadir",
        action="store_true",
        help="Generate nadir spice pointing, an isd that has pointing directly towards "
             "the center of the target body."
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f"ale version {ale.__version__}",
        help="Shows ale version number."
    )
    parser.add_argument(
        "-w", "--use_web_spice",
        action="store_true",
        help="Get spice over the restful interface."
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Path to image or label file (or multiple)."
    )
    args = parser.parse_args()

    log_level = logging.INFO
    if args.verbose:
        log_level = logging.WARNING

    logging.basicConfig(format="%(message)s", level=log_level)
    logger.setLevel(log_level)

    if args.kernel is None:
        k = None
    else:
        try:
            k = ale.util.generate_kernels_from_cube(args.kernel, expand=True)
        except (KeyError, pvl.exceptions.LexerError):
            k = [args.kernel, ]

    if args.semimajor is None:
        radii = None
    else:
        if args.semiminor is None:  # set a sphere
          radii = [args.semimajor, args.semimajor]
        else:                       # set as ellipsoid
          radii = [args.semimajor, args.semiminor]

    if len(args.input) == 1:
        try:
            file_to_isd(args.input[0], args.out, radii, kernels=k, log_level=log_level, compress=args.compress, only_isis_spice=args.only_isis_spice, only_naif_spice=args.only_naif_spice, local=args.local, nadir=args.nadir)
        except Exception as err:
            # Seriously, this just throws a generic Exception?
            sys.exit(f"File {args.input[0]}: {err}")
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    file_to_isd, f, **{"radii": radii,
                                       "kernels": k, 
                                       "log_level": log_level, 
                                       "only_isis_spice": args.only_isis_spice, 
                                       "only_naif_spice": args.only_naif_spice,
                                       "local": args.local,
                                       "nadir": args.nadir,
                                       "use_web":args.use_web_spice}
                ): f for f in args.input
            }
            for f in concurrent.futures.as_completed(futures):
                # Since file_to_isd() doesn't return anything,
                # we don't need to do anything with the return value,
                # just check its result for an Exception, if we get
                # one, note it and continue.
                try:
                    f.result()
                except Exception as err:
                    logger.error(f"File {futures[f]}: {err}")


def file_to_isd(
    file: os.PathLike,
    out: os.PathLike = None,
    radii: list = None,
    kernels: list = None,
    log_level=logging.WARNING,
    compress=False,
    only_isis_spice=False,
    only_naif_spice=False,
    local=False,
    nadir=False,
    use_web=False):
    """
    Returns nothing, but acts as a thin wrapper to take the *file* and generate
    an ISD at *out* (if given, defaults to replacing the extension on *file*
    with .json), optionally using the passed *kernels*.
    """
    # Yes, it is aggravating to have to pass the log_level into the function.
    # If this weren't trying to be fancy with multiprocessing, it wouldn't
    # be needed, and if this program were more complex, you'd build different
    # infrastructure.  Probably overkill to use logging here.

    if out is None:
        isd_file = Path(file).with_suffix(".json")
    else:
        isd_file = Path(out)

    # These two lines might seem redundant, but they are the only
    # way to guarantee that when file_to_isd() is spun up in its own
    # process, that these are set properly.
    logging.basicConfig(format="%(message)s", level=log_level)
    logger.setLevel(log_level)

    logger.info(f"Reading: {file}")
    props = {}

    if local:
        props['landed'] = local

    if nadir:
        props['nadir'] = nadir

    if use_web:
        props["web"] = use_web

    if kernels is not None:
        kernels = [str(PurePath(p)) for p in kernels]
        props["kernels"] = kernels
        usgscsm_str = ale.loads(file, props=props, verbose=log_level>logging.INFO, only_isis_spice=only_isis_spice, only_naif_spice=only_naif_spice)
    else:
        usgscsm_str = ale.loads(file, props=props, verbose=log_level>logging.INFO, only_isis_spice=only_isis_spice, only_naif_spice=only_naif_spice)

    if radii is not None:
        # first convert to kilometers for ISD
        radii = [x / 1000.0 for x in radii] 
        
        usgscsm_json = json.loads(usgscsm_str)
        usgscsm_json["radii"]["semimajor"] = radii[0]
        usgscsm_json["radii"]["semiminor"] = radii[1]
        logger.info(f"Overriding radius to (km):")
        logger.info(usgscsm_json["radii"])
        usgscsm_str = json.dumps(usgscsm_json, indent=2)

    logger.info(f"Writing: {isd_file}")
    if compress:
        logger.info(f"Writing: {os.path.splitext(isd_file)[0] + '.br'}")
        compress_json(usgscsm_str, os.path.splitext(isd_file)[0] + '.br')
    else:
        logger.info(f"Writing: {isd_file}")  
        isd_file.write_text(usgscsm_str)


    return

def compress_json(json_data, output_file):
    """
    Compresses inputted JSON data using brotli compression algorithm.
    
    Parameters
    ----------
    json_data : str
        JSON data

    output_file : str
        The output compressed file path with .br extension.

    """
    binary_json = json.dumps(json_data).encode('utf-8')

    if not os.path.splitext(output_file)[1] == '.br':
        raise ValueError("Output file {} is not a valid .br file extension".format(output_file.split(".")[1]))
    
    with open(output_file, 'wb') as f:
        f.write(brotli.compress(binary_json))


def decompress_json(compressed_json_file):
    """
    Decompresses inputted .br file.
    
    Parameters
    ----------
    compressed_json_file : str
        .br file path

    Returns
    -------
    str
        Decompressed .json file path
    """
    if not os.path.splitext(compressed_json_file)[1] == '.br':
        raise ValueError("Inputted file {} is not a valid .br file extension".format(compressed_json_file))
    with open(compressed_json_file, 'rb') as f:
        data = f.read()
    with open(compressed_json_file, 'wb') as f:
        f.write(brotli.decompress(data))

    os.rename(compressed_json_file, os.path.splitext(compressed_json_file)[0] + '.json')

    return os.path.splitext(compressed_json_file)[0] + '.json'

if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as err:
        sys.exit(err)

