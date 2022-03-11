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
from pathlib import Path
import sys

import ale

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
        "-v", "--verbose",
        action="store_true",
        help="Display information as program runs."
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f"ale version {ale.__version__}",
        help="Shows ale version number."
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Path to image or label file (or multiple)."
    )
    args = parser.parse_args()

    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO

    logging.basicConfig(format="%(message)s", level=log_level)
    logger.setLevel(log_level)

    if args.kernel is None:
        k = None
    else:
        try:
            k = ale.util.generate_kernels_from_cube(args.kernel, expand=True)
        except KeyError:
            k = [args.kernel, ]

    if len(args.input) == 1:
        try:
            file_to_isd(args.input[0], args.out, kernels=k, log_level=log_level)
        except Exception as err:
            # Seriously, this just throws a generic Exception?
            sys.exit(f"File {args.input[0]}: {err}")
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    file_to_isd, f, **{"kernels": k, "log_level": log_level}
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
    kernels: list = None,
    log_level=logging.WARNING
):
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
    if kernels is not None:
        usgscsm_str = ale.loads(file, props={'kernels': kernels})
    else:
        usgscsm_str = ale.loads(file)

    logger.info(f"Writing: {isd_file}")
    isd_file.write_text(usgscsm_str)

    return


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as err:
        sys.exit(err)
