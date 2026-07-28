import argparse
import json
import logging
import os, sys
import pyspiceql as psql
import spiceypy as spice

from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)

# ISD dictionary key constants
ISD_KEY_INSTRUMENT_POSITION = "instrument_position"
ISD_KEY_INSTRUMENT_POINTING = "instrument_pointing"
ISD_KEY_SPK_TABLE_START = "spk_table_start_time"
ISD_KEY_SPK_TABLE_END = "spk_table_end_time"
ISD_KEY_CK_TABLE_START = "ck_table_start_time"
ISD_KEY_CK_TABLE_END = "ck_table_end_time"
ISD_KEY_TIME_DEPENDENT_FRAMES = "time_dependent_frames"
ISD_KEY_NAIF_KEYWORDS = "naif_keywords"
ISD_KEY_BODY_CODE = "BODY_CODE"
ISD_KEY_POSITIONS = "positions"
ISD_KEY_EPHEMERIS_TIMES = "ephemeris_times"
ISD_KEY_VELOCITIES = "velocities"
ISD_KEY_REFERENCE_FRAME = "reference_frame"
ISD_KEY_QUATERNIONS = "quaternions"
ISD_KEY_ANGULAR_VELOCITIES = "angular_velocities"
ISD_KEY_NAME_SENSOR = "name_sensor"
ISD_KEY_NAME_PLATFORM = "name_platform"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f", "--isd_file",
        type=Path,
        help="Input ISD (Image Support Data) JSON file to extract kernel information from."
    )
    parser.add_argument(
        "-k", "--kernel_type",
        default=None,
        type=str,
        help="Kernel type to create from ISD. Acceptable kernel types are "
             "[spk, ck, fk, ik, lsk, mk, pck, sclk]."
    )
    parser.add_argument(
        "-o", "--outfile",
        type=str,
        help="Optional output file.  If not specified, this will be set to "
             "the ISD file name with the appropriate kernel extension."
    )
    parser.add_argument(
        "-d", "--data",
        type=str,
        help="JSON object of keywords for text kernels only."
    )
    parser.add_argument(
        "-c", "--comment",
        required="--semiminor" in sys.argv or "-b" in sys.argv,
        type=str,
        default=None,
        help="Optional comment string to append to the kernel."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Optional boolean flag on overwriting an existing kernel."
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Enable web SpiceQL search."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display information as program runs."
    )
    args = parser.parse_args()

    log_level = logging.ERROR
    if args.verbose:
        log_level = logging.INFO

    logger.setLevel(log_level)

    try:
        isd_to_kernel(isd_file=args.isd_file,
                        kernel_type=args.kernel_type,
                        outfile=args.outfile,
                        data=args.data,
                        overwrite=args.overwrite,
                        use_web=args.web,
                        log_level=log_level)
    except Exception as err:
        sys.exit(f"Could not complete isd_to_kernel task: {err}")

    
def spk_comment(outfile: str,
                segment_id: str,
                start_time: str,
                end_time: str,
                instrument_id: str,
                target_body: str,
                target_name: str,
                center_body: str,
                center_name: str,
                reference_frame: str,
                records: int,
                degree: int,
                kernels: dict,
                comment: str = ""):
    """
    Generates a formatted metadata header for an SPK file.

    The resulting string follows a standardized template containing pedigree, 
    usage notes, and a segment summary. This header is typically written to 
    the comment area of the binary SPK kernel.

    Parameters
    ----------
        outfile : str
            Output kernel file
        segment_id : str
            Unique identifier for the data segment
        start_time : str
            Ephemeris start time.
        end_time : str
            Ephemeris end time.
        instrument_id : str
            Name of the instrument
        target_body : str
            NAIF integer code for the target body
        target_name : str
            Name of the target
        center_body : str
            NAIF integer code for the center body
        center_name : str
            Name of the center body
        reference_frame : str 
            Reference frame name
        records : int
            Number of states in the kernel
        degree : int
            Polynomial degree used for interpolation
        kernels : dict
            Dictionary of supporting kernels
        comment : str, optional
            Additional user-provided notes to append

    Returns:
        str: A multi-line string formatted as a NAIF SPK comment block.
    """
    current_datetime = datetime.now().isoformat(sep=" ", timespec="seconds")
    spk_comment = f"""****************************************************************************
       USGS ALE Generated SPK Kernel
       Created By:   ALE
       Date Created: {current_datetime}
     ****************************************************************************
     
     
     Position Data in the File
     -----------------------------------------------------------------------
     
           This file contains time ordered array of geometric states 
           (kilometers) and rates of change (kilometers/second) of body
           relative to center, specified relative to frame.
     
     
     Status
     -----------------------------------------------------------------------
     
           This kernel was generated for the purposes of storing C-Smithed
           position updates that may have been generated from ALE processing
           techniques (controlled imaging, jitter analysis, etc...).
     
     
     Pedigree
     -----------------------------------------------------------------------
     
           This file was generated by an automated process.  The ALE
           application 'isd_to_kernel' was given an ISD to parse and extract the
           necessary information to create an SPK file.
     
     
     Angular Rates
     -----------------------------------------------------------------------
     
           This kernel typically contains state vectors of rates of change
           as a function of time but may only contain position vectors.  The
           ephemeris given is for the body moving relative to the center of
           motion.
     
     
     Usage Note
     -----------------------------------------------------------------------
     
           To make use of this file in a typical SPICE based application,
           users must supply at least a leapseconds kernel. This file is
           necessary for time-based conversions.  They should be the same
           kernels that were originally used to initialize the image.
     
           Note that ALE defaults to applying light time and stellar
           abberation correction when computing positions relative to s/c and
           target body.  Currently, this correction should not be utilized
           for kernels created by ALE.  Therefore the computation correcting 
           for light time/stellar abberation is turned off. It should be 
           noted that this option applies to all files
           contained herein.  (ID:USGS_SPK_ABCORR=NONE)
    
           The contents of this kernel are summarized below.
    
    User Comments
    -----------------------------------------------------------------------
    {comment}

    Segment (by file) Summary\n\
    -----------------------------------------------------------------------

           The following sections describe each segment in this SPK kernel.  
           Each segment is a file in the input list.  Kernels were 
           consolidated using SpiceQL.

    -----------------------------------------------------------------------
        File:        {outfile}
        Segment ID:  {segment_id}
        StartTime:   {start_time}
        EndTime:     {end_time}
        Instrument:  {instrument_id}
        Target Body: {target_body}, {target_name}
        Center Body: {center_body}, {center_name}
        RefFrame:    {reference_frame}
        Records:     {records}
        PolyDegree:  {degree}
        Kernels:     {kernels}
    """
    return spk_comment


def ck_comment(outfile: str,
               segment_id: str,
               start_time: str,
               end_time: str,
               instrument_id: str,
               target_body: str,
               target_name: str,
               center_body: str,
               center_name: str,
               reference_frame: str,
               records: int,
               has_av: bool,
               kernels: dict,
               comment: str = ""):
    """
    Generates a formatted metadata header for an CK file.

    Standardizes the orientation data documentation, including pedigree and 
    usage notes regarding SCLK and LSK requirements for the specific mission.

    Parameters
    ----------
        outfile : str
            Output kernel file
        segment_id : str
            Unique identifier for the data segment
        start_time : str
            Ephemeris start time.
        end_time : str
            Ephemeris end time.
        instrument_id : str
            Name of the instrument
        target_body : str
            NAIF integer code for the target body
        target_name : str
            Name of the target
        center_body : str
            NAIF integer code for the center body
        center_name : str
            Name of the center body
        reference_frame : str 
            Reference frame name
        records : int
            Number of orientations in the kernel
        has_av : bool
            Indicates if angular velocity is included
        kernels : dict
            Dictionary of supporting kernels
        comment : str, optional
            Additional user-provided notes to append

    Returns:
        str: A multi-line string formatted as a NAIF CK comment block.
    """
    current_datetime = datetime.now().isoformat(sep=" ", timespec="seconds")
    ck_comment = f"""****************************************************************************
       USGS ALE Generated CK Kernel
       Created By:   ALE
       Date Created: {current_datetime}
     ****************************************************************************
     
     Orientation Data in the File
    -----------------------------------------------------------------------
     
          This file contains orientation and potentially derived angular
          rates (where possible/specified).
     
     
    Status
    -----------------------------------------------------------------------
     
          This kernel was generated for the purpose of storing C-Smithed
          pointing updates generated through ALE processing techniques
          (control nets, jitter analysis, etc...).  These CK kernels
          are intended to mimick CKs provided by individual mission
          (NAV teams).
     
    Pedigree
    -----------------------------------------------------------------------
     
          This file was generated by an automated process.  The ALE
          application 'isd_to_kernel' was used to create the CK kernel 
          given an ISD.
     
     
    Angular Rates
    -----------------------------------------------------------------------
     
          This kernel may or may not contain angular velocity vectors. Efforts
          are made to preserve and provide angular velocities where they
          originally existed.
     
     
    Usage Note
    -----------------------------------------------------------------------
     
          To make use of this file in a typical SPICE based application,
          you must supply a leapseconds kernel, a mission spacecraft clock
          kernel, and the instrument/spacecraft frame kernel.  These files
          provide the supporting ancillary data to properly query this
          C-kernel for attitude content.  They should be the same kernels that
          were originally used to initialize the image.
    
    User Comments
    -----------------------------------------------------------------------
     
          {comment}
     
    Segment (by file) Summary
    -----------------------------------------------------------------------
     
          The follow sections describe each segment in this CK kernel.  Each
          segment is a file in the input list.  Kernels were consolidated 
          using SpiceQL.

    -----------------------------------------------------------------------
        File:        {outfile}
        Segment ID:  {segment_id}
        StartTime:   {start_time}
        EndTime:     {end_time}
        Instrument:  {instrument_id}
        Target Body: {target_body}, {target_name}
        Center Body: {center_body}, {center_name}
        RefFrame:    {reference_frame}
        Records:     {records}
        HasAV:       {has_av}
        Kernels:     {kernels}
    """
    return ck_comment


def check_env(use_web: bool = False):
    """
    Checks environment setup for SpiceQL.

    When using the web SpiceQL service, no local SPICE data is required: kernel
    searching and any ET->SCLK encoding needed to write binary kernels are done
    server-side.

    When using local data, SpiceQL requires ISISDATA and SPICEQL_CACHE_DIR. ALE
    defaults SPICEQL_CACHE_DIR to $ISISDATA if it is not already set.

    Parameters
    -------
        use_web: bool
            If True, uses USGS Astrogeology's SpiceQL web service.
            Defaults to False.
    """
    if not use_web:
        cache_dir = os.environ.get("SPICEQL_CACHE_DIR")
        if not cache_dir:
            isisdata = os.environ.get("ISISDATA")
            if not isisdata:
                raise Exception("ISISDATA is not set. Point ISISDATA to " \
                                "your local data area.")
            os.environ["SPICEQL_CACHE_DIR"] = isisdata
            logger.info(f"SPICEQL_CACHE_DIR not set; defaulting to ISISDATA [{isisdata}].")


def load_isd(isd_file: os.PathLike) -> dict:
    """
    Read and parse an ISD JSON file into a dictionary.

    Validates that the file is present and is JSON (by extension and content),
    raising a clear exception on any failure.

    Parameters
    ----------
        isd_file : os.PathLike
            Path to the input ISD JSON file.

    Returns
    ----------
        dict: The parsed ISD contents.
    """
    if isd_file is None:
        raise Exception("Missing ISD file.")
    if Path(isd_file).suffix != ".json":
        raise Exception("ISD must be in JSON.")
    with open(isd_file, "r") as f:
        contents = f.read()
    try:
        return json.loads(contents)
    except ValueError:
        raise Exception(f"ISD [{isd_file}] is not valid JSON.")


def isd_to_kernel(
    isd_file: os.PathLike = None,
    kernel_type: str = "mk",
    outfile: os.PathLike = None,
    data: str = None,
    comment: str = None,
    overwrite: bool = False,
    use_web: bool = False,
    log_level=logging.ERROR
):
    """
    Converts ALE Image Support Data (ISD) to SPICE kernels.

    This function orchestrates the extraction of geometric and pointing data 
    from an ISD JSON file, performs necessary time and frame translations 
    via SpiceQL, and writes the resulting data into a binary (SPK, CK) or 
    text-based (IK, FK, etc.) SPICE kernel.

    Parameters
    ----------
        isd_file : os.PathLike, optional
            Path to the input ISD JSON file. Required for binary kernels.
            For text kernels it is optional; when provided, its 'naif_keywords'
            are written to the kernel.
        kernel_type : str
            The type of kernel to create. Defaults to 'mk'.
        outfile : os.PathLike, optional
            The desired output kernel file name/path.
            Defaults to ISD filename + kernel extension.
        data : str, optional
            A JSON string containing keyword-value pairs. For text kernels this
            is required only when no ISD is provided; when both are given, these
            keywords are appended after (and override) the ISD's naif_keywords.
        comment : str, optional 
            Custom user text to include in the kernel comment area.
        overwrite : bool
            If True, deletes an existing outfile path.
            Defaults to False.
        use_web: bool
            If True, uses USGS Astrogeology's SpiceQL web service.
            Defaults to False.
        log_level : int
            Logging severity level. Defaults to logging.ERROR.

    Returns
    ----------
        None: The function writes the kernel directly to the filesystem.
    """
    logging.basicConfig(format="%(message)s", level=log_level)
    logger.setLevel(log_level)

    # Ensure the environment is set up before any SpiceQL calls are made
    if not use_web and psql.Kernel.isBinary(kernel_type):
        check_env(use_web)

    # Default comment if empty
    if comment is None:
        comment = f"Auto-generated comment by ALE at {datetime.now().isoformat(sep=' ', timespec='seconds')}"
    out_comment = comment

    # If outfile is not specified, name output file as same
    # name as isd_file with appropriate kernel file extension
    if outfile is None:
        if psql.Kernel.isBinary(kernel_type):
            if isd_file is None:
                raise Exception("Missing ISD file.")
            elif Path(isd_file).suffix != ".json":
                raise Exception("ISD must be in JSON.")
            outfile = Path(isd_file).with_suffix(psql.Kernel.getExt(kernel_type))
        elif psql.Kernel.isText(kernel_type):
            raise Exception("Must enter an outfile name for text kernels.")
        else:
            raise Exception(f"{psql.Kernel.getExt(kernel_type)}")
    outfile = str(os.path.abspath(outfile))
    logger.info(f"outfile={outfile}")

    # Default, no overwrite
    if os.path.isfile(outfile):
        if overwrite:
            os.remove(outfile)
        else:
            raise Exception(f"Output file [{outfile}] already exists.")

    filename, ext = os.path.splitext(Path(outfile))

    # Check that the outfile extension matches the kernel_type
    # If not, append correct extension and proceed
    expected_ext = psql.Kernel.getExt(kernel_type)
    if ext.lower() != expected_ext.lower():
        outfile = str(Path(filename).with_suffix(expected_ext))
        logger.info(
            f"Extension mismatch: The output file extension [{ext}] does not match "
            f"the expected extension [{expected_ext}] for kernel type [{kernel_type.upper()}]."
            f"The kernel will output to file [{outfile}] instead."
        )

    if psql.Kernel.isBinary(kernel_type):
        # Get properties from isd_file
        isd_dict = load_isd(isd_file)

        # Get common properties from ISD
        naif_keywords = isd_dict[ISD_KEY_NAIF_KEYWORDS]
        body_code = naif_keywords[ISD_KEY_BODY_CODE]

        # Cache instrument position and pointing dictionaries for multiple accesses
        inst_position = isd_dict.get(ISD_KEY_INSTRUMENT_POSITION, {})
        inst_pointing = isd_dict.get(ISD_KEY_INSTRUMENT_POINTING, {})

        # Determine kernel type
        is_spk = psql.Kernel.isSpk(kernel_type)
        is_ck = psql.Kernel.isCk(kernel_type)

        if not (is_spk or is_ck):
            raise Exception(f"Unexpected binary kernel type: {kernel_type}")

        # Validate required section exists
        if is_spk and not inst_position:
            raise Exception(f"ISD [{isd_file}] missing '{ISD_KEY_INSTRUMENT_POSITION}' section required for SPK generation.")
        if is_ck and not inst_pointing:
            raise Exception(f"ISD [{isd_file}] missing '{ISD_KEY_INSTRUMENT_POINTING}' section required for CK generation.")

        # Extract time range - try kernel-specific times first, fall back to other section
        if is_spk:
            start_time = (inst_position.get(ISD_KEY_SPK_TABLE_START) or
                          inst_pointing.get(ISD_KEY_CK_TABLE_START))
            end_time = (inst_position.get(ISD_KEY_SPK_TABLE_END) or
                        inst_pointing.get(ISD_KEY_CK_TABLE_END))
        else:  # is_ck
            start_time = (inst_pointing.get(ISD_KEY_CK_TABLE_START) or
                          inst_position.get(ISD_KEY_SPK_TABLE_START))
            end_time = (inst_pointing.get(ISD_KEY_CK_TABLE_END) or
                        inst_position.get(ISD_KEY_SPK_TABLE_END))

        if not start_time or not end_time:
            raise Exception(f"ISD [{isd_file}] missing time range in both {ISD_KEY_INSTRUMENT_POSITION} and {ISD_KEY_INSTRUMENT_POINTING} sections.")

        # Extract instrument frame code
        if is_spk:
            # For SPK, try multiple locations for frame code
            if ISD_KEY_TIME_DEPENDENT_FRAMES in inst_position:
                inst_frame_code = inst_position[ISD_KEY_TIME_DEPENDENT_FRAMES][0]
            elif ISD_KEY_TIME_DEPENDENT_FRAMES in inst_pointing:
                inst_frame_code = inst_pointing[ISD_KEY_TIME_DEPENDENT_FRAMES][0]
            else:
                # Fall back to deriving from body_code
                inst_frame_code = body_code * 1000 if body_code < 1000 else body_code
        else:  # is_ck
            # For CK, frame code is always in instrument_pointing
            inst_frame_code = inst_pointing[ISD_KEY_TIME_DEPENDENT_FRAMES][0]

        logger.info(f"start_time={start_time}, end_time={end_time}")

        # Verify valid frame and target code
        # FYI - necessary for chandrayaan_m2_nadir_isd.json
        target_code = int(inst_frame_code/1000)
        if target_code == 0:
            target_code = inst_frame_code
            inst_frame_code = inst_frame_code*1000
        logger.info(f"frame_code={inst_frame_code}, target_code={target_code}")
        
        # Get frame and mission names
        # Priority:
        # 1. NAIF keyword: FRAME_<code>_NAME
        # 2. Sensor name: name_sensor
        # 3. Platform name: name_platform
        # 4. Custom combination name: <platform_name>_<sensor_name>
        # FYI, combination name necessary for apolloPanImage_isd.json
        naif_frame_name = next((v for k, v in isd_dict.get(ISD_KEY_NAIF_KEYWORDS, {}).items()
                if k.startswith("FRAME_") and k.endswith("_NAME")), None)
        sensor_name = isd_dict.get(ISD_KEY_NAME_SENSOR)
        platform_name = isd_dict.get(ISD_KEY_NAME_PLATFORM)
        platform_sensor = f"{platform_name}_{sensor_name}"
        frame_candidates = [
            (naif_frame_name, ISD_KEY_NAIF_KEYWORDS),
            (sensor_name, ISD_KEY_NAME_SENSOR),
            (platform_name, ISD_KEY_NAME_PLATFORM),
            (platform_sensor, "platform_sensor")
        ]

        mission_name = None
        for candidate_value, label in frame_candidates:
            if not candidate_value:
                continue
            result = psql.getSpiceqlName(candidate_value)
            if result:
                frame_name = candidate_value
                mission_name = result
                logger.info(f"Resolved mission_name [{mission_name}] using {label} [{frame_name}]")
                break
            else:
                logger.info(f"Frame name [{candidate_value}] from {label} not found in SpiceQL's aliasMap.")

        if not mission_name:
            raise Exception(
                f"Could not find a valid mission name. Checked NAIF keyword [{naif_frame_name}], "
                f"sensor name [{sensor_name}], "
                f"platform name [{platform_name}], "
                f"and custom PLATFORM_SENSOR name [{platform_sensor}]."
            )
        logger.info(f"frame_name={frame_name}, mission_name={mission_name}")

        # Get kernels
        _, kernels = psql.searchForKernelsets(
            spiceqlNames=["base", mission_name],
            startTime=start_time,
            stopTime=end_time,
            ckQualities=["smithed", "reconstructed"],
            spkQualities=["smithed", "reconstructed"],
            useWeb=use_web)
        logger.info(f"kernels={kernels}")

        # Validate kernels were found
        if kernels is None:
            raise Exception(
                f"Could not find kernels for mission [{mission_name}] "
                f"in time range [{start_time}] to [{end_time}]. "
                f"Verify SpiceQL database contains required kernels."
            )

        # Translate codes to name
        target_name, _ = psql.translateCodeToName(target_code, mission_name, use_web, True)
        body_name, _ = psql.translateCodeToName(body_code, mission_name, use_web, True)

        # Create segmentId
        # Note: 40 char limit
        segment_id = f"{mission_name}:{frame_name}"
        if len(segment_id) > 40:
            logger.info(f"Segment ID [{segment_id}] with length {str(len(segment_id))} "
                         "is over the 40 char max limit. Truncating.")
            segment_id = segment_id[:40]
        logger.info(f"segment_id={segment_id}")

        if psql.Kernel.isSpk(kernel_type):
            # Extract SPK-specific data from ISD (using cached inst_position)
            state_positions = inst_position[ISD_KEY_POSITIONS]
            state_times = inst_position[ISD_KEY_EPHEMERIS_TIMES]
            state_velocities = inst_position[ISD_KEY_VELOCITIES]

            if len(state_positions) != len(state_times):
                raise ValueError("Positions and Times length mismatch!")

            records = len(state_positions)

            # Calculate degree for Hermite interpolation (SPK type 13)
            # Degree must be odd and at most min(7, number_of_states-1)
            number_of_states = len(state_positions)  # Number of time points
            degree = min(7, number_of_states - 1)
            # Ensure degree is odd (required for SPK type 13)
            if degree % 2 == 0:
                degree -= 1

            # Get reference frame for SPK
            spk_reference_frame_id = inst_position[ISD_KEY_REFERENCE_FRAME]
            spk_reference_frame = spice.frmnam(spk_reference_frame_id)
            logger.info(f"SPK generation: records={records}, degree={degree}, ref_frame={spk_reference_frame}")
            logger.info(f"  First position: {state_positions[0]}")
            logger.info(f"  First time: {state_times[0]}")
            logger.info(f"  Target: {target_code}, Center: {body_code}")
            
            out_comment = spk_comment(
                outfile=outfile,
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                instrument_id=frame_name,
                target_body=target_code,
                target_name=target_name,
                center_body=body_code,
                center_name=body_name,
                reference_frame=spk_reference_frame,
                records=records,
                degree=degree,
                kernels=kernels,
                comment=comment)
            
            psql.writeSpk(
                outfile,
                state_positions,
                state_times,
                target_code,
                body_code,
                spk_reference_frame,
                segment_id,
                degree,
                state_velocities,
                out_comment
            )
            logger.info(f"SPK written: target={target_code}, center={body_code}, degree={degree}, records={records}")
        elif psql.Kernel.isCk(kernel_type):
            # Extract CK-specific data from ISD (using cached inst_pointing)
            inst_pt_quaternions = inst_pointing[ISD_KEY_QUATERNIONS]
            inst_pt_times = inst_pointing[ISD_KEY_EPHEMERIS_TIMES]

            # Angular velocities
            has_av = True
            inst_pt_velocities = inst_pointing.get(ISD_KEY_ANGULAR_VELOCITIES)
            if inst_pt_velocities is None:
                logger.info(f"ISD [{isd_file}] does not have angular velocities.")
                inst_pt_velocities = []
                has_av = False

            records = len(inst_pt_quaternions)

            # Get reference frame for CK (last frame in time_dependent_frames chain)
            ck_reference_frame_id = inst_pointing[ISD_KEY_TIME_DEPENDENT_FRAMES][-1]
            ck_reference_frame = spice.frmnam(ck_reference_frame_id)
            logger.info(f"ck_reference_frame={ck_reference_frame}")

            # Get sclks and lsk
            if "sclk" in kernels:
                sclk_kernels = ",".join(kernels["sclk"])
            else:
                raise Exception(f"Could not find SCLKs for [{isd_file}].")
            if "lsk" in kernels:
                lsk_kernel = str(kernels["lsk"][0])
            else:
                raise Exception(f"Could not find LSK for [{isd_file}].")
            logger.info(f"sclk_kernels={sclk_kernels}, lsk_kernel={lsk_kernel}")

            # Writing a CK requires encoding the ephemeris times to SCLK ticks,
            # which normally furnishes the SCLK/LSK from a local SPICE data dir.
            # In web mode we do that encoding server-side (etsToSclkTicks) and
            # hand writeCk the pre-encoded ticks so no local data dir is needed.
            ck_times = inst_pt_times
            sc_id = int(inst_frame_code / 1000)
            ck_times, _ = psql.doubleEtsToSclkTicks(sc_id, inst_pt_times, mission_name, use_web)
            logger.info(f"Encoded {len(ck_times)} ETs to SCLK ticks via web for sc={sc_id}.")

            out_comment = ck_comment(
                outfile=outfile,
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                instrument_id=frame_name,
                target_body=target_code,
                target_name=target_name,
                center_body=body_code,
                center_name=body_name,
                reference_frame=ck_reference_frame,
                records=records,
                has_av=has_av,
                kernels=kernels,
                comment=comment)

            psql.writeCk(
                outfile,
                inst_pt_quaternions,
                ck_times,
                inst_frame_code,
                ck_reference_frame,
                segment_id,
                inst_pt_velocities,
                out_comment
            )
    elif psql.Kernel.isText(kernel_type):

        def is_valid_json(json_str):
            try:
                json.loads(json_str)
                return True
            except ValueError:
                return False

        # Text kernel keywords can come from an ISD's naif_keywords, from the
        # user-provided data payload, or both. When an ISD is given, its
        # naif_keywords are used; any user data is appended after (and takes
        # precedence over) them. When no ISD is given, the user must supply data.
        keywords = {}

        if isd_file is not None:
            naif_keywords = load_isd(isd_file).get(ISD_KEY_NAIF_KEYWORDS, {})
            if not naif_keywords:
                logger.info(f"ISD [{isd_file}] has no '{ISD_KEY_NAIF_KEYWORDS}' to add.")
            keywords.update(naif_keywords)

        if data is not None:
            if not is_valid_json(data):
                raise Exception("The 'data' payload is not valid JSON.")
            keywords.update(json.loads(data))

        if not keywords:
            raise Exception(
                f"Must provide an ISD with '{ISD_KEY_NAIF_KEYWORDS}' and/or JSON "
                f"data to generate text kernel [{outfile}]."
            )

        logger.info(f"Generating text kernel type [{kernel_type}]")
        psql.writeTextKernel(
            outfile,
            kernel_type,
            keywords,
            out_comment
        )
    else:
        raise Exception(f"Could not generate kernel [{outfile}] for kernel type [{kernel_type}].")
    