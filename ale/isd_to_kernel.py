import argparse
import json
import logging
import os, sys
import pyspiceql as psql
import spiceypy as spice

from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


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
        kernel_type : str
            The type of kernel to create. Defaults to 'mk'.
        outfile : os.PathLike, optional
            The desired output kernel file name/path.
        data : str, optional
            A JSON string containing keyword-value pairs. Required for text kernels.
        comment : str, optional 
            Custom user text to include in the kernel comment area.
        overwrite : bool
            If True, deletes an existing outfile path.
            Defaults to False.
        log_level : int
            Logging severity level. Defaults to logging.ERROR.

    Returns
    ----------
        None: The function writes the kernel directly to the filesystem.
    """
    logging.basicConfig(format="%(message)s", level=log_level)
    logger.setLevel(log_level)

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
        with open(isd_file, 'r') as f:
            isd_data = f.read()
        
        # ISD data
        isd_dict = json.loads(isd_data)

        # spk properties
        state_positions = isd_dict["instrument_position"]["positions"]     
        state_times = isd_dict["instrument_position"]["ephemeris_times"] 
        state_velocities = isd_dict["instrument_position"]["velocities"]

        # ck properties
        inst_pt_quaternions = isd_dict["instrument_pointing"]["quaternions"]
        inst_pt_times = isd_dict["instrument_pointing"]["ephemeris_times"]

        # angular velocities
        has_av = True
        inst_pt_velocities = isd_dict.get("instrument_pointing")["angular_velocities"]
        if inst_pt_velocities is None:
            logger.info(f"ISD [{isd_file}] does not have angular velocities.")
            inst_pt_velocities = []
            has_av = False
        
        # Comment properties
        body_code = isd_dict["naif_keywords"]["BODY_CODE"]             
        body_frame_code = isd_dict["naif_keywords"]["BODY_FRAME_CODE"]       
        start_time = isd_dict["instrument_pointing"]["ck_table_start_time"]
        end_time = isd_dict["instrument_pointing"]["ck_table_end_time"]
        inst_frame_code = isd_dict["instrument_pointing"]["time_dependent_frames"][0]
        target_code = int(inst_frame_code/1000)
        records = len(state_positions)
        logger.info(f"start_time={start_time}, end_time={end_time}")
        
        # Get frame and mission names
        # Priority:
        # 1. NAIF keyword: FRAME_<code>_NAME
        # 2. Sensor name: name_sensor
        # 3. Platform name: name_platform
        # 4. Custom combination name: <platform_name>_<sensor_name>
        # FYI, combination name necessary for apolloPanImage_isd.json
        platform_sensor = f"{isd_dict.get('name_platform')}_{isd_dict.get('name_sensor')}"
        frame_candidates = [
            (next((v for k, v in isd_dict.get("naif_keywords", {}).items() 
                if k.startswith("FRAME_") and k.endswith("_NAME")), None), "naif_keywords"),
            (isd_dict.get("name_sensor"), "name_sensor"),
            (isd_dict.get("name_platform"), "name_platform"),
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
                logger.info(f"Frame name [{candidate_value}] from {label} not found in SpiceQL aliasMap.")

        if not mission_name:
            raise Exception(
                f"Could not find a valid mission name. Checked NAIF keywords, "
                f"sensor name [{isd_dict.get('name_sensor')}], "
                f"platform name [{isd_dict.get('name_platform')}], "
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

        # Translate codes to name
        target_name, _ = psql.translateCodeToName(target_code, mission_name, use_web, True)
        body_name, _ = psql.translateCodeToName(body_code, mission_name, use_web, True)
        
        # Calculate degree
        number_of_states = len(state_positions[0])
        degree_min = min(7, number_of_states-1)
        degree_output = (((degree_min - 1) / 2) * 2) + 1    
        if degree_output%2 == 0 or degree_output >= degree_min:
            degree = degree_output - 1
            degree = int(degree)
        
        # Create segmentId
        # Note: 40 char limit
        segment_id = f"{mission_name}:{frame_name}"
        if len(segment_id) > 40:
            logger.info(f"Segment ID [{segment_id}] with length {str(len(segment_id))} " 
                         "is over the 40 char max limit. Truncating.")
            segment_id = segment_id[:40]
        logger.info(f"segment_id={segment_id}")

        # Get referenceFrame
        reference_frame_id = isd_dict["instrument_position"]["reference_frame"]
        reference_frame = spice.frmnam(reference_frame_id)
        logger.info(f"reference_frame={reference_frame}")
        
        if psql.Kernel.isSpk(kernel_type):
            if len(state_positions) != len(state_times):
                raise ValueError("Positions and Times length mismatch!")
            
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
                reference_frame=reference_frame,
                records=records,
                degree=degree,
                kernels=kernels,
                comment=comment)
            psql.writeSpk(
                outfile,
                state_positions,
                state_times,
                body_code,
                body_frame_code,
                reference_frame,
                segment_id,
                degree,
                state_velocities,
                out_comment
            )
        elif psql.Kernel.isCk(kernel_type):  
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
                reference_frame=reference_frame,
                records=records,
                has_av=has_av,
                kernels=kernels,
                comment=comment)
            psql.writeCk(
                outfile,
                inst_pt_quaternions,
                inst_pt_times,
                inst_frame_code,
                reference_frame,
                segment_id,
                sclk_kernels,
                lsk_kernel,
                inst_pt_velocities,
                out_comment
            )
    elif psql.Kernel.isText(kernel_type):

        def is_valid_json(json_str):
            try:
                json.loads(json_str)
                return True
            except ValueError as e:
                return False

        if data is None:
            raise Exception(f"Must enter JSON keywords to generate kernel [{outfile}].")
        elif not is_valid_json(data):
            raise Exception("The 'data' payload is not valid JSON.")
        
        data = json.loads(data)

        logger.info(f"Generating text kernel type [{kernel_type}]")
        psql.writeTextKernel(
            outfile,
            kernel_type,
            data,
            out_comment
        )
    else:
        raise Exception(f"Could not generate kernel [{outfile}] for kernel type [{kernel_type}].")
    