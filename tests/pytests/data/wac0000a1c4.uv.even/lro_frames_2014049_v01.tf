KPL/FK


LRO Frame Definitions Kernel
==============================================================================

   This frame kernel contains the LRO spacecraft and science instrument
   frame definitions. This frame kernel also contains name - to - NAIF
   ID mappings for LRO science instruments and s/c structures (see the
   last section of the file.)


Version and Date
--------------------------------------------------------
  
   Version 2014049 -- January 14, 2014 -- Emerson Speyerer
      
      Added frames 8563[1-5] and 8564[1-2] for each LROC WAC filter. These 
      ids are required to identify the FOV of each band and to use the 
      improved camera model based on on-orbit calibration results.
      
      Changed the LROC NAC frames (85600 and 85610) and WAC frame (85620)
      to be CK dependent.

   Version 2014027 (Third Archive release) -- Jan 27, 2014 

      Ralph Casasanta with support from external groups including
      SOCs and ACS.  External groups determined the Euler Angle
      sign values were incorrect as listed.  Updated sign values for
      all components for primary and secondary Star Trackers because
      spice CK uses opposite sign convention.

   Version 2012255 (Second Archive release) -- Sept 12, 2012 
      
      Ralph Casasanta with support from GNC, ACS, and FSW groups
      Updated the Star tracker alignments after new calibrations

   Version 2010214 (First Archive Release) -- August 4, 2010 -- Boris Semenov

      Incorporated LRO_LROCWAC_VIS and LRO_LROCWAC_UV frames and
      name/ID mapping used by the LROC team.

      Changed angles in the LOLA frame definition to zeros to make the
      frame co-aligned with the spacecraft frame because LOLA detector
      view vectors used by the LOLA team are defined in the spacecraft
      frame.

      Removed TKFRAME_-85500_SC_MAPPING_CG_LOC keyword and accompanying
      description; this keyword is now provided in the LOLA IK file,

      Changed description of the HGA and SA frames to refer to the
      principal "view" directions.

      Added ASCII diagram illustrating the spacecraft frame.
       
      Wrapped all paragraphs to 78 character line length.

      Added some blank lines to improve readability.

      Replaced incomplete IK names and TBDs in the References section
      with the generic reference "latest".

      Added Casasanta and Semenov to contacts.

      Replaced version numbers (0.0, 0.2, ... 2.0) in the "Version and
      Date" section with the YYYYDOY-based numbers (2007318, 2008127,
      ... 2010036) to improve correlation with the FK file names.

      Spell-checked. Removed TABs.

   Version 2010036 (Post-Launch Release) -- February 5, 2010 --
                         Ralph Casasanta with inputs from LAMP and LOLA SOCs
  
      Updated frames information for LAMP and LOLA instrument
      alignments as calculated during calibration activities conducted
      during the LRO commissioning phase (approximately 24 June through
      14 September 2009). The updated frames are derived from on-orbit
      calibration from the LRO commissioning orbit(30 x 216 Km). Also
      modified "origin to reference frame" information to reflect
      metric units (in centimeters) rather than inches.

  Version 2009168 (Pre-Launch release) -- June 15, 2009 -- Leslie Hartz, 
                                                           Ralph Casasanta
  
      Updated frames information for instrument alignment as calculated
      during calibration activities conducted by LRO Project personnel during
      from April 2009. The updated frames are based on 
      LRO optical measurements or LRO mechanical drawings.
      
      The spacecraft body orbital SPK file is modeled as a point-mass.

      Currently, the offsets noted within this kernel are referenced in
      inches; we will modify this information to SI units when we
      release the post-launch version sometime after the commissioning
      and calibration portion of the mission.
      This will occur sometime after the L + 60D time-frame or whenever 
      LRO completes the calibration and commissioning phase. 

   Version 2008127 (draft) -- May 06, 2008 -- Ralph Casasanta, Boris Semenov

      Modified HGA and SA IDs for the CK identifier to indicate we use
      the main object structure and not to the articulating booms.
      NOTE: Still does not contain a description for any of the frames.

   Version 2007318 (draft) -- November 14, 2007 -- Boris Semenov

      Added HGA and SA definitions and changed their IDs and
      relationship. Fixed frame ID that is a part of the keyword name
      in numerous fixed offset frames. Added name-ID mapping keywords.
      Minor revisions to the comments. Still does not contain a
      description for any of the frames.

   Version 2007318 (draft) -- November 14, 2007 -- Eric B. Holmes

      Initial Release. Contains Euler angles from LRO I-Kernel
      files. Does not contain a description for any of the frames.


References
--------------------------------------------------------

   1. C-kernel Required Reading

   2. Kernel Pool Required Reading

   3. Frames Required Reading

   4. Cosmic Ray Telescope for the Effects of Radiation (CRaTER) I-Kernel
      file; latest

   5a. Diviner Lunar Radiometer Experiment detector layout  
       relative to the instrument fixed reference frame LRO_DLRE; 
       "lro_dlre_frames_2009160_v01.tf"; Version 1.0, Jia Zong, June 09, 2009

   5b Diviner Lunar Radiometer Experiment (DLRE) I-Kernel File
      "lro_dlre_2009160_v01.ti"; Version 1.0, Jia Zong, May 21 2009

   6. Lyman-Alpha Mapping Project (LAMP) I-Kernel file; latest

   7. Lunar Explorer Neutron Detector (Lend) I-Kernel file; latest

   8. Lunar Orbiter Laser Altimeter (LOLA) I-Kernel file; latest

   9. Lunar Reconnaissance Orbiter Camera Instrument Kernel for NAC-L
      NAC-R and the WAC; latest

  10. Mini RF I-Kernel file; latest

  11. Primary Star Tracker (STARP) using the LRO Attitude Control System
      (ACS) Alignment and Coordinate Systems document (431-SPEC-000653)

  12. Secondary Star Tracker (STARS) using the LRO Attitude Control System
      (ACS) Alignment and Coordinate Systems document (431-SPEC-000653)

  13. Miniature Inertial Measurement Unit (MIMU) I-Kernel file; latest


Contact Information
--------------------------------------------------------

   Eric B. Holmes, Code 591, (301)-286-4046, eric.b.holmes@nasa.gov
   Ralph T. Casasanta, Code 730, (301)-614-5321, Ralph.T.Casasanta@nasa.gov
   Boris Semenov, NAIF, (818)-354-8136, Boris.Semenov@jpl.nasa.gov


Implementation Notes
--------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must ``load'' the kernel, normally during
   program initialization. The SPICELIB routine FURNSH loads a kernel
   file into the pool as shown below.

      CALL FURNSH ( 'frame_kernel_name; )       -- FORTRAN
      furnsh_c ( "frame_kernel_name" );         -- C
      cspice_furnsh, "frame_kernel_name"        -- IDL
      cspice_furnsh ( 'frame_kernel_name' );    -- MATLAB

   This file was created and may be updated with a text editor or word
   processor.


LRO Frames
--------------------------------------------------------

   The following LRO frames are defined in this kernel file:

        Frame Name                   Relative to            Type    NAIF ID
   =========================   =========================   =======  =======

   Spacecraft Bus and Spacecraft Structure Frames:
   -----------------------------------------------
      LRO_SC_BUS               rel.to J2000                CK       -85000
      
      LRO_STARP                rel.to SC_BUS               FIXED    -85010
      LRO_STARS                rel.to SC_BUS               FIXED    -85011

      LRO_MIMU                 rel.to SC_BUS               FIXED    -85012

      LRO_HGA                  rel.to SC_BUS               CK       -85020

      LRO_SA                   rel.to SC_BUS               CK       -85030

   Instrument Frames:
   ------------------
      LRO_CRATER               rel.to SC_BUS               FIXED    -85100

      LRO_DLRE                 rel.to SC_BUS               FIXED    -85200

      LRO_LAMP                 rel.to SC_BUS               FIXED    -85300

      LRO_LEND                 rel.to SC_BUS               FIXED    -85400

      LRO_LOLA                 rel.to SC_BUS               FIXED    -85500

      LRO_LROCNACL             rel.to SC_BUS               CK       -85600
      LRO_LROCNACR             rel.to SC_BUS               CK       -85610

      LRO_LROCWAC              rel.to SC_BUS               CK       -85620
      LRO_LROCWAC_VIS          rel.to LRO_LROCWAC          FIXED    -85621
      LRO_LROCWAC_UV           rel.to LRO_LROCWAC          FIXED    -85626
      LRO_LROCWAC_VIS_FILTER_1 rel.to LRO_LROCWAC_VIS      FIXED    -85631
      LRO_LROCWAC_VIS_FILTER_2 rel.to LRO_LROCWAC_VIS      FIXED    -85632
      LRO_LROCWAC_VIS_FILTER_3 rel.to LRO_LROCWAC_VIS      FIXED    -85633
      LRO_LROCWAC_VIS_FILTER_4 rel.to LRO_LROCWAC_VIS      FIXED    -85634
      LRO_LROCWAC_VIS_FILTER_5 rel.to LRO_LROCWAC_VIS      FIXED    -85635
      LRO_LROCWAC_UV_FILTER_1  rel.to LRO_LROCWAC_UV       FIXED    -85641
      LRO_LROCWAC_UV_FILTER_2  rel.to LRO_LROCWAC_UV       FIXED    -85642

      LRO_MINIRF               rel.to SC_BUS               FIXED    -85700


LRO Frames Hierarchy
--------------------------------------------------------

   The diagram below shows LRO frames hierarchy:


                               "J2000" INERTIAL
               +--------------------------------------------+
               |                                            |
               | <--ck                                      | <--pck
               |                                            |
               |                                            V
               |                                        "IAU_EARTH"
               |                                        EARTH BFR(*)
               |                                        ------------
               |
               |
               |
               |
               |
               |
               V
          "LRO_SC_BUS"
      +----------------------------------------------------------------------+
      |              |                |              |            |          |
      |<--fixed      |<--fixed        |<-fixed       |<--fixed    |<--fixed  |
      |              |                |              |            |          |
      V              V                V             V             V          |
   "LRO_CRATER"   "LRO_DLRE"       "LRO_LAMP"     "LRO_LEND"   "LRO_LOLA"    |
   ------------   ----------       ----------     ----------   ----------    |
                                                                             |
   +-------------------------------------------------------------------------+
   |    |               |               |              |
   |    |<--ck          |<--ck          |<--ck         |<-fixed
   |    |               |               |              |
   |    V               V               V              V
   | "LRO_LROCNACR"  "LRO_LROCNACL"  "LRO_LROCWAC"  "LRO_MINIRF"
   | --------------  --------------  -------------  ------------
   |                                       |
   |                                       |
   |            +--------------------------+--------------+
   |            |                                         |
   |            |<--fixed                                 |<--fixed
   |            |                                         | 
   |      "LRO_WAC_VIS"                              "LRO_WAC_UV"
   |      -------------                              ------------
   |            |                                         |
   |            +----> "LRO_WAC_VIS_FILTER_1" (fixed)     |
   |            |                                         |
   |            +----> "LRO_WAC_VIS_FILTER_2" (fixed)     |
   |            |                                         |
   |            +----> "LRO_WAC_VIS_FILTER_3" (fixed)     |
   |            |                                         |
   |            +----> "LRO_WAC_VIS_FILTER_4" (fixed)     |
   |            |                                         |
   |            +----> "LRO_WAC_VIS_FILTER_5" (fixed)     |
   |                                                      |
   |                                                      |
   |                                                      |
   |                    "LRO_WAC_UV_FILTER_1" (fixed)<----+
   |                                                      |
   |                    "LRO_WAC_UV_FILTER_2" (fixed)<----+
   |
   |
   +--------------------------------------------------------------
     |               |            |             |              |
     |<--fixed       |<--fixed    |<--fixed     |<--ck         |<--ck         
     V               V            V             V              V
  "LRO_STARP"   "LRO_STARS"   "LRO_MIMU"     "LRO SA"       "LRO HGA"
  -----------   -----------   ----------     -------        ---------   


Spacecraft Bus Frame
--------------------------------------------------------

   The spacecraft bus frame is defined by the spacecraft design as
   follows:

      *  +X axis is in the direction of the velocity vector half the
         year. The other half of the year, the +X axis is opposite the
         velocity vector ;

      *  +Y axis is the anti-sun side of the spacecraft;

      *  +Z axis is in the in the nadir direction, instrument boresight
         direction;

      *  the origin of this frame is at the center of the spacecraft to
         launch vehicle interface;

   This diagram illustrates the spacecraft bus frame:

                               ^
                               | S/c velocity
                               |

                              _
                         .---'o| CRaTER
                         |     |
                         |_____|
                         |     |             .----------------
                   .-----------------------. |               ' 
                   |   .'\                 | |               `
             ._____| .' .' DLRE            | |               '
    LROC/WAC |  o  | \.'        .---------.| |               `
             |   - |            |         || |  Solar Array  '
       LOLA  | o|o||            |         || |               `
             |   - |            |         || |               '
             | .-. |            |         || |               `
   LROC/NACs || o ||            |         || |               '
           .-| `-' |            |         || |               ` 
      LAMP | | .-. |            | Mini RF || |               '
           |o|| o ||            |         ||  ===============' 
           `-| `_' |            |         || |               '
             `-----|            |         || |               `
              .----|            |         || |               '
              |o o |            |         || |               `
              |o o |      +Xsc ^|         || |               ' 
              `----|           |`---------'|=|               '
           LEND    `-----------|-----------' |               `
                           `.__|__.' HGA     |               '
                         <-----o             |               `
                     +Ysc       +Zsc         |               '
                                              ~ ~ ~ ~ ~ ~ ~ ~

                               |
                               | S/c velocity
                               V

                                      +Zsc is out of the page.

   Spacecraft bus attitude with respect to an inertial frame is
   provided by a C kernel (see [1] for more information).

   \begindata

      FRAME_LRO_SC_BUS         = -85000
      FRAME_-85000_NAME        = 'LRO_SC_BUS'
      FRAME_-85000_CLASS       = 3
      FRAME_-85000_CLASS_ID    = -85000
      FRAME_-85000_CENTER      = -85
      CK_-85000_SCLK           = -85
      CK_-85000_SPK            = -85

   \begintext


Cosmic Ray Telescope for the Effects of Radiation Frame
--------------------------------------------------------

   The CRaTER instrument frame is defined by the instrument design as
   follows:

      *  +X axis is parallel to the spacecraft +X axis;

      *  +Y axis is parallel to the spacecraft +Y axis;

      *  +Z axis is the boresight of the nadir telescope and parallel
         to the +Z axis of the spacecraft;

      *  the origin of this frame is at the spacecraft to instrument
         interface, 276.86, 25.40, 55.88; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern.  There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame is fixed with respect to the
   spacecraft frame. This frame is based on LRO mechanical drawings.  
   It was not verified by measurement.

   \begindata

      FRAME_LRO_CRATER         = -85100
      FRAME_-85100_NAME        = 'LRO_CRATER'
      FRAME_-85100_CLASS       = 4
      FRAME_-85100_CLASS_ID    = -85100
      FRAME_-85100_CENTER      = -85
      TKFRAME_-85100_SPEC      = 'ANGLES'
      TKFRAME_-85100_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85100_ANGLES    = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85100_AXES      = ( 1,   2,   3   )
      TKFRAME_-85100_UNITS     = 'DEGREES'

   \begintext


Diviner Lunar Radiometer Experiment Frame
--------------------------------------------------------

   The DLRE instrument frame is defined by the instrument alignment 
   cube face normals measured in the LRO_SC_BUS:

      *  +X axis is the normal of instrument alignment cube face 1;

      *  +Y axis is the normal of instrument alignment cube face 2;

      *  +Z axis is the cross product of X and Y;

      *  the origin of this frame is at the spacecraft to instrument
         interface, 236.22, 45.72, 60.96; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern.  There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles provided in the frame
   definition below are extracted from [5a]. The following transforms 
   convert directions from LRO_DLRE into LRO_SC_BUS.

   \begindata

      FRAME_LRO_DLRE           = -85200
      FRAME_-85200_NAME        = 'LRO_DLRE'
      FRAME_-85200_CLASS       = 4
      FRAME_-85200_CLASS_ID    = -85200
      FRAME_-85200_CENTER      = -85
      TKFRAME_-85200_SPEC      = 'MATRIX'
      TKFRAME_-85200_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85200_MATRIX    = ( 
                                -0.867153123 
                                -0.498040818 
                                -0.000898401621
                                -0.498028823
                                 0.867025903
                                 0.0152766628
                                -0.00682946414
                                 0.0136946357 
                                -0.999882902
                                )

   \begintext


Lyman-Alpha Mapping Project Frame
--------------------------------------------------------

   The LAMP instrument frame is defined by the instrument design as
   follows:

      *  +X axis is parallel to the +X axis of the spacecraft;

      *  +Y axis is rotated slightly from the +Y axis of the spacecraft;

      *  +Z axis is the boresight of the instrument and is nearly
         parallel to the +Z axis of the spacecraft;

      *  the origin of this frame is at the instrument to spacecraft
         interface, 142.24, 109.22, 20.32; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern. There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles, as documented in the frame
   definition below. 

   To determine the pointing of LAMP, we conducted the LAMP-403 scans,
   which are raster observations of the stars gam Gru and zet Cas
   across the open LAMP aperture. The LAMP instrument team decided that
   the following information would define the boresight of the
   instrument such that when the boresight was pointed towards a point
   source, the target spectrum would appear primarily in detector row
   14 (zero-indexed). The end result of the LAMP-403 analysis is that
   to go from the spacecraft frame to the LAMP instrument frame it is
   necessary to rotate:

       -0.01516987 degrees about Y (offset in cross-slit direction)
        0.57339189 degrees about X (offset in along-slit direction)
        0.13802984 degrees about Z (rotation of slit about boresight)

   \begindata

      FRAME_LRO_LAMP           = -85300
      FRAME_-85300_NAME        = 'LRO_LAMP'
      FRAME_-85300_CLASS       = 4
      FRAME_-85300_CLASS_ID    = -85300
      FRAME_-85300_CENTER      = -85
      TKFRAME_-85300_SPEC      = 'ANGLES'
      TKFRAME_-85300_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85300_ANGLES    = ( -0.015169872, 0.57339189, 0.13802984 )
      TKFRAME_-85300_AXES      = (  2,   1,   3 )
      TKFRAME_-85300_UNITS     = 'DEGREES'

   \begintext


Lunar Explorer Neutron Detector (LEND) Frame
--------------------------------------------------------

   The LEND instrument frame is defined by the instrument design as
   follows:

      *  +X axis is parallel to the +X axis of the spacecraft;

      *  +Y axis is parallel to the +Y axis of the spacecraft;

      *  +Z axis is parallel to the +Z axis of the spacecraft and it in
         the same direction as the LEND collimators;

      *  the origin of this frame is at the spacecraft to instrument
         interface, 45.72, 81.28, 60.96; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern. There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles, as documented in the frame
   definition below, are based on LRO mechanical drawings.

   \begindata

      FRAME_LRO_LEND           = -85400
      FRAME_-85400_NAME        = 'LRO_LEND'
      FRAME_-85400_CLASS       = 4
      FRAME_-85400_CLASS_ID    = -85400
      FRAME_-85400_CENTER      = -85
      TKFRAME_-85400_SPEC      = 'ANGLES'
      TKFRAME_-85400_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85400_ANGLES    = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85400_AXES      = ( 1,   2,   3   )
      TKFRAME_-85400_UNITS     = 'DEGREES'

   \begintext


Lunar Orbiter Laser Altimeter (LOLA) Frame
--------------------------------------------------------

   The LOLA instrument frame is defined by the instrument design as
   follows:

      *  +X axis is nominally parallel to the +X axis of the spacecraft;

      *  +Y axis is nominally parallel to the +Y axis of the spacecraft;

      *  +Z axis is nominally parallel to the spacecraft +Z axis and is
         laser channel 1;

      *  The LOLA offset is identified as the distance from the
         spacecraft bus frame to the LOLA reference cube, which is at
         the base of the laser beam expander telescope, plus the
         additional 0.15174 m to the top of the LOLA telescope along
         the z-axis. The following values are the x-, y-, and z-
         components for this offset and are listed in meters.
         (2.04608938, 0.96087438, 0.52301394)

   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles, as documented in the frame
   definition below, are based on LRO optical measurements.

   The components of the LOLA Cube in the s/c Frame as a matrix are:

         ( 0.99998757, 0.00477017, 0.00145188,
          -0.0047642,  0.99998032, -0.0040808,
          -0.0014713, 0.00407379, 0.99999062 )

   However this alignment data is not used in the LOLA frame definition.
   Instead the LOLA is defined to be co-aligned with the spacecraft
   frame because LOLA detector view vectors used by the LOLA team and
   provided in the FOV definitions in the LOLA IK file are defined in
   the spacecraft frame.

   \begindata

      FRAME_LRO_LOLA           = -85500
      FRAME_-85500_NAME        = 'LRO_LOLA'
      FRAME_-85500_CLASS       = 4
      FRAME_-85500_CLASS_ID    = -85500
      FRAME_-85500_CENTER      = -85
      TKFRAME_-85500_SPEC      = 'ANGLES'
      TKFRAME_-85500_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85500_ANGLES    = ( 0, 0, 0, )
      TKFRAME_-85500_AXES      = ( 1, 2, 3  )
      TKFRAME_-85500_UNITS     = 'DEGREES'

   \begintext


Lunar Reconnaissance Orbiter Camera-Narrow Angle Camera 1 (LROCNACL) Frame
--------------------------------------------------------

   The location of the LROC NACL instrument is provided by [ref 9];
   this provides all information for the NAC-L, NAC-R, and the WAC.
   
   The LROC NACL instrument frame is defined by the instrument design as
   follows:

      *  +X axis is nominally parallel to the +X axis of the spacecraft;

      *  +Y axis is nominally parallel to the +Y axis of the spacecraft;

      *  +Z axis is the boresight of the NACL;

      *  the origin of this frame is at the instrument to spacecraft
         interface, 134.62, 88.90, -17.78; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern. There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame is variable with respect to the
   spacecraft frame, and depends on temperature.  This pointing
   information is provided in CK files produced by the LROC SOC.

   \begindata

      FRAME_LRO_LROCNACL       = -85600
      FRAME_-85600_NAME        = 'LRO_LROCNACL'
      FRAME_-85600_CLASS       = 3
      FRAME_-85600_CLASS_ID    = -85600
      FRAME_-85600_CENTER      = -85
      TKFRAME_-85600_RELATIVE  = 'LRO_SC_BUS'
      CK_-85600_SCLK           = -85
      CK_-85600_SPK            = -85

   \begintext


Lunar Reconnaissance Orbiter Camera-Narrow Angle Camera 2 (LROCNACR) Frame
--------------------------------------------------------

   The location of the LROC NACR instrument is provided by [ref 9];
   this provides all information for the NAC-L, NAC-R, and the WAC.

   The LROC NACR reference frame is rotated 180 degrees about the Z
   axis. This rotation is performed first. The LROC NACR instrument
   frame is defined by the instrument design as follows:

      *  +X axis is nominally parallel to the spacecraft -X axis;

      *  +Y axis is nominally parallel to the spacecraft -Y axis;

      *  +Z axis is the boresight of the NACR camera and is
         approximately in the +Z axis of the spacecraft;

      *  the origin of this frame is at the spacecraft to instrument
         interface, 101.60, 88.90, -17.78; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern. There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame is variable with respect to the
   spacecraft frame, and depends on temperature.  This pointing 
   information is provided in CK files produced by the LROC SOC.

   \begindata

      FRAME_LRO_LROCNACR       = -85610
      FRAME_-85610_NAME        = 'LRO_LROCNACR'
      FRAME_-85610_CLASS       = 3
      FRAME_-85610_CLASS_ID    = -85610
      FRAME_-85610_CENTER      = -85
      TKFRAME_-85610_RELATIVE  = 'LRO_SC_BUS'
      CK_-85610_SCLK           = -85
      CK_-85610_SPK            = -85

   \begintext



Lunar Reconnaissance Orbiter Camera-Wide Angle Camera (LROCWAC) Frames
--------------------------------------------------------

   The location of the LROC WAC instrument is provided by [ref 9];
   this provides all information for the NAC-L, NAC-R, and the WAC.

   The LROC WAC instrument frame is defined by the instrument design as
   follows:

      *  +X axis is parallel to the spacecraft +X axis;

      *  +Y axis is parallel to the spacecraft +Y axis;

      *  +Z axis is parallel to the spacecraft +Z axis and is the
         boresight of the camera;

      *  the origin of this frame is at the spacecraft to instrument
         interface, 200.66, 106.68, 50.80; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern. There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame (85620) is variable with respect to the
   spacecraft frame, and depends on temperature.  This pointing information
   is provided in CK files produced by the LROC SOC.

   A separate fixed reference frame is defined for each of the two WAC 
   channels (UV and VIS) since the camera has separate optics for each. In
   addition, each WAC band has an independent frame.

   The follwoing table provides a listing of the NAIF ID associated with the 
   LROC WAC:


      NAIF ID  |         Frame Name          |    Notes
     ------------------------------------------------------ 
       85620   |  LRO_LROCWAC                |    CK-dependent
       85621   |  LRO_LROCWAC_VIS            |    Fixed offset
       85626   |  LRO_LROCWAC_UV             |    Fixed offset
       85631   |  LRO_LROCWAC_VIS_FILTER_1   |    415 nm
       85632   |  LRO_LROCWAC_VIS_FILTER_2   |    566 nm
       85633   |  LRO_LROCWAC_VIS_FILTER_3   |    604 nm
       85634   |  LRO_LROCWAC_VIS_FILTER_4   |    643 nm
       85635   |  LRO_LROCWAC_VIS_FILTER_5   |    689 nm
       85641   |  LRO_LROCWAC_UV_FILTER_1    |    321 nm
       85642   |  LRO_LROCWAC_UV_FILTER_2    |    360 nm
 
   The angles provided in the frame definitions below are the values
   used by the LROC team in the image processing pipeline during
   operations.

   \begindata

      FRAME_LRO_LROCWAC               = -85620
      FRAME_-85620_NAME               = 'LRO_LROCWAC'
      FRAME_-85620_CLASS              = 3
      FRAME_-85620_CLASS_ID           = -85620
      FRAME_-85620_CENTER             = -85
      TKFRAME_-85620_RELATIVE         = 'LRO_SC_BUS'
      CK_-85620_SCLK                  = -85
      CK_-85620_SPK                   = -85
      
      FRAME_LRO_LROCWAC_VIS           = -85621
      FRAME_-85621_NAME               = 'LRO_LROCWAC_VIS'
      FRAME_-85621_CLASS              = 4
      FRAME_-85621_CLASS_ID           = -85621
      FRAME_-85621_CENTER             = -85
      TKFRAME_-85621_SPEC             = 'ANGLES'
      TKFRAME_-85621_RELATIVE         = 'LRO_LROCWAC'
      TKFRAME_-85621_ANGLES           = ( -0.229875, -1.082653, -0.079418 )
      TKFRAME_-85621_AXES             = (  1,      2,     3   )
      TKFRAME_-85621_UNITS            = 'DEGREES'
      
      FRAME_LRO_LROCWAC_UV            = -85626
      FRAME_-85626_NAME               = 'LRO_LROCWAC_UV'
      FRAME_-85626_CLASS              = 4
      FRAME_-85626_CLASS_ID           = -85626
      FRAME_-85626_CENTER             = -85
      TKFRAME_-85626_SPEC             = 'ANGLES'
      TKFRAME_-85626_RELATIVE         = 'LRO_LROCWAC'
      TKFRAME_-85626_ANGLES           = ( -0.140354, -1.146190, -0.099517 )
      TKFRAME_-85626_AXES             = (  1,      2,     3   )
      TKFRAME_-85626_UNITS            = 'DEGREES'

      FRAME_LRO_LROCWAC_VIS_FILTER_1  = -85631
      FRAME_-85631_NAME               = 'LRO_LROCWAC_VIS_FILTER_1'
      FRAME_-85631_CLASS              = 4
      FRAME_-85631_CLASS_ID           = -85631
      FRAME_-85631_CENTER             = -85
      TKFRAME_-85631_SPEC             = 'ANGLES'
      TKFRAME_-85631_RELATIVE         = 'LRO_LROCWAC_VIS'
      TKFRAME_-85631_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85631_AXES             = (  1,      2,     3   )
      TKFRAME_-85631_UNITS            = 'DEGREES'

      FRAME_LRO_LROCWAC_VIS_FILTER_2  = -85632
      FRAME_-85632_NAME               = 'LRO_LROCWAC_VIS_FILTER_2'
      FRAME_-85632_CLASS              = 4
      FRAME_-85632_CLASS_ID           = -85632
      FRAME_-85632_CENTER             = -85
      TKFRAME_-85632_SPEC             = 'ANGLES'
      TKFRAME_-85632_RELATIVE         = 'LRO_LROCWAC_VIS'
      TKFRAME_-85632_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85632_AXES             = (  1,      2,     3   )
      TKFRAME_-85632_UNITS            = 'DEGREES'

      FRAME_LRO_LROCWAC_VIS_FILTER_3  = -85633
      FRAME_-85633_NAME               = 'LRO_LROCWAC_VIS_FILTER_3'
      FRAME_-85633_CLASS              = 4
      FRAME_-85633_CLASS_ID           = -85633
      FRAME_-85633_CENTER             = -85
      TKFRAME_-85633_SPEC             = 'ANGLES'
      TKFRAME_-85633_RELATIVE         = 'LRO_LROCWAC_VIS'
      TKFRAME_-85633_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85633_AXES             = (  1,      2,     3   )
      TKFRAME_-85633_UNITS            = 'DEGREES'

      FRAME_LRO_LROCWAC_VIS_FILTER_4  = -85634
      FRAME_-85634_NAME               = 'LRO_LROCWAC_VIS_FILTER_4'
      FRAME_-85634_CLASS              = 4
      FRAME_-85634_CLASS_ID           = -85634
      FRAME_-85634_CENTER             = -85
      TKFRAME_-85634_SPEC             = 'ANGLES'
      TKFRAME_-85634_RELATIVE         = 'LRO_LROCWAC_VIS'
      TKFRAME_-85634_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85634_AXES             = (  1,      2,     3   )
      TKFRAME_-85634_UNITS            = 'DEGREES'

      FRAME_LRO_LROCWAC_VIS_FILTER_5  = -85635
      FRAME_-85635_NAME               = 'LRO_LROCWAC_VIS_FILTER_5'
      FRAME_-85635_CLASS              = 4
      FRAME_-85635_CLASS_ID           = -85635
      FRAME_-85635_CENTER             = -85
      TKFRAME_-85635_SPEC             = 'ANGLES'
      TKFRAME_-85635_RELATIVE         = 'LRO_LROCWAC_VIS'
      TKFRAME_-85635_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85635_AXES             = (  1,      2,     3   )
      TKFRAME_-85635_UNITS            = 'DEGREES'

      FRAME_LRO_LROCWAC_UV_FILTER_1   = -85641
      FRAME_-85641_NAME               = 'LRO_LROCWAC_UV_FILTER_1'
      FRAME_-85641_CLASS              = 4
      FRAME_-85641_CLASS_ID           = -85641
      FRAME_-85641_CENTER             = -85
      TKFRAME_-85641_SPEC             = 'ANGLES'
      TKFRAME_-85641_RELATIVE         = 'LRO_LROCWAC_UV'
      TKFRAME_-85641_ANGLES           = ( 0.0, 0.0, 0.0 ) 
      TKFRAME_-85641_AXES             = (  1,      2,     3   )
      TKFRAME_-85641_UNITS            = 'DEGREES'

      FRAME_LRO_LROCWAC_UV_FILTER_2   = -85642
      FRAME_-85642_NAME               = 'LRO_LROCWAC_UV_FILTER_2'
      FRAME_-85642_CLASS              = 4
      FRAME_-85642_CLASS_ID           = -85642
      FRAME_-85642_CENTER             = -85
      TKFRAME_-85642_SPEC             = 'ANGLES'
      TKFRAME_-85642_RELATIVE         = 'LRO_LROCWAC_UV'
      TKFRAME_-85642_ANGLES           = ( 0.0, 0.0, 0.0 ) 
      TKFRAME_-85642_AXES             = (  1,      2,     3   )
      TKFRAME_-85642_UNITS            = 'DEGREES'

   \begintext


MINI RF (MINIRF) Frame
--------------------------------------------------------

   The location of the Mini-RF instrument is provided by [ref 10].

   The MINI RF frame is defined by the instrument design as follows:

      *  +X axis is parallel to the spacecraft +X axis;

      *  +Y axis completes the frame;

      *  +Z axis is perpendicular the plane of the Mini RF antenna;

      *  the origin of this frame is at the instrument to spacecraft
         interface, 114.30, -38.10, 93.98; this offset (in centimeters)
         is from the LRO separation plane to the center of the
         instrument to spacecraft bolt pattern. There is no implied
         accuracy/precision in the conversion from inches to
         centimeters, other than the standard 2.54 centimeters per inch
         conversion.

   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles, as documented in the frame
   definition below, are from LRO mechanical drawings.

   \begindata

      FRAME_LRO_MINIRF         = -85700
      FRAME_-85700_NAME        = 'LRO_MINIRF'
      FRAME_-85700_CLASS       = 4
      FRAME_-85700_CLASS_ID    = -85700
      FRAME_-85700_CENTER      = -85
      TKFRAME_-85700_SPEC      = 'ANGLES'
      TKFRAME_-85700_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85700_ANGLES    = ( -47.6, 0.0, 0.0 )
      TKFRAME_-85700_AXES      = (   1,   2,   3   )
      TKFRAME_-85700_UNITS     = 'DEGREES'

   \begintext


Primary Star Tracker (STARP) Frame
--------------------------------------------------------

   The Primary Star Tracker frame is defined by the instrument design
   as follows:

      *  +X axis is defined by the alignment cube face 1;

      *  +Y axis is defined by alignment cube face 2;

      *  +Z axis is the boresight of the star tracker;


   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles provided in the frame
   definition below are extracted from [11].   

   \begindata

      FRAME_LRO_STARP          = -85010
      FRAME_-85010_NAME        = 'LRO_STARP'
      FRAME_-85010_CLASS       = 4
      FRAME_-85010_CLASS_ID    = -85010
      FRAME_-85010_CENTER      = -85
      TKFRAME_-85010_SPEC      = 'ANGLES'
      TKFRAME_-85010_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85010_ANGLES    = ( -119.95519238, 
                                     30.063992492, 
                                   -150.16190703  )
      TKFRAME_-85010_AXES      = (   2,     1,   3   )
      TKFRAME_-85010_UNITS     = 'DEGREES'

   \begintext


Secondary Star Tracker (STARS) Frame
--------------------------------------------------------

   The Secondary Star Tracker frame is defined by the instrument design
   as follows:

      *  +X axis is defined by alignment cube face 1;

      *  +Y axis is defined by alignment cube face 2;

      *  +Z axis is the boresight of the star tracker;

   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles provided in the frame
   definition below are extracted from [12].

   \begindata

      FRAME_LRO_STARS          = -85011
      FRAME_-85011_NAME        = 'LRO_STARS'
      FRAME_-85011_CLASS       = 4
      FRAME_-85011_CLASS_ID    = -85011
      FRAME_-85011_CENTER      = -85
      TKFRAME_-85011_SPEC      = 'ANGLES'
      TKFRAME_-85011_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85011_ANGLES    = ( -179.81255618, 
                                     30.057893306, 
                                     89.969314304  )
      TKFRAME_-85011_AXES      = (   2,     1,   3   )
      TKFRAME_-85011_UNITS     = 'DEGREES'

   \begintext


Miniature Inertial Measurement Unit (MIMU) Frame
--------------------------------------------------------

   The MIMU frame is defined by the instrument design as follows:

      *  +X axis is parallel to the +X axis of the spacecraft;

      *  +Y axis is parallel to the +Y axis of the spacecraft;

      *  +Z axis is parallel to the +Z axis of the spacecraft;

   The orientation of this frame is fixed with respect to the
   spacecraft frame. The rotation angles provided in the frame
   definition below are extracted from [13].

   \begindata

      FRAME_LRO_MIMU           = -85012
      FRAME_-85012_NAME        = 'LRO_MIMU'
      FRAME_-85012_CLASS       = 4
      FRAME_-85012_CLASS_ID    = -85012
      FRAME_-85012_CENTER      = -85
      TKFRAME_-85012_SPEC      = 'ANGLES'
      TKFRAME_-85012_RELATIVE  = 'LRO_SC_BUS'
      TKFRAME_-85012_ANGLES    = ( 0.0, 0.0, 0.0 )
      TKFRAME_-85012_AXES      = ( 1,   2,   3   )
      TKFRAME_-85012_UNITS     = 'DEGREES'

   \begintext


High Gain Antenna (HGA) Frame
--------------------------------------------------------

   The HGA frame is defined by the antenna design as follows:

      *  -Z is along the HGA boresight;

      *  +Y is parallel to the TBD gimbal rotation axis and points TBD;

      *  +X completes the right handed frame;

   The orientation of this frame is provided in CK files. In zero gimbal
   position the HGA frame is co-aligned with the spacecraft frame.

   \begindata

      FRAME_LRO_HGA            = -85020
      FRAME_-85020_NAME        = 'LRO_HGA'
      FRAME_-85020_CLASS       = 3
      FRAME_-85020_CLASS_ID    = -85020
      FRAME_-85020_CENTER      = -85
      CK_-85020_SCLK           = -85
      CK_-85020_SPK            = -85

   \begintext


Solar Array (SA) Frame
--------------------------------------------------------

   The SA frame is defined by the array design as follows:

      *  -Y is along the normal on the solar array active cell side;

      *  +Z is parallel to the TBD gimbal rotation axis and points TBD;

      *  +X completes the right handed frame;

   The orientation of this frame is provided in CK files. In zero gimbal
   position the SA frame is co-aligned with the spacecraft frame.

   \begindata

      FRAME_LRO_SA             = -85030
      FRAME_-85030_NAME        = 'LRO_SA'
      FRAME_-85030_CLASS       = 3
      FRAME_-85030_CLASS_ID    = -85030
      FRAME_-85030_CENTER      = -85
      CK_-85030_SCLK           = -85
      CK_-85030_SPK            = -85

   \begintext


Lunar Reconnaissance Orbiter NAIF ID Codes -- Definitions
--------------------------------------------------------

   This section contains name to NAIF ID mappings for the LRO mission.
   Once the contents of this file is loaded into the KERNEL POOL, these
   mappings become available within SPICE, making it possible to use
   names instead of ID code in the high level SPICE routine calls.

   The set of codes below is not complete. Additional ID codes for some
   LRO instruments are defined in the IK files.

   Spacecraft:
   -----------

      LRO                           -85
      LUNAR RECONNAISSANCE ORBITER  -85
      LRO_SPACECRAFT                -85000
      LRO_SC_BUS                    -85000

   Spacecraft structures:
   ----------------------

      LRO_STARP                     -85010
      LRO_STARS                     -85011
      LRO_MIMU                      -85012
      LRO_HGA                       -85020
      LRO_SA                        -85030

   Science Instruments:
   --------------------

      LRO_CRATER                    -85100
      LRO_DLRE                      -85200
      LRO_LAMP                      -85300
      LRO_LEND                      -85400
      LRO_LOLA                      -85500
      LRO_LROCNACL                  -85600
      LRO_LROCNACR                  -85610
      LRO_LROCWAC                   -85620
      LRO_LROCWAC_VIS               -85621
      LRO_LROCWAC_UV                -85626
      LRO_LROCWAC_VIS_FILTER_1      -85631
      LRO_LROCWAC_VIS_FILTER_2      -85632
      LRO_LROCWAC_VIS_FILTER_3      -85633
      LRO_LROCWAC_VIS_FILTER_4      -85634
      LRO_LROCWAC_VIS_FILTER_5      -85635
      LRO_LROCWAC_UV_FILTER_1       -85641
      LRO_LROCWAC_UV_FILTER_2       -85642
      LRO_MINIRF                    -85700

   The mappings summarized in this table are implemented by the keywords
   below.

   \begindata

      NAIF_BODY_NAME += ( 'LRO' )
      NAIF_BODY_CODE += ( -85 )

      NAIF_BODY_NAME += ( 'LUNAR RECONNAISSANCE ORBITER' )
      NAIF_BODY_CODE += ( -85 )

      NAIF_BODY_NAME += ( 'LRO_SPACECRAFT' )
      NAIF_BODY_CODE += ( -85000 )

      NAIF_BODY_NAME += ( 'LRO_SC_BUS' )
      NAIF_BODY_CODE += ( -85000 )

      NAIF_BODY_NAME += ( 'LRO_STARP' )
      NAIF_BODY_CODE += ( -85010 )

      NAIF_BODY_NAME += ( 'LRO_STARS' )
      NAIF_BODY_CODE += ( -85011 )

      NAIF_BODY_NAME += ( 'LRO_MIMU' )
      NAIF_BODY_CODE += ( -85012 )

      NAIF_BODY_NAME += ( 'LRO_HGA' )
      NAIF_BODY_CODE += ( -85020 )

      NAIF_BODY_NAME += ( 'LRO_SA' )
      NAIF_BODY_CODE += ( -85030 )

      NAIF_BODY_NAME += ( 'LRO_CRATER' )
      NAIF_BODY_CODE += ( -85100 )

      NAIF_BODY_NAME += ( 'LRO_DLRE' )
      NAIF_BODY_CODE += ( -85200 )

      NAIF_BODY_NAME += ( 'LRO_LAMP' )
      NAIF_BODY_CODE += ( -85300 )

      NAIF_BODY_NAME += ( 'LRO_LEND' )
      NAIF_BODY_CODE += ( -85400 )

      NAIF_BODY_NAME += ( 'LRO_LOLA' )
      NAIF_BODY_CODE += ( -85500 )

      NAIF_BODY_NAME += ( 'LRO_LROCNACL' )
      NAIF_BODY_CODE += ( -85600 )

      NAIF_BODY_NAME += ( 'LRO_LROCNACR' )
      NAIF_BODY_CODE += ( -85610 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC' )
      NAIF_BODY_CODE += ( -85620 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_VIS' )
      NAIF_BODY_CODE += ( -85621 )
      
      NAIF_BODY_NAME += ( 'LRO_LROCWAC_UV' )
      NAIF_BODY_CODE += ( -85626 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_VIS_FILTER_1' )
      NAIF_BODY_CODE += ( -85631 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_VIS_FILTER_2' )
      NAIF_BODY_CODE += ( -85632 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_VIS_FILTER_3' )
      NAIF_BODY_CODE += ( -85633 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_VIS_FILTER_4' )
      NAIF_BODY_CODE += ( -85634 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_VIS_FILTER_5' )
      NAIF_BODY_CODE += ( -85635 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_UV_FILTER_1' )
      NAIF_BODY_CODE += ( -85641 )

      NAIF_BODY_NAME += ( 'LRO_LROCWAC_UV_FILTER_2' )
      NAIF_BODY_CODE += ( -85642 )

      NAIF_BODY_NAME += ( 'LRO_MINIRF' )
      NAIF_BODY_CODE += ( -85700 )

   \begintext

End of FK file.
