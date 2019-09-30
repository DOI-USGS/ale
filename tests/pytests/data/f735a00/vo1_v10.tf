KPL/FK

Viking Orbiter 1 Frames Kernel
===============================================================================

   This frame kernel contains the complete set of frame definitions for
   the Viking Orbiter 1 (VO1) spacecraft including definitions for the
   spacecraft frame, instrument platform frame, and science instrument
   frames.


Version and Date
-------------------------------------------------------------------------------

   Version 1.0 -- December 1, 2005 -- Boris Semenov, NAIF

      Initial Release.


References
-------------------------------------------------------------------------------

   1. ``Frames Required Reading''

   2. ``Kernel Pool Required Reading''

   3. ``C-Kernel Required Reading''

   4. VO1 Supplemental Experiment Data Record (SEDR) file, 1976-1980.

   5. Thomas C. Duxbury/JPL, March-July, 1996


Contact Information
-------------------------------------------------------------------------------

   Boris V. Semenov, NAIF/JPL, (818)-354-8136, Boris.Semenov@jpl.nasa.gov


Implementation Notes
-------------------------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must `load' the kernel, normally during
   program initialization. The SPICELIB/CSPICE/ICY routine FURNSH loads
   a kernel file into the pool as shown below.

      CALL FURNSH ( 'frame_kernel_name' )           SPICELIB/FORTRAN

      furnsh_c ( "frame_kernel_name" );             CSPICE/C

      cspice_furnsh, "frame_kernel_name"            ICY/IDL

   This file was created and may be updated with a text editor or word
   processor.


VO1 Frames
-------------------------------------------------------------------------------

   The following VO1 frames are defined in this kernel file:

           Name                  Relative to           Type       NAIF ID
      ======================  ===================  ============   =======

   Spacecraft frame:
   -----------------
      VO1_SPACECRAFT          rel.to B1950         CK             -27900

   Instrument Platform frame:
   --------------------------
      VO1_PLATFORM            rel.to B1950         CK             -27000

   Science Instrument frames:
   --------------------------
      VO1_VISA                rel.to PLATFORM      FIXED          -27001
      VO1_VISA_TD             rel.to PLATFORM      FIXED          -27011
      VO1_VISB                rel.to PLATFORM      FIXED          -27002
      VO1_VISB_TD             rel.to PLATFORM      FIXED          -27012
      VO1_IRTM                rel.to PLATFORM      FIXED          -27003
      VO1_MAWD                rel.to PLATFORM      FIXED          -27004


VO1 Frames Hierarchy
-------------------------------------------------------------------------------

   The diagram below shows VO1 frames hierarchy:


                               "B1950" INERTIAL
        +-----------------------------------------------------------+
        |                   |                   |                   |
        | <--pck            | <--ck             | <--ck             | <--pck
        |                   |                   |                   |
        V                   V                   |                   V
    "IAU_MARS"      "V01_SPACECRAFT"            |              "IAU_EARTH"
    -----------     ----------------            |              ------------
                                                |
                                                |
                                                V
                                          "VO1_PLATFORM"
        +---------------------------------------------------------+
        |                 |                     |     |     |     |
        | <--fixed        | <--fixed   fixed--> |     |     |     | <--fixed
        |                 |                     |     |     |     |
        V                 V                     V     |     |     V
    "VO1_IRTM"        "VO1_MAWD"          "VO1_VISA"  |     | "VO1_VISB"
    ----------        ----------          ----------  |     | ----------
                                                      |     |
                                                      |     |
                                             fixed--> |     | <--fixed
                                                      |     |
                                                      V     V
                                          "VO1_VISA_TD"     "VO1_VISB_TD"
                                          -------------     -------------


Spacecraft Bus Frame
-------------------------------------------------------------------------------

   The spacecraft frame is defined as follows:

      [TBD]

   This diagram illustrates the spacecraft bus frame:

      [TBD]

   Since the S/C bus attitude with respect to an inertial frame is
   provided by a C kernel (see [3] for more information), this frame is
   defined as a CK-based frame.

   \begindata

      FRAME_VO1_SPACECRAFT     = -27900
      FRAME_-27900_NAME        = 'VO1_SPACECRAFT'
      FRAME_-27900_CLASS       = 3
      FRAME_-27900_CLASS_ID    = -27900
      FRAME_-27900_CENTER      = -27
      CK_-27900_SCLK           = -27
      CK_-27900_SPK            = -27

   \begintext


Instrument Platform Frame
-------------------------------------------------------------------------------

   The instrument platform frame is defined as follows:

      [TBD]

   This diagram illustrates the instrument platform frame:

      [TBD]

   Since the platform attitude with respect to an inertial frame is
   provided by a C kernel (see [3] for more information), this frame is
   defined as a CK-based frame.

   \begindata

      FRAME_VO1_PLATFORM       = -27000
      FRAME_-27000_NAME        = 'VO1_PLATFORM'
      FRAME_-27000_CLASS       = 3
      FRAME_-27000_CLASS_ID    = -27000
      FRAME_-27000_CENTER      = -27
      CK_-27000_SCLK           = -27
      CK_-27000_SPK            = -27

   \begintext


VISA and VISB Frames
-------------------------------------------------------------------------------

   The frames for the Visual Imaging Subsystem Camera A (VISA) and
   Camera B (VISB) -- VO1_VISA and VO1_VISB -- are defined as follows:

      -  Z axis is along the camera boresight;

      -  X axis is along detector lines and points toward the right
         side of the image;

      -  Y axis completes the right hand frame;

      -  the origin of this frame is located at the camera focal point.

   This diagram illustrates the camera frames:

      [TBD]

   Since the cameras were rigidly mounted on the instrument platform
   their frames are defined as fixed offset frames with respect to the
   platform frame.

   The camera orientation with respect to the platform was specified in
   [4] as the following three rotation angles:

      T pl-cam = R (RASTER_ORIENTATION) R (CONE) R (-CROSS_CONE)
                  Z                      Y        X

   with the following angle values for each of the two cameras:

      VISA:

         CROSS_CONE         =  -0.707350, deg
         CONE               =  -0.007580, deg
         RASTER_ORIENTATION =  89.735690, deg

      VISB:

         CROSS_CONE         =   0.681000, deg
         CONE               =  -0.032000, deg
         RASTER_ORIENTATION =  90.022800, deg

   These angles are used in the frame definition below.

   (The frame definition below contains the opposite of the rotation
   described above because Euler angles specified in it define
   transformation from the camera to the platform frame -- see [1].)

   \begindata

      FRAME_VO1_VISA           = -27001
      FRAME_-27001_NAME        = 'VO1_VISA'
      FRAME_-27001_CLASS       = 4
      FRAME_-27001_CLASS_ID    = -27001
      FRAME_-27001_CENTER      = -27
      TKFRAME_-27001_SPEC      = 'ANGLES'
      TKFRAME_-27001_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-27001_ANGLES    = ( -0.707350, 0.007580, -89.735690 )
      TKFRAME_-27001_AXES      = (  1,        2,          3        )
      TKFRAME_-27001_UNITS     = 'DEGREES'

      FRAME_VO1_VISB           = -27002
      FRAME_-27002_NAME        = 'VO1_VISB'
      FRAME_-27002_CLASS       = 4
      FRAME_-27002_CLASS_ID    = -27002
      FRAME_-27002_CENTER      = -27
      TKFRAME_-27002_SPEC      = 'ANGLES'
      TKFRAME_-27002_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-27002_ANGLES    = (  0.681000, 0.032000, -90.022800 )
      TKFRAME_-27002_AXES      = (  1,        2,          3        )
      TKFRAME_-27002_UNITS     = 'DEGREES'

   \begintext

   Thomas Duxbury provided a different set of angle values for the cameras
   (see [5]):

      VISA:

         CROSS_CONE         =  -0.707400, deg
         CONE               =  -0.007600, deg
         RASTER_ORIENTATION =  89.735400, deg

      VISB:

         CROSS_CONE         =   0.681300, deg
         CONE               =  -0.032100, deg
         RASTER_ORIENTATION =  90.083900, deg

   To make this alignment solution available, two more frames parallel
   to the camera frames and including the ``_TB'' suffix in the name
   and defined below.

   (The frame definition below contains the opposite of the rotation
   described above because Euler angles specified in it define
   transformation from the camera to the platform frame -- see [1].)

   \begindata

      FRAME_VO1_VISA_TD        = -27011
      FRAME_-27011_NAME        = 'VO1_VISA_TD'
      FRAME_-27011_CLASS       = 4
      FRAME_-27011_CLASS_ID    = -27011
      FRAME_-27011_CENTER      = -27
      TKFRAME_-27011_SPEC      = 'ANGLES'
      TKFRAME_-27011_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-27011_ANGLES    = ( -0.707400, 0.007600, -89.735400 )
      TKFRAME_-27011_AXES      = (  1,        2,          3        )
      TKFRAME_-27011_UNITS     = 'DEGREES'

      FRAME_VO1_VISB_TD        = -27012
      FRAME_-27012_NAME        = 'VO1_VISB_TD'
      FRAME_-27012_CLASS       = 4
      FRAME_-27012_CLASS_ID    = -27012
      FRAME_-27012_CENTER      = -27
      TKFRAME_-27012_SPEC      = 'ANGLES'
      TKFRAME_-27012_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-27012_ANGLES    = (  0.681300, 0.032100, -90.083900 )
      TKFRAME_-27012_AXES      = (  1,        2,          3        )
      TKFRAME_-27012_UNITS     = 'DEGREES'

   \begintext


IRTM Frame
-------------------------------------------------------------------------------

   The frame for the Infrared Thermal Mapper -- VO1_IRTM -- is defined
   as follows:

      [TBD]

   This diagram illustrates the IRTM frame:

      [TBD]

   Since the IRTM instrument was rigidly mounted on the instrument
   platform its frame is defined as fixed offset frames with respect to
   the platform frame.

   The instrument orientation with respect to the platform can be
   specified using the following three rotation angles:

      T pl-cam = R (RASTER_ORIENTATION) R (CONE) R (-CROSS_CONE)
                  Z                      Y        X

   with the nominal angle values:

      CROSS_CONE         =   0.0, deg
      CONE               =   0.0, deg
      RASTER_ORIENTATION =  90.0, deg

   These angles are used in the frame definition below.

   (The frame definition below contains the opposite of the rotation
   described above because Euler angles specified in it define
   transformation from the instrument to the platform frame -- see
   [1].)

   \begindata

      FRAME_VO1_IRTM           = -27003
      FRAME_-27003_NAME        = 'VO1_IRTM'
      FRAME_-27003_CLASS       = 4
      FRAME_-27003_CLASS_ID    = -27003
      FRAME_-27003_CENTER      = -27
      TKFRAME_-27003_SPEC      = 'ANGLES'
      TKFRAME_-27003_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-27003_ANGLES    = (  0.0, 0.0, -90.0 )
      TKFRAME_-27003_AXES      = (  1,   2,     3   )
      TKFRAME_-27003_UNITS     = 'DEGREES'

   \begintext


MAWD Frame
-------------------------------------------------------------------------------

   The frame for the Mars Atmosphere Water Detector (MAWD) -- VO1_MAWD
   -- is defined as follows:

      [TBD]

   This diagram illustrates the MAWD frame:

      [TBD]

   Since the MAWD instrument was rigidly mounted on the instrument
   platform its frame is defined as fixed offset frames with respect to
   the platform frame.

   The instrument orientation with respect to the platform can be
   specified using the following three rotation angles:

      T pl-cam = R (RASTER_ORIENTATION) R (CONE) R (-CROSS_CONE)
                  Z                      Y        X

   with the nominal angle values:

      CROSS_CONE         =   0.0, deg
      CONE               =   0.0, deg
      RASTER_ORIENTATION =  90.0, deg

   These angles are used in the frame definition below.

   (The frame definition below contains the opposite of the rotation
   described above because Euler angles specified in it define
   transformation from the instrument to the platform frame -- see
   [1].)

   \begindata

      FRAME_VO1_MAWD           = -27004
      FRAME_-27004_NAME        = 'VO1_MAWD'
      FRAME_-27004_CLASS       = 4
      FRAME_-27004_CLASS_ID    = -27004
      FRAME_-27004_CENTER      = -27
      TKFRAME_-27004_SPEC      = 'ANGLES'
      TKFRAME_-27004_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-27004_ANGLES    = (  0.0, 0.0, -90.0 )
      TKFRAME_-27004_AXES      = (  1,   2,     3   )
      TKFRAME_-27004_UNITS     = 'DEGREES'

   \begintext


Viking Orbiter 1 NAIF ID Codes -- Definition Section
========================================================================

   This section contains name to NAIF ID mappings for the VO1. Once the
   contents of this file is loaded into the KERNEL POOL, these mappings
   become available within SPICE, making it possible to use names
   instead of ID code in the high level SPICE routine calls.

   Spacecraft and its structures:
   ------------------------------

      VIKING ORBITER 1           -27
      VO1                        -27

      VO1_SPACECRAFT_BUS         -27900
      VO1_SC_BUS                 -27900
      VO1_SPACECRAFT             -27900

      VO1_INSTRUMENT_PLATFORM    -27000
      VO1_PLATFORM               -27000

   Science Instruments:
   --------------------

      VO1_VISA                   -27001
      VO1_VISB                   -27002

      VO1_IRTM                   -27003

      VO1_MAWD                   -27004

   The mappings summarized in this table are implemented by the keywords
   below.

   \begindata

      NAIF_BODY_NAME += ( 'VIKING ORBITER 1'          )
      NAIF_BODY_CODE += ( -27                         )

      NAIF_BODY_NAME += ( 'VO1'                       )
      NAIF_BODY_CODE += ( -27                         )

      NAIF_BODY_NAME += ( 'VO1_SPACECRAFT_BUS'        )
      NAIF_BODY_CODE += ( -27900                      )

      NAIF_BODY_NAME += ( 'VO1_SC_BUS'                )
      NAIF_BODY_CODE += ( -27900                      )

      NAIF_BODY_NAME += ( 'VO1_SPACECRAFT'            )
      NAIF_BODY_CODE += ( -27900                      )

      NAIF_BODY_NAME += ( 'VO1_INSTRUMENT_PLATFORM'   )
      NAIF_BODY_CODE += ( -27000                      )

      NAIF_BODY_NAME += ( 'VO1_PLATFORM'              )
      NAIF_BODY_CODE += ( -27000                      )

      NAIF_BODY_NAME += ( 'VO1_VISA'                  )
      NAIF_BODY_CODE += ( -27001                      )

      NAIF_BODY_NAME += ( 'VO1_VISB'                  )
      NAIF_BODY_CODE += ( -27002                      )

      NAIF_BODY_NAME += ( 'VO1_IRTM'                  )
      NAIF_BODY_CODE += ( -27003                      )

      NAIF_BODY_NAME += ( 'VO1_MAWD'                  )
      NAIF_BODY_CODE += ( -27004                      )

   \begintext
