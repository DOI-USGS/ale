KPL/FK

Viking Orbiter 2 Frames Kernel
===============================================================================

   This frame kernel contains the complete set of frame definitions for
   the Viking Orbiter 2 (VO2) spacecraft including definitions for the
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

   4. VO2 Supplemental Experiment Data Record (SEDR) file, 1976-1980.
 
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


VO2 Frames
-------------------------------------------------------------------------------

   The following VO2 frames are defined in this kernel file:

           Name                  Relative to           Type       NAIF ID
      ======================  ===================  ============   =======

   Spacecraft frame:
   -----------------
      VO2_SPACECRAFT          rel.to B1950         CK             -30900

   Instrument Platform frame:
   --------------------------
      VO2_PLATFORM            rel.to B1950         CK             -30000

   Science Instrument frames:
   --------------------------
      VO2_VISA                rel.to PLATFORM      FIXED          -30001
      VO2_VISA_TD             rel.to PLATFORM      FIXED          -30011
      VO2_VISB                rel.to PLATFORM      FIXED          -30002
      VO2_VISB_TD             rel.to PLATFORM      FIXED          -30012
      VO2_IRTM                rel.to PLATFORM      FIXED          -30003
      VO2_MAWD                rel.to PLATFORM      FIXED          -30004


VO2 Frames Hierarchy
-------------------------------------------------------------------------------

   The diagram below shows VO2 frames hierarchy:


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
                                          "VO2_PLATFORM"                       
        +---------------------------------------------------------+
        |                 |                     |     |     |     |
        | <--fixed        | <--fixed   fixed--> |     |     |     | <--fixed
        |                 |                     |     |     |     |
        V                 V                     V     |     |     V
    "VO2_IRTM"        "VO2_MAWD"          "VO2_VISA"  |     | "VO2_VISB"
    ----------        ----------          ----------  |     | ----------
                                                      |     |
                                                      |     |
                                             fixed--> |     | <--fixed
                                                      |     |
                                                      V     V
                                          "VO2_VISA_TD"     "VO2_VISB_TD"
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

      FRAME_VO2_SPACECRAFT     = -30900
      FRAME_-30900_NAME        = 'VO2_SPACECRAFT'
      FRAME_-30900_CLASS       = 3
      FRAME_-30900_CLASS_ID    = -30900
      FRAME_-30900_CENTER      = -30
      CK_-30900_SCLK           = -30
      CK_-30900_SPK            = -30

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

      FRAME_VO2_PLATFORM       = -30000
      FRAME_-30000_NAME        = 'VO2_PLATFORM'
      FRAME_-30000_CLASS       = 3
      FRAME_-30000_CLASS_ID    = -30000
      FRAME_-30000_CENTER      = -30
      CK_-30000_SCLK           = -30
      CK_-30000_SPK            = -30

   \begintext


VISA and VISB Frames
-------------------------------------------------------------------------------

   The frames for the Visual Imaging Subsystem Camera A (VISA) and
   Camera B (VISB) -- VO2_VISA and VO2_VISB -- are defined as follows:

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

         CROSS_CONE         =  -0.679330, deg
         CONE               =  -0.023270, deg
         RASTER_ORIENTATION =  89.880691, deg

      VISB:

         CROSS_CONE         =   0.663000, deg
         CONE               =  -0.044000, deg
         RASTER_ORIENTATION =  89.663790, deg

   These angles are used in the frame definition below.

   (The frame definition below contains the opposite of the rotation
   described above because Euler angles specified in it define
   transformation from the camera to the platform frame -- see [1].)

   \begindata

      FRAME_VO2_VISA           = -30001
      FRAME_-30001_NAME        = 'VO2_VISA'
      FRAME_-30001_CLASS       = 4
      FRAME_-30001_CLASS_ID    = -30001
      FRAME_-30001_CENTER      = -30
      TKFRAME_-30001_SPEC      = 'ANGLES'
      TKFRAME_-30001_RELATIVE  = 'VO2_PLATFORM'
      TKFRAME_-30001_ANGLES    = ( -0.679330, 0.023270, -89.880691 )
      TKFRAME_-30001_AXES      = (  1,        2,          3        )
      TKFRAME_-30001_UNITS     = 'DEGREES'

      FRAME_VO2_VISB           = -30002
      FRAME_-30002_NAME        = 'VO2_VISB'
      FRAME_-30002_CLASS       = 4
      FRAME_-30002_CLASS_ID    = -30002
      FRAME_-30002_CENTER      = -30
      TKFRAME_-30002_SPEC      = 'ANGLES'
      TKFRAME_-30002_RELATIVE  = 'VO2_PLATFORM'
      TKFRAME_-30002_ANGLES    = (  0.663000, 0.044000, -89.663790 )
      TKFRAME_-30002_AXES      = (  1,        2,          3        )
      TKFRAME_-30002_UNITS     = 'DEGREES'

   \begintext

   Thomas Duxbury provided a different set of angle values for the cameras
   (see [5]):

      VISA:

         CROSS_CONE         =  -0.679300, deg
         CONE               =  -0.023300, deg
         RASTER_ORIENTATION =  89.880700, deg
      
      VISB:

         CROSS_CONE         =   0.663000, deg
         CONE               =  -0.044000, deg
         RASTER_ORIENTATION =  89.663800, deg

   To make this alignment solution available, two more frames parallel
   to the camera frames and including the ``_TB'' suffix in the name
   and defined below.
 
   (The frame definition below contains the opposite of the rotation
   described above because Euler angles specified in it define
   transformation from the camera to the platform frame -- see [1].)

   \begindata

      FRAME_VO1_VISA_TD        = -30011
      FRAME_-30011_NAME        = 'VO1_VISA_TD'
      FRAME_-30011_CLASS       = 4
      FRAME_-30011_CLASS_ID    = -30011
      FRAME_-30011_CENTER      = -30
      TKFRAME_-30011_SPEC      = 'ANGLES'
      TKFRAME_-30011_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-30011_ANGLES    = ( -0.679300, 0.023300, -89.880700 )
      TKFRAME_-30011_AXES      = (  1,        2,          3        )
      TKFRAME_-30011_UNITS     = 'DEGREES'

      FRAME_VO1_VISB_TD        = -30012
      FRAME_-30012_NAME        = 'VO1_VISB_TD'
      FRAME_-30012_CLASS       = 4
      FRAME_-30012_CLASS_ID    = -30012
      FRAME_-30012_CENTER      = -30
      TKFRAME_-30012_SPEC      = 'ANGLES'
      TKFRAME_-30012_RELATIVE  = 'VO1_PLATFORM'
      TKFRAME_-30012_ANGLES    = (  0.663000, 0.044000, -89.663800 )
      TKFRAME_-30012_AXES      = (  1,        2,          3        )
      TKFRAME_-30012_UNITS     = 'DEGREES'

   \begintext


IRTM Frame
-------------------------------------------------------------------------------

   The frame for the Infrared Thermal Mapper -- VO2_IRTM -- is defined 
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

      FRAME_VO2_IRTM           = -30003
      FRAME_-30003_NAME        = 'VO2_IRTM'
      FRAME_-30003_CLASS       = 4
      FRAME_-30003_CLASS_ID    = -30003
      FRAME_-30003_CENTER      = -30
      TKFRAME_-30003_SPEC      = 'ANGLES'
      TKFRAME_-30003_RELATIVE  = 'VO2_PLATFORM'
      TKFRAME_-30003_ANGLES    = (  0.0, 0.0, -90.0 )
      TKFRAME_-30003_AXES      = (  1,   2,     3   )
      TKFRAME_-30003_UNITS     = 'DEGREES'

   \begintext


MAWD Frame
-------------------------------------------------------------------------------

   The frame for the Mars Atmosphere Water Detector (MAWD) -- VO2_MAWD
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

      FRAME_VO2_MAWD           = -30004
      FRAME_-30004_NAME        = 'VO2_MAWD'
      FRAME_-30004_CLASS       = 4
      FRAME_-30004_CLASS_ID    = -30004
      FRAME_-30004_CENTER      = -30
      TKFRAME_-30004_SPEC      = 'ANGLES'
      TKFRAME_-30004_RELATIVE  = 'VO2_PLATFORM'
      TKFRAME_-30004_ANGLES    = (  0.0, 0.0, -90.0 )
      TKFRAME_-30004_AXES      = (  1,   2,     3   )
      TKFRAME_-30004_UNITS     = 'DEGREES'

   \begintext


Viking Orbiter 2 NAIF ID Codes -- Definition Section
========================================================================

   This section contains name to NAIF ID mappings for the VO2. Once the
   contents of this file is loaded into the KERNEL POOL, these mappings
   become available within SPICE, making it possible to use names
   instead of ID code in the high level SPICE routine calls.

   Spacecraft and its structures: 
   ------------------------------

      VIKING ORBITER 2           -30
      VO2                        -30

      VO2_SPACECRAFT_BUS         -30900
      VO2_SC_BUS                 -30900
      VO2_SPACECRAFT             -30900

      VO2_INSTRUMENT_PLATFORM    -30000
      VO2_PLATFORM               -30000

   Science Instruments:
   --------------------

      VO2_VISA                   -30001
      VO2_VISB                   -30002

      VO2_IRTM                   -30003

      VO2_MAWD                   -30004

   The mappings summarized in this table are implemented by the keywords 
   below.

   \begindata

      NAIF_BODY_NAME += ( 'VIKING ORBITER 2'          )
      NAIF_BODY_CODE += ( -30                         )

      NAIF_BODY_NAME += ( 'VO2'                       )
      NAIF_BODY_CODE += ( -30                         )

      NAIF_BODY_NAME += ( 'VO2_SPACECRAFT_BUS'        )
      NAIF_BODY_CODE += ( -30900                      )

      NAIF_BODY_NAME += ( 'VO2_SC_BUS'                )
      NAIF_BODY_CODE += ( -30900                      )

      NAIF_BODY_NAME += ( 'VO2_SPACECRAFT'            )
      NAIF_BODY_CODE += ( -30900                      )

      NAIF_BODY_NAME += ( 'VO2_INSTRUMENT_PLATFORM'   )
      NAIF_BODY_CODE += ( -30000                      )

      NAIF_BODY_NAME += ( 'VO2_PLATFORM'              )
      NAIF_BODY_CODE += ( -30000                      )

      NAIF_BODY_NAME += ( 'VO2_VISA'                  )
      NAIF_BODY_CODE += ( -30001                      )

      NAIF_BODY_NAME += ( 'VO2_VISB'                  )
      NAIF_BODY_CODE += ( -30002                      )

      NAIF_BODY_NAME += ( 'VO2_IRTM'                  )
      NAIF_BODY_CODE += ( -30003                      )

      NAIF_BODY_NAME += ( 'VO2_MAWD'                  )
      NAIF_BODY_CODE += ( -30004                      )

   \begintext
