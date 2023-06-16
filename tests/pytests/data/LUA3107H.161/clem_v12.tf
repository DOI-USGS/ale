KPL/FK

\beginlabel
PDS_VERSION_ID               = PDS3
RECORD_TYPE                  = STREAM
RECORD_BYTES                 = "N/A"
^SPICE_KERNEL                = "clem_v21.tf"
MISSION_NAME                 = "DEEP SPACE PROGRAM SCIENCE EXPERIMENT"
SPACECRAFT_NAME              = "CLEMENTINE 1"
DATA_SET_ID                  = "CLEM1-L-SPICE-6-V1.0"
KERNEL_TYPE_ID               = FK
PRODUCT_ID                   = "clem_v21.tf"
PRODUCT_CREATION_TIME        = 2017-10-01T00:00:00
PRODUCER_ID                  = "NAIF/JPL"
MISSION_PHASE_NAME           = "N/A"
PRODUCT_VERSION_TYPE         = ACTUAL
PLATFORM_OR_MOUNTING_NAME    = "N/A"
START_TIME                   = "N/A"
STOP_TIME                    = "N/A"
SPACECRAFT_CLOCK_START_COUNT = "N/A"
SPACECRAFT_CLOCK_STOP_COUNT  = "N/A"
TARGET_NAME                  = MOON
INSTRUMENT_NAME              = "N/A"
NAIF_INSTRUMENT_ID           = "N/A"
SOURCE_PRODUCT_ID            = "N/A"
NOTE                         = "See comments in the file for details"
OBJECT                       = SPICE_KERNEL
  INTERCHANGE_FORMAT         = ASCII
  KERNEL_TYPE                = FRAMES
  DESCRIPTION                = "Clementine SPICE FK file providing the
complete set of frame definitions for the Clementine spacecraft and its
science instruments, including band specific frames for the UVVIS 
camera. Created by NAIF with additions by ASU."
END_OBJECT                   = SPICE_KERNEL
\endlabel


Clementine Frame Definitions Kernel
==============================================================================

   This frame kernel (FK) contains the Clementine spacecraft and
   science instrument frame definitions. It also contains name -
   to - NAIF ID mappings for the Clementine science instruments (see
   the last section of the file.)

   Additional frames for each UVVIS camera filter have been added to 
   handle band specific optical distortion parameters.


Version and Date
--------------------------------------------------------

   Version 2.1 -- October 01, 2017 -- Emerson Speyerer, ASU

      Added descriptions and new frames for each UVVIS filter.

   Version 2.0 -- June 11, 2007 -- Boris Semenov

      Added descriptions. Added name-ID mapping keywords.

   Version 1.0 -- September 29, 2000 -- Boris Semenov

      Initial Release. Contains Euler angles from Clementine I-Kernel
      files. Does not contain a description for any of the frames.


References
--------------------------------------------------------

   1. C-kernel Required Reading

   2. Kernel Pool Required Reading

   3. Frames Required Reading

   4. High-Resolution Imager (HIRES) I-Kernel File "clem_hires_008.ti"

   5. Ultraviolet and Visible Imaging Camera (UVVIS) I-Kernel File 
      "clem_uvvis_008.ti"

   6. Near Infrared Mapping Spectrometer (NIR) I-Kernel File 
      "clem_nir_009.ti"

   7. Long Wavelength Infrared Mapping Spectrometer (LWIR) I-Kernel 
      File "clem_lwir_008.ti"

   8. Laser Ranger (LIDAR) I-Kernel File "clem_lidar_005.ti"

   9. Star Tracker A (ASTAR) I-Kernel File "clem_astar_006.ti"

  10. Star Tracker B (BSTAR) I-Kernel File "clem_bstar_006.ti"

  11. Charged Particle Telescope (CPT) I-Kernel File "clem_cpt_002.ti"

  12. "Post Launch Alignment and Geometric Calibration of the 
      Clementine Spacecraft and Remote Sensing Science Instruments",
      T. Duxbury, Rough Draft, no date

  13. "Models of the Clementine Spacecraft and Remote Sensing Science
      Instruments for Geodesy, Cartography, and Dynamical Sciences",
      Draft version 1.0, December 1993.
   

Contact Information
--------------------------------------------------------

   Boris V. Semenov, NAIF/JPL, (818)-354-8136, Boris.Semenov@jpl.nasa.gov


Implementation Notes
--------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must ``load'' the kernel, normally during
   program initialization (see [2]). The SPICELIB routine FURNSH
   (furnsh_c in CSPICE, cspice_furnsh in ICY) loads a kernel file into
   the pool as follows:

      CALL FURNSH  ( 'frame_kernel_name' )
      furnsh_c     ( "frame_kernel_name" );
      cspice_furnsh, "frame_kernel_name"

   This file was created and may be updated with a text editor or word
   processor. Should you need to update this kernel in any way,
   please, modify the "Version and Date" section above to reflect the
   changes.


Clementine Frames
--------------------------------------------------------

   The following Clementine frames are defined in this kernel file:

        Frame Name           Relative to          Type    NAIF ID
   =====================   ==================   =======   =======

   Spacecraft Bus Frame:
   ---------------------
      CLEM_SC_BUS          rel.to J2000         CK        -40000

   Instrument Frames:
   ------------------
      CLEM_HIRES           rel.to SC_BUS        FIXED     -40001
      CLEM_UVVIS           rel.to SC_BUS        FIXED     -40002
      CLEM_NIR             rel.to SC_BUS        FIXED     -40003
      CLEM_LWIR            rel.to SC_BUS        FIXED     -40004
      CLEM_LIDAR           rel.to SC_BUS        FIXED     -40005
      CLEM_ASTAR           rel.to SC_BUS        FIXED     -40006
      CLEM_BSTAR           rel.to SC_BUS        FIXED     -40007
      CLEM_CPT             rel.to SC_BUS        FIXED     -40008

   UVVIS Specific Frames:
   ----------------------
      CLEM_UVVIS_A         rel.to CLEM_UVVIS    FIXED     -40021
      CLEM_UVVIS_B         rel.to CLEM_UVVIS    FIXED     -40022
      CLEM_UVVIS_C         rel.to CLEM_UVVIS    FIXED     -40023
      CLEM_UVVIS_D         rel.to CLEM_UVVIS    FIXED     -40024
      CLEM_UVVIS_E         rel.to CLEM_UVVIS    FIXED     -40025
      CLEM_UVVIS_F         rel.to CLEM_UVVIS    FIXED     -40026


Clementine Frames Hierarchy
--------------------------------------------------------

   The diagram below shows Clementine frames hierarchy:


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
               |                                 "CLEM_ASTAR"  "CLEM_BSTAR"
               |                                 ------------  ------------
               |                                      ^           ^
               |                                      |           |
               V                                      |<--fixed   |<--fixed
          "CLEM_SC_BUS"                               |           |
       +----------------------------------------------------------+
       |           |           |          |           |           |
       |<--fixed   |<--fixed   |<-fixed   |<--fixed   |<--fixed   |<--fixed
       |           |           |          |           |           |
       V           V           V          V           V           V 
   "CLEM_HIRES" "CLEM_UVVIS" "CLEM_NIR" "CLEM_LWIR" "CLEM_LIDAR" "CLEM_CPT"
   ------------ ------------ ---------- ----------- ------------ ----------
                   |
                   |---> CLEM_UVVIS_A (fixed)
                   |
                   |---> CLEM_UVVIS_B (fixed)
                   |
                   |---> CLEM_UVVIS_C (fixed)
                   |
                   |---> CLEM_UVVIS_D (fixed)
                   |
                   |---> CLEM_UVVIS_E (fixed)
                   |
                   |---> CLEM_UVVIS_F (fixed)


Spacecraft Bus Frame
--------------------------------------------------------

   The Clementine spacecraft bus frame is defined by the spacecraft
   design as follows ([12]):

      *  +Z axis is in the direction of the nominal remote sensing
         science instrument boresight vectors;

      *  +X axis is in the direction of the main rocket engine;
 
      *  +Y axis completes the right handed frame and is in the plane
         of the solar arrays;
 
      *  the origin of this frame is at the geometric center of the
         spacecraft's octagonal prizm shaped body.
 
   This diagram illustrates the spacecraft frame:

                            ________________
                             `.           .'         
         _______ _______   -------------------   _______ _______ 
        |       |       | |    |         |    | |       |       |
        |       |       | |    | Science |    | |       |       |
        |       |       | |    |  Deck   |    | |       |       |
        |       |       | |    |         |    | |       |       |
        |       |       | |+Ysc|     +Zsc|    | |       |       |
        |       |       |=|  <------o    |    |=|       |       |
        |       |       | |    |    |    |    | |       |       |
        |       |       | |    |    |    |    | |       |       |
        |       |       | |    |    |    |    | |       |       |
        |       |       | |    |    V    |    | |       |       |
        |       |       | |    |     +Xsc|    | |       |       |
         ------- -------   -------------------   ------- ------- 
                                   |_|
                                  /   \
                                 .     .
                                 ._____.
 
   The spacecraft bus attitude with respect to an inertial frame is
   provided by CK kernels. Therefore the spacecraft frame is defined as
   a CK-based frame.

   \begindata

      FRAME_CLEM_SC_BUS         = -40000
      FRAME_-40000_NAME        = 'CLEM_SC_BUS'
      FRAME_-40000_CLASS       = 3
      FRAME_-40000_CLASS_ID    = -40000
      FRAME_-40000_CENTER      = -40
      CK_-40000_SCLK           = -40
      CK_-40000_SPK            = -40

   \begintext


Science Instrument Frames
--------------------------------------------------------

   All Clementine instrument frames are defined in the same way as
   follows ([12,13]):

      *  +Z axis is along the instrument boresight vector;
 
      *  +X axis is in the image plane and in the direction of
         increasing pixels;
 
      *  +Y axis is in the image plane and in the direction of
         increasing lines;

   Since all insruments are rigidly mounted on the spacecraft, the
   orientation of their frames is constant with respect to the
   spacecraft frame. Therefore, all instrument frames are defined as
   fixed offset frames with respect to the spacecraft frame.

   Originally mounting alignment of the instruments was included in
   the IK files by providing the orientation of the instrument frame
   relative to the spacecraft frame as three angles: thetaX, thetaY,
   and thetaZ. The spacecraft-to-instrument frame transformation,
   TXYZxyz, could derived from these angles as follows:

       TXYZxyz = R (thetaZ) R (thetaY) R (thetaX) 
                  Z          Y          X

   where R  (theta) represents a 3x3 rotation matrix of angle theta
          i
   about the i axis.

   The instrument frame definitions below incorporate the rotation
   angles from the latest versions of the IKs available at the time
   when this FK was created.


High-Resolution Imager (HIRES) Frame

   The rotation angles provided in the HIRES frame definition below are
   from [4].

   \begindata

      FRAME_CLEM_HIRES         = -40001
      FRAME_-40001_NAME        = 'CLEM_HIRES'
      FRAME_-40001_CLASS       = 4
      FRAME_-40001_CLASS_ID    = -40001
      FRAME_-40001_CENTER      = -40
      TKFRAME_-40001_SPEC      = 'ANGLES'
      TKFRAME_-40001_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40001_ANGLES    = ( -0.00504400153826, 
                                   -0.00125663706144, 
                                   -3.14333798284179  )
      TKFRAME_-40001_AXES      = ( 1, 2, 3 )
      TKFRAME_-40001_UNITS     = 'RADIANS'

   \begintext


Ultraviolet and Visible Imaging Camera (UVVIS) Frame

   The rotation angles provided in the UVVIS frame definition below are
   from [5]. Band specific frames are included to handle the different
   optical distortion parameters

   \begindata

      FRAME_CLEM_UVVIS         = -40002
      FRAME_-40002_NAME        = 'CLEM_UVVIS'
      FRAME_-40002_CLASS       = 4
      FRAME_-40002_CLASS_ID    = -40002
      FRAME_-40002_CENTER      = -40
      TKFRAME_-40002_SPEC      = 'ANGLES'
      TKFRAME_-40002_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40002_ANGLES    = ( 0.0, 0.0, 0.0 )
      TKFRAME_-40002_AXES      = ( 1, 2, 3 )
      TKFRAME_-40002_UNITS     = 'RADIANS'

      FRAME_CLEM_UVVIS_A       = -40021
      FRAME_-40021_NAME        = 'CLEM_UVVIS_A'
      FRAME_-40021_CLASS       = 4
      FRAME_-40021_CLASS_ID    = -40021
      FRAME_-40021_CENTER      = -40
      TKFRAME_-40021_SPEC      = 'ANGLES'
      TKFRAME_-40021_RELATIVE  = 'CLEM_UVVIS'
      TKFRAME_-40021_ANGLES    = ( -0.00553269372882, -0.00104719755120, 1.57690497917688 )
      TKFRAME_-40021_AXES      = ( 1, 2, 3 )
      TKFRAME_-40021_UNITS     = 'RADIANS'

      FRAME_CLEM_UVVIS_B       = -40022
      FRAME_-40022_NAME        = 'CLEM_UVVIS_B'
      FRAME_-40022_CLASS       = 4
      FRAME_-40022_CLASS_ID    = -40022
      FRAME_-40022_CENTER      = -40
      TKFRAME_-40022_SPEC      = 'ANGLES'
      TKFRAME_-40022_RELATIVE  = 'CLEM_UVVIS'
      TKFRAME_-40022_ANGLES    = ( -0.00553269372882, -0.00104719755120, 1.57690497917688 )
      TKFRAME_-40022_AXES      = ( 1, 2, 3 )
      TKFRAME_-40022_UNITS     = 'RADIANS'

      FRAME_CLEM_UVVIS_C       = -40023
      FRAME_-40023_NAME        = 'CLEM_UVVIS_C'
      FRAME_-40023_CLASS       = 4
      FRAME_-40023_CLASS_ID    = -40023
      FRAME_-40023_CENTER      = -40
      TKFRAME_-40023_SPEC      = 'ANGLES'
      TKFRAME_-40023_RELATIVE  = 'CLEM_UVVIS'
      TKFRAME_-40023_ANGLES    = ( -0.00553269372882, -0.00104719755120, 1.57690497917688 )
      TKFRAME_-40023_AXES      = ( 1, 2, 3 )
      TKFRAME_-40023_UNITS     = 'RADIANS'

      FRAME_CLEM_UVVIS_D       = -40024
      FRAME_-40024_NAME        = 'CLEM_UVVIS_D'
      FRAME_-40024_CLASS       = 4
      FRAME_-40024_CLASS_ID    = -40024
      FRAME_-40024_CENTER      = -40
      TKFRAME_-40024_SPEC      = 'ANGLES'
      TKFRAME_-40024_RELATIVE  = 'CLEM_UVVIS'
      TKFRAME_-40024_ANGLES    = ( -0.00553269372882, -0.00104719755120, 1.57690497917688 )
      TKFRAME_-40024_AXES      = ( 1, 2, 3 )
      TKFRAME_-40024_UNITS     = 'RADIANS'

      FRAME_CLEM_UVVIS_E       = -40025
      FRAME_-40025_NAME        = 'CLEM_UVVIS_E'
      FRAME_-40025_CLASS       = 4
      FRAME_-40025_CLASS_ID    = -40025
      FRAME_-40025_CENTER      = -40
      TKFRAME_-40025_SPEC      = 'ANGLES'
      TKFRAME_-40025_RELATIVE  = 'CLEM_UVVIS'
      TKFRAME_-40025_ANGLES    = ( -0.00553269372882, -0.00104719755120, 1.57690497917688 )
      TKFRAME_-40025_AXES      = ( 1, 2, 3 )
      TKFRAME_-40025_UNITS     = 'RADIANS'

      FRAME_CLEM_UVVIS_F       = -40026
      FRAME_-40026_NAME        = 'CLEM_UVVIS_F'
      FRAME_-40026_CLASS       = 4
      FRAME_-40026_CLASS_ID    = -40026
      FRAME_-40026_CENTER      = -40
      TKFRAME_-40026_SPEC      = 'ANGLES'
      TKFRAME_-40026_RELATIVE  = 'CLEM_UVVIS'
      TKFRAME_-40026_ANGLES    = ( -0.00553269372882, -0.00104719755120, 1.57690497917688 )
      TKFRAME_-40026_AXES      = ( 1, 2, 3 )
      TKFRAME_-40026_UNITS     = 'RADIANS'


   \begintext


Near Infrared Mapping Spectrometer (NIR) Frame

   The rotation angles provided in the frame NIT definition below are
   from [6].

   \begindata

      FRAME_CLEM_NIR           = -40003
      FRAME_-40003_NAME        = 'CLEM_NIR'
      FRAME_-40003_CLASS       = 4
      FRAME_-40003_CLASS_ID    = -40003
      FRAME_-40003_CENTER      = -40
      TKFRAME_-40003_SPEC      = 'ANGLES'
      TKFRAME_-40003_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40003_ANGLES    = ( -0.005525711614,
                                   -0.001685987814,
                                   -0.023816759530  )
      TKFRAME_-40003_AXES      = ( 1, 2, 3 )
      TKFRAME_-40003_UNITS     = 'RADIANS'

   \begintext


Long Wavelength Infrared Mapping Spectrometer (LWIR) Frame

   The rotation angles provided in the frame LWIR definition below are
   from [7].

   \begindata

      FRAME_CLEM_LWIR          = -40004
      FRAME_-40004_NAME        = 'CLEM_LWIR'
      FRAME_-40004_CLASS       = 4
      FRAME_-40004_CLASS_ID    = -40004
      FRAME_-40004_CENTER      = -40
      TKFRAME_-40004_SPEC      = 'ANGLES'
      TKFRAME_-40004_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40004_ANGLES    = ( -0.00053581608036,
                                   -0.00019722220548,
                                   -3.14159265358979  )
      TKFRAME_-40004_AXES      = ( 1, 2, 3 )
      TKFRAME_-40004_UNITS     = 'RADIANS'

   \begintext


Laser Ranger (LIDAR) Frame

   The rotation angles provided in the frame LIDAR definition below are
   from [8].

   \begindata

      FRAME_CLEM_LIDAR         = -40005
      FRAME_-40005_NAME        = 'CLEM_LIDAR'
      FRAME_-40005_CLASS       = 4
      FRAME_-40005_CLASS_ID    = -40005
      FRAME_-40005_CENTER      = -40
      TKFRAME_-40005_SPEC      = 'ANGLES'
      TKFRAME_-40005_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40005_ANGLES    = ( -0.00504400153826,
                                   -0.00125663706144,
                                    0.0               )
      TKFRAME_-40005_AXES      = ( 1, 2, 3 )
      TKFRAME_-40005_UNITS     = 'RADIANS'

   \begintext


Star Tracker A (ASTAR) Frame
 
   The rotation angles provided in the ASTAR frame definition below are
   from [9].

   \begindata

      FRAME_CLEM_ASTAR         = -40006
      FRAME_-40006_NAME        = 'CLEM_ASTAR'
      FRAME_-40006_CLASS       = 4
      FRAME_-40006_CLASS_ID    = -40006
      FRAME_-40006_CENTER      = -40
      TKFRAME_-40006_SPEC      = 'ANGLES'
      TKFRAME_-40006_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40006_ANGLES    = (  2.04022706372830,
                                   -0.52166146012859,
                                   -3.06113297507285  )
      TKFRAME_-40006_AXES      = ( 1, 2, 3 )
      TKFRAME_-40006_UNITS     = 'RADIANS'

   \begintext


Star Tracker B (BSTAR) Frame

   The rotation angles provided in the frame BSTAR definition below are
   from [10].

   \begindata

      FRAME_CLEM_BSTAR         = -40007
      FRAME_-40007_NAME        = 'CLEM_BSTAR'
      FRAME_-40007_CLASS       = 4
      FRAME_-40007_CLASS_ID    = -40007
      FRAME_-40007_CENTER      = -40
      TKFRAME_-40007_SPEC      = 'ANGLES'
      TKFRAME_-40007_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40007_ANGLES    = ( -2.05078106971511,
                                   -0.51341128875441,
                                   -0.08063421144214  )
      TKFRAME_-40007_AXES      = ( 1, 2, 3 )
      TKFRAME_-40007_UNITS     = 'RADIANS'

   \begintext


Charged Particle Telescope (CPT) Frame


   The rotation angles provided in the frame definition below are
   from [11].

   \begindata

      FRAME_CLEM_CPT           = -40008
      FRAME_-40008_NAME        = 'CLEM_CPT'
      FRAME_-40008_CLASS       = 4
      FRAME_-40008_CLASS_ID    = -40008
      FRAME_-40008_CENTER      = -40
      TKFRAME_-40008_SPEC      = 'ANGLES'
      TKFRAME_-40008_RELATIVE  = 'CLEM_SC_BUS'
      TKFRAME_-40008_ANGLES    = (  0.0, 
                                   -3.141592654,
                                    0.0          )
      TKFRAME_-40008_AXES      = ( 1, 2, 3 )
      TKFRAME_-40008_UNITS     = 'RADIANS' 

   \begintext


Clementine NAIF ID Codes Definitions
--------------------------------------------------------

   This section contains name - to - NAIF ID mappings for the Clementine
   mission. Once the contents of this file is loaded into the KERNEL
   POOL, these mappings become available within SPICE, making it
   possible to use these names in the high level SPICE routine calls.

   Spacecraft:
   -----------

      DSPSE                   -40
      CLEM                    -40
      CLEMENTINE_1            -40
      CLEMENTINE              -40
 
      CLEM_SPACECRAFT         -40000
      CLEM_SPACECRAFT_BUS     -40000
      CLEM_SC_BUS             -40000

   Science Instruments:
   --------------------

      CLEM_HIRES              -40001
      CLEM_UVVIS              -40002
      CLEM_NIR                -40003
      CLEM_LWIR               -40004
      CLEM_LIDAR              -40005
      CLEM_ASTAR              -40006
      CLEM_BSTAR              -40007
      CLEM_CPT                -40008
      
   UVVIS Specific Frames:
   ----------------------

      CLEM_UVVIS_A            -40021
      CLEM_UVVIS_B            -40022
      CLEM_UVVIS_C            -40023
      CLEM_UVVIS_D            -40024
      CLEM_UVVIS_E            -40025
      CLEM_UVVIS_F            -40026


   The mappings summarized in this table are implemented by the keywords
   below.

   \begindata

      NAIF_BODY_NAME += ( 'DSPSE'               )
      NAIF_BODY_CODE += ( -40                   )

      NAIF_BODY_NAME += ( 'CLEM'                )
      NAIF_BODY_CODE += ( -40                   )

      NAIF_BODY_NAME += ( 'CLEMENTINE_1'        )
      NAIF_BODY_CODE += ( -40                   )

      NAIF_BODY_NAME += ( 'CLEMENTINE'          )
      NAIF_BODY_CODE += ( -40                   )

      NAIF_BODY_NAME += ( 'CLEM_SPACECRAFT'     )
      NAIF_BODY_CODE += ( -40000                )

      NAIF_BODY_NAME += ( 'CLEM_SPACECRAFT_BUS' )
      NAIF_BODY_CODE += ( -40000                )

      NAIF_BODY_NAME += ( 'CLEM_SC_BUS'         )
      NAIF_BODY_CODE += ( -40000                )

      NAIF_BODY_NAME += ( 'CLEM_HIRES'          )
      NAIF_BODY_CODE += ( -40001                )

      NAIF_BODY_NAME += ( 'CLEM_UVVIS'          )
      NAIF_BODY_CODE += ( -40002                )

      NAIF_BODY_NAME += ( 'CLEM_NIR'            )
      NAIF_BODY_CODE += ( -40003                )

      NAIF_BODY_NAME += ( 'CLEM_LWIR'           )
      NAIF_BODY_CODE += ( -40004                )

      NAIF_BODY_NAME += ( 'CLEM_LIDAR'          )
      NAIF_BODY_CODE += ( -40005                )

      NAIF_BODY_NAME += ( 'CLEM_ASTAR'          )
      NAIF_BODY_CODE += ( -40006                )

      NAIF_BODY_NAME += ( 'CLEM_BSTAR'          )
      NAIF_BODY_CODE += ( -40007                )

      NAIF_BODY_NAME += ( 'CLEM_CPT'            )
      NAIF_BODY_CODE += ( -40008                )

      NAIF_BODY_NAME += ( 'CLEM_UVVIS_A'        )
      NAIF_BODY_CODE += ( -40021                )

      NAIF_BODY_NAME += ( 'CLEM_UVVIS_B'        )
      NAIF_BODY_CODE += ( -40022                )

      NAIF_BODY_NAME += ( 'CLEM_UVVIS_C'        )
      NAIF_BODY_CODE += ( -40023                )

      NAIF_BODY_NAME += ( 'CLEM_UVVIS_D'        )
      NAIF_BODY_CODE += ( -40024                )

      NAIF_BODY_NAME += ( 'CLEM_UVVIS_E'        )
      NAIF_BODY_CODE += ( -40025                )

      NAIF_BODY_NAME += ( 'CLEM_UVVIS_F'        )
      NAIF_BODY_CODE += ( -40026                )

   \begintext

