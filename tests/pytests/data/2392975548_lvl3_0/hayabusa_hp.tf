KPL/FK

\beginlabel
PDS_VERSION_ID               = PDS3
RECORD_TYPE                  = STREAM
RECORD_BYTES                 = "N/A"
^SPICE_KERNEL                = "hayabusa_hp.tf"
MISSION_NAME                 = HAYABUSA
SPACECRAFT_NAME              = HAYABUSA
DATA_SET_ID                  = "HAY-A-SPICE-6-V1.0"
KERNEL_TYPE_ID               = FK
PRODUCT_ID                   = "hayabusa_hp.tf"
PRODUCT_CREATION_TIME        = 2009-11-30T13:50:41
PRODUCER_ID                  = JAXA
MISSION_PHASE_NAME           = "N/A"
PRODUCT_VERSION_TYPE         = ACTUAL
PLATFORM_OR_MOUNTING_NAME    = "N/A"
START_TIME                   = "N/A"
STOP_TIME                    = "N/A"
SPACECRAFT_CLOCK_START_COUNT = "N/A"
SPACECRAFT_CLOCK_STOP_COUNT  = "N/A"
TARGET_NAME                  = "25143 ITOKAWA"
INSTRUMENT_NAME              = "N/A"
NAIF_INSTRUMENT_ID           = "N/A"
SOURCE_PRODUCT_ID            = "N/A"
NOTE                         = "See comments in the file for details"
OBJECT                       = SPICE_KERNEL
  INTERCHANGE_FORMAT         = ASCII
  KERNEL_TYPE                = FRAMES
  DESCRIPTION                = "SPICE Frames Kernel (FK) file containing
the definitions of the Hayabusa spacecraft frame and mission-specific
dynamic frames, created by the Hayabusa Joint Science Team. The original
name of this file was HAYABUSA_HP.TF. "
END_OBJECT                   = SPICE_KERNEL
\endlabel


Hayabusa Frames Definition Kernel
===========================================================================

   This frame kernel contains a set of frame definitions for the 
   Hayabusa mission. It also contains NAIF name-ID mappings for 
   Hayabusa instruments.
   

Version and Date
---------------------------------------------------------------------------

   Version 1.0 -- Sep. 22, 2009 -- Boris Semenov, NAIF/JPL.

      Lowercased file name (HAYABUSA_HP.TF -> hayabusa_hp.tf);

      Corrected spacecraft name to be consistent with the duplicate
      definitions provided in IKs (HAYABUSA_SC_BUS ->
      HAYABUSA_SC_BUS_PRIME);

      Added name-ID mapping keywords for science instruments;

      Added comments describing frames defined in the IK and 
      Itokawa FK;

      Added other miscellaneous comments.

   Version 1.0 -- Sep. 19, 2005 -- Naru Hirata, Aizu Univ.


Hayabusa NAIF IDs
---------------------------------------------------------------------------

   The following names and NAIF ID codes are assigned to the Hayabusa 
   spacecraft, its structures, and science instruments:

      NAME                                    NAIF ID
      -----------------------------------------------
      HAYABUSA                                   -130
      HAYABUSA_AMICA                          -130102
      HAYABUSA_NIRS                           -130200
      HAYABUSA_LIDAR                          -130300
      HAYABUSA_XRS                            -130400

   The keywords below implement the Hayabusa name-ID mappings.

      \begindata

         NAIF_BODY_NAME += ( 'HAYABUSA' )
         NAIF_BODY_CODE += ( -130 )

         NAIF_BODY_NAME += ( 'HAYABUSA_AMICA' )
         NAIF_BODY_CODE += ( -130102 )

         NAIF_BODY_NAME += ( 'HAYABUSA_NIRS' )
         NAIF_BODY_CODE += ( -130200 )

         NAIF_BODY_NAME += ( 'HAYABUSA_LIDAR' )
         NAIF_BODY_CODE += ( -130300 )

         NAIF_BODY_NAME += ( 'HAYABUSA_XRS' )
         NAIF_BODY_CODE += ( -130400 )

      \begintext


Hayabusa Reference Frames
---------------------------------------------------------------------------

   The following reference frames for Hayabusa spacecraft, its
   structures, and science instruments are defined in this FK and other
   Hayabusa kernels (Itokawa FK, PCKs, and IKs):


   Itokawa body-fixed frame (defined in a separate FK and PCKs):
   -------------------------------------------------------------

           Name                  Relative to           Type         Frame ID
      ======================  =====================  ============   ========
      ITOKAWA_FIXED           J2000                  PCK             2025143


   Hayabusa spacecraft frame (duplicated in the IKs):
   ----------------------------------------------------
           
           Name                  Relative to           Type         Frame ID
      ======================  =====================  ============   ========
      HAYABUSA_SC_BUS_PRIME   J2000                  CK              -130000


   Hayabusa Science instrument frames (defined in the instrument IK files):
   ------------------------------------------------------------------------

           Name                  Relative to           Type         Frame ID
      ======================  =====================  ============   ========
      HAYABUSA_AMICA_IDEAL    HAYABUSA_SC_BUS_PRIME  FIXED           -130101
      HAYABUSA_AMICA          HAYABUSA_AMICA         FIXED           -130102

      HAYABUSA_NIRS_IDEAL     HAYABUSA_SC_BUS_PRIME  FIXED           -130201
      HAYABUSA_NIRS           HAYABUSA_NIRS_IDEAL    FIXED           -130200

      HAYABUSA_LIDAR_IDEAL    HAYABUSA_SC_BUS_PRIME  FIXED           -130301
      HAYABUSA_LIDAR          HAYABUSA_LIDAR_IDEAL   FIXED           -130300

      HAYABUSA_XRS_IDEAL      HAYABUSA_SC_BUS_PRIME  FIXED           -130410
      HAYABUSA_XRS            HAYABUSA_XRS_IDEAL     FIXED           -130400


   Hayabusa-specific dynamic frames:
   ---------------------------------

           Name                  Relative to           Type         Frame ID
      ======================  =====================  ============   ========
      HAYABUSA_HP_FRAME       J2000                  DYNAMIC         -130900
      ITOKAWA_-Z_SUN_+X       J2000                  DYNAMIC         -130910
      EARTH_ITOKAWA_P         J2000                  DYNAMIC         -130920
   

Hayabusa Reference Frame Tree
---------------------------------------------------------------------------


   This diagram shows the Hayabusa frame tree:


                               "J2000" INERTIAL
         +-----------------------------------------------------------+
         |                   |         |         |                   |
         |<-dynamic          |         |         |          dynamic->|
         |                   |         |         |                   |
         V                   |         |         |                   V
    "HAYABUSA_HP_FRAME"      |         |         |        "ITOKAWA_-Z_SUN_+X"
    -------------------      |         |         |        -------------------
                             |         |         |
                             |         |         |
                    dynamic->|         |         |<-pck
                             |         |         |
                             V         |         V
                 "EARTH_ITOKAWA_P"     |   "ITOKAWA_FIXED"
                 -----------------     |   ---------------
                                       |
                                       |
                                       |<-ck
                                       |
                                       V
                           "HAYABUSA_SC_BUS_PRIME"
         +-----------------------------------------------------------+
         |                   |                   |                   |
         |                   |<-fixed            |                   |<-fixed
         |                   |                   |                   |
         |                   V                   |                   V
         |         "HAYABUSA_NIRS_IDEAL"         |        "HAYABUSA_XRS_IDEAL"
         |         ---------------------         |        --------------------
         |                   |                   |                   |
         |                   |<-fixed            |                   |<-fixed
         |                   |                   |                   |
         |                   V                   |                   V
         |            "HAYABUSA_NIRS"            |             "HAYABUSA_XRS"
         |            ---------------            |             --------------
         |                                       | 
         |                                       | 
         |<-fixed                                |<-fixed 
         |                                       | 
         V                                       V
   "HAYABUSA_AMICA_IDEAL"               "HAYABUSA_LIDAR_IDEAL"
   ----------------------               ----------------------
         |                                       | 
         |<-fixed                                |<-fixed 
         |                                       | 
         V                                       V
     "HAYABUSA_AMICA"                      "HAYABUSA_LIDAR"
     ----------------                      ----------------


Itokawa Body-Fixed Frame
---------------------------------------------------------------------------

   The Itokawa body-fixed frame, ITOKAWA_FIXED, is defined in a 
   separate FK file (itokawa_fixed.tf). Its definition is also
   duplicated in some of the Itokawa PCK files.


Hayabusa Spacecraft Frame
---------------------------------------------------------------------------

   The Hayabusa spacecraft frame, HAYABUSA_SC_BUS_PRIME, is defined 
   as follows:

      - +Z axis is parallel to the nominal HGA boresight;

      - +X axis is parallel to the ion engines thrust vector;

      - +Y axis completes the right-handed frame;

   The Hayabusa spacecraft frame shown on this diagram:

                                      ^
                                      | HGA Boresight
                                      |

                                      ^ 
                                     / \  HGA
                            --------------------
                             \                 /
                              \        +Z     /
                         --------     ^    ---------
                         |            |            | Ion Engines
                         |            |            |_
                         |            |      +X    |_]  
                         |            x----->      |_]   ----->
                        _|          +Y             |     Thrust Vector
                  NIRS | |                         |
                          ----| |---------------| |-
                           | | LIDAR             XRS
                           | |
                  Sample   | |        |
                   horn    | |        |  Science Instrument
                          /   \       V     Boresights


   This frame is defined as a CK-based frame below. The frame
   definition is also duplicated in Hayabusa science instrument IKs.

      \begindata

         FRAME_HAYABUSA_SC_BUS_PRIME  = -130000
         FRAME_-130000_NAME           = 'HAYABUSA_SC_BUS_PRIME'
         FRAME_-130000_CLASS          = 3
         FRAME_-130000_CLASS_ID       = -130000
         FRAME_-130000_CENTER         = -130
         CK_-130000_SCLK              = -130
         CK_-130000_SPK               = -130

      \begintext


Hayabusa Science Instrument Frames
---------------------------------------------------------------------------

   Hayabusa science instrument frames are defined in the IK files.


Hayabusa-Specific Dynamic Frames
---------------------------------------------------------------------------

   The HAYABUSA_HP_FRAME frame, used as the reference frame in the
   Hayabusa SPKs produces by JAXA, is defined as follows:

      - +Z axis is along the geometric direction from Itokawa to the
        Earth

      - +X axis is as close as possible to the geometric direction 
        from Itokawa to the Sun

      - +Y axis completes the right-handed frame;

   This frame is defined as a two-vector style dynamic frame below.
   
      \begindata

         FRAME_HAYABUSA_HP            = -130900
         FRAME_-130900_NAME           = 'HAYABUSA_HP'
         FRAME_-130900_CLASS          = 5
         FRAME_-130900_CLASS_ID       = -130900
         FRAME_-130900_CENTER         = 2025143
         FRAME_-130900_RELATIVE       = 'J2000'
         FRAME_-130900_DEF_STYLE      = 'PARAMETERIZED'
         FRAME_-130900_FAMILY         = 'TWO-VECTOR'
         FRAME_-130900_PRI_AXIS       = 'Z'
         FRAME_-130900_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
         FRAME_-130900_PRI_OBSERVER   = 2025143
         FRAME_-130900_PRI_TARGET     = 'EARTH'
         FRAME_-130900_PRI_ABCORR     = 'NONE'
         FRAME_-130900_SEC_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
         FRAME_-130900_SEC_OBSERVER   = 2025143
         FRAME_-130900_SEC_TARGET     = 'SUN'
         FRAME_-130900_SEC_AXIS       = 'X'
         FRAME_-130900_SEC_ABCORR     = 'NONE'

      \begintext


   The ITOKAWA_-Z_SUN_+X frame, representing the nominal Itokawa-pointed
   orientation, is defined as follows:

      - -Z axis is along the geometric direction from Hayabusa to
        Itokawa

      - +X axis is as close as possible to the geometric direction from
        Hayabusa to the Sun

      - +Y axis completes the right-handed frame;

   This frame is defined as a two-vector style dynamic frame below.

      \begindata

         FRAME_ITOKAWA_-Z_SUN_+X      = -130910
         FRAME_-130910_NAME           = 'ITOKAWA_-Z_SUN_+X'
         FRAME_-130910_CLASS          = 5
         FRAME_-130910_CLASS_ID       = -130910
         FRAME_-130910_CENTER         = -130
         FRAME_-130910_RELATIVE       = 'J2000'
         FRAME_-130910_DEF_STYLE      = 'PARAMETERIZED'
         FRAME_-130910_FAMILY         = 'TWO-VECTOR'
         FRAME_-130910_PRI_AXIS       = '-Z'
         FRAME_-130910_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
         FRAME_-130910_PRI_OBSERVER   = -130
         FRAME_-130910_PRI_TARGET     = 2025143
         FRAME_-130910_PRI_ABCORR     = 'NONE'
         FRAME_-130910_SEC_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
         FRAME_-130910_SEC_OBSERVER   = -130
         FRAME_-130910_SEC_TARGET     = 'SUN'
         FRAME_-130910_SEC_AXIS       = 'X'
         FRAME_-130910_SEC_ABCORR     = 'NONE'

      \begintext


   The EARTH_ITOKAWA_P frame is defined as follows:

      - +Z axis is along the geometric direction from the Earth to
        Itokawa

      - +Y axis is as close as possible to the geometric direction from
        the Earth to the Sun

      - +X axis completes the right-handed frame;

   This frame is defined as a two-vector style dynamic frame below.

      \begindata

         FRAME_EARTH_ITOKAWA_P        = -130920
         FRAME_-130920_NAME           = 'EARTH_ITOKAWA_P'
         FRAME_-130920_CLASS          = 5
         FRAME_-130920_CLASS_ID       = -130920
         FRAME_-130920_CENTER         = 399
         FRAME_-130920_RELATIVE       = 'J2000'
         FRAME_-130920_DEF_STYLE      = 'PARAMETERIZED'
         FRAME_-130920_FAMILY         = 'TWO-VECTOR'
         FRAME_-130920_PRI_AXIS       = 'Z'
         FRAME_-130920_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
         FRAME_-130920_PRI_OBSERVER   = 'EARTH'
         FRAME_-130920_PRI_TARGET     = 2025143
         FRAME_-130920_PRI_ABCORR     = 'NONE'
         FRAME_-130920_SEC_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
         FRAME_-130920_SEC_OBSERVER   = 'EARTH'
         FRAME_-130920_SEC_TARGET     = 'SUN'
         FRAME_-130920_SEC_AXIS       = 'Y'
         FRAME_-130920_SEC_ABCORR     = 'NONE'

     \begintext


End of FK file.
