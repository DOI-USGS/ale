KPL/FK

\beginlabel
PDS_VERSION_ID               = PDS3
RECORD_TYPE                  = STREAM
RECORD_BYTES                 = "N/A"
^SPICE_KERNEL                = "SEL_V01.TF"
MISSION_NAME                 = "SELENE"
SPACECRAFT_NAME              = "SELENE-M"
DATA_SET_ID                  = "SEL-L-SPICE-6-V1.0"
KERNEL_TYPE_ID               = FK
PRODUCT_ID                   = "SEL_V01.TF"
PRODUCT_CREATION_TIME        = 2015-04-28T10:10:10
PRODUCER_ID                  = "ISAS/JAXA"
MISSION_PHASE_NAME           = "N/A"
PRODUCT_VERSION_TYPE         = ACTUAL
PLATFORM_OR_MOUNTING_NAME    = "SELENE-M"
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
  DESCRIPTION                = "SPICE FK file defining reference frames for
the SELENE-M spacecraft and its structures and instruments. "
END_OBJECT                   = SPICE_KERNEL
\endlabel


SELENE Main Spacecraft Frame Definitions Kernel
===============================================================================

   This frame kernel contains a set of frame definitions for the SELENE-M
   spacecraft.

Version and Date
---------------------------------------------------------------------

   Version 1.0 -- Apr. 28, 2015 -- Yoshiaki Ishihara, ISAS/JAXA


Implementation Notes
---------------------------------------------------------------------

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

SELENE Frames
-------------------------------------------------------------------------------
   The following names and NAIF ID codes are assigned to the SELENE-M
   spacecraft, its structure, and science instruments frames:
	
            NAME                     Relative to                    NAIF ID
    ====================        =========================        ============
           SELENE                                                   -131
           SELENE_M                                                 -131

    Spacecraft Bus Frame:
    ---------------------
    SELENE_M_SPACECRAFT	             rel.to J2000                   -131000

	Instrument Frames:
    --------------------
        XRS_XRFA                rel.to SELENE_M_SPACECRAFT          -131110
        XRS_SOLB_S0             rel.to SELENE_M_SPACECRAFT          -131120
        XRS_SOLB_S1             rel.to SELENE_M_SPACECRAFT          -131121
        XRS_SOLC                rel.to SELENE_M_SPACECRAFT          -131130
        LISM_LRU_SP             rel.to SELENE_M_SPACECRAFT          -131310
        LISM_LRU_TC_MI          rel.to SELENE_M_SPACECRAFT          -131320
        LISM_MI_V_HEAD          rel.to LISM_LRU_TC_MI               -131330
        LISM_MI_N_HEAD          rel.to LISM_LRU_TC_MI               -131340
        LISM_TC1_HEAD           rel.to LISM_LRU_TC_MI               -131350
        LISM_TC2_HEAD           rel.to LISM_LRU_TC_MI               -131370
        SELENE_LALT             rel.to SELENE_M_SPACECRAFT          -131501
        SELENE_HDTV             rel.to SELENE_M_SPACECRAFT          -131800
        SELENE_HDTV_TELE        rel.to SELENE_HDTV                  -131810
        SELENE_HDTV_WIDE        rel.to SELENE_HDTV                  -131820


SELENE Frames Hierarchy
-------------------------------------------------------------------------------
    The diagram below shows SELENE frames hierarchy:

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
	V
   "SELENE_M_SPACECRAFT"
  +----------------------------------------------------------------------+
  |    |              |                |             |            |          
  |    |<--fixed      |<--fixed        |<-fixed      |<--fixed    |<--fixed  
  |    |              |                |             |            |         
  |    V              V                V             V            V          
  |  "XRS_XRFA"	  "XRS_SOLB_S0"	  "XRS_SOLB_S1"  "XRS_SOLC"   "LISM_LRU_SP" 	
  |  ----------   -------------   ------------   ----------   ------------  
  |
  +----------------------------------------------------------------------+
  |                                 |                   |
  |<--fixed                         |<--fixed           |<-fixed
  |                                 |                   |
  V                                 V                   V
 "LISM_LRU_TC_MI"                  "SELENE_LALT"       "SELENE_HDTV"
 ----------------                  -------------       -------------
 |    |    |    |                                      |           |
 |    |    |    V                                      |           V
 |    |    |  "LISM_TC2_HEAD"                          |   "SELENE_HDTV_WIDE"  
 |    |    |  ---------------                          |   ------------------
 |    |    V                                           V
 |    |  "LISM_TC1_HEAD"                             "SELENE_HDTV_TELE"
 |    |  ---------------                             ------------------
 |    V
 | "LISM_MI_N_HEAD"
 | ---------------
 V
 "LISM_MI_V_HEAD"
 ----------------


SELENE_M_SPACECRAFT Frame
---------------------------------------------------------------------------

  SELENE_M_SPACECRAFT defines a frame fixed to the body of SELENE_M, which
  is agreeable to the coordinate system defined in the developments: solar
  paddle on -Y, high gain antenna -Z, and main thruster -X. In normal 
  operations SELENE moves to +X direction seeing the lunar surface in +Z.

\begindata

   FRAME_SELENE_M_SPACECRAFT = -131000
   FRAME_-131000_NAME        = 'SELENE_M_SPACECRAFT'
   FRAME_-131000_CLASS       =  3
   FRAME_-131000_CLASS_ID    = -131000
   FRAME_-131000_CENTER      = -131
   CK_-131000_SCLK           = -131
   CK_-131000_SPK            = -131

\begintext


Science Instruments Frames
---------------------------------------------------------------------------

\begindata

   FRAME_XRS_XRFA            =  -131110
   FRAME_-131110_NAME        = 'XRS_XRFA'
   FRAME_-131110_CLASS       =  4
   FRAME_-131110_CLASS_ID    =  -131110
   FRAME_-131110_CENTER      =  -131
   FRAME_-131110_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131110_SPEC        = 'ANGLES'
   FRAME_-131110_UNITS       = 'DEGREES'
   FRAME_-131110_AXES        = ( 1,    1,    1   )
   FRAME_-131110_ANGLES      = ( 0.0,  0.0,  0.0 )

   FRAME_XRS_SOLB_S0         =  -131120
   FRAME_-131120_NAME        = 'XRS_SOLB_S0'
   FRAME_-131120_CLASS       =  4
   FRAME_-131120_CLASS_ID    =  -131120
   FRAME_-131120_CENTER      =  -131
   FRAME_-131120_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131120_SPEC        = 'ANGLES'
   FRAME_-131120_UNITS       = 'DEGREES'
   FRAME_-131120_AXES        = (  1,       2,       1   )
   FRAME_-131120_ANGLES      = ( -135.0,   45.0,    0.0 )

   FRAME_XRS_SOLB_S1         =  -131121
   FRAME_-131121_NAME        = 'XRS_SOLB_S1'
   FRAME_-131121_CLASS       =  4
   FRAME_-131121_CLASS_ID    =  -131121
   FRAME_-131121_CENTER      =  -131
   FRAME_-131121_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131121_SPEC        = 'ANGLES'
   FRAME_-131121_UNITS       = 'DEGREES'
   FRAME_-131121_AXES        = (  1,       2,       1   )
   FRAME_-131121_ANGLES      = ( -135.0,  -45.0,    0.0 )

   FRAME_XRS_SOLC            =  -131130
   FRAME_-131130_NAME        = 'XRS_SOLC'
   FRAME_-131130_CLASS       =  4
   FRAME_-131130_CLASS_ID    =  -131130
   FRAME_-131130_CENTER      =  -131
   FRAME_-131130_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131130_SPEC        = 'ANGLES'
   FRAME_-131130_UNITS       = 'DEGREES'
   FRAME_-131130_AXES        = ( 1,       1,       1   )
   FRAME_-131130_ANGLES      = ( 180.0,   0.0,     0.0 )

   FRAME_LISM_LRU_SP         =  -131310
   FRAME_-131310_NAME        = 'LISM_LRU_SP'
   FRAME_-131310_CLASS       =  4
   FRAME_-131310_CLASS_ID    =  -131310
   FRAME_-131310_CENTER      =  -131
   FRAME_-131310_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131310_SPEC        = 'ANGLES'
   FRAME_-131310_UNITS       = 'DEGREES'
   FRAME_-131310_AXES        = ( 3,   2,   1   )
   FRAME_-131310_ANGLES      = ( 0.1429, -0.0202, 0.0689 )

   FRAME_LISM_LRU_TC_MI      =  -131320
   FRAME_-131320_NAME        = 'LISM_LRU_TC_MI'
   FRAME_-131320_CLASS       =  4
   FRAME_-131320_CLASS_ID    =  -131320
   FRAME_-131320_CENTER      =  -131
   FRAME_-131320_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131320_SPEC        = 'ANGLES'
   FRAME_-131320_UNITS       = 'DEGREES'
   FRAME_-131320_AXES        = ( 3,   2,   1   )
   FRAME_-131320_ANGLES      = ( 0.0099, 0.0396, -0.0179 )

   FRAME_LISM_MI_V_HEAD      =  -131330
   FRAME_-131330_NAME        = 'LISM_MI_V_HEAD'
   FRAME_-131330_CLASS       =  4
   FRAME_-131330_CLASS_ID    =  -131330
   FRAME_-131330_CENTER      =  -131
   FRAME_-131330_RELATIVE    = 'LISM_LRU_TC_MI'
   FRAME_-131330_SPEC        = 'ANGLES'
   FRAME_-131330_UNITS       = 'DEGREES'
   FRAME_-131330_AXES        = ( 3,   2,   1   )
   FRAME_-131330_ANGLES      = ( 0.0138, -0.0486, -0.0536 )

   FRAME_LISM_MI_N_HEAD      =  -131340
   FRAME_-131340_NAME        = 'LISM_MI_N_HEAD'
   FRAME_-131340_CLASS       =  4
   FRAME_-131340_CLASS_ID    =  -131340
   FRAME_-131340_CENTER      =  -131
   FRAME_-131340_RELATIVE    = 'LISM_LRU_TC_MI'
   FRAME_-131340_SPEC        = 'ANGLES'
   FRAME_-131340_UNITS       = 'DEGREES'
   FRAME_-131340_AXES        = ( 3,   2,   1   )
   FRAME_-131340_ANGLES      = ( 0.0232, -0.0068, -0.0332 )

   FRAME_LISM_TC1_HEAD       =  -131350
   FRAME_-131350_NAME        = 'LISM_TC1_HEAD'
   FRAME_-131350_CLASS       =  4
   FRAME_-131350_CLASS_ID    =  -131350
   FRAME_-131350_CENTER      =  -131
   FRAME_-131350_RELATIVE    = 'LISM_LRU_TC_MI'
   FRAME_-131350_SPEC        = 'ANGLES'
   FRAME_-131350_UNITS       = 'DEGREES'
   FRAME_-131350_AXES        = ( 3,   2,   1   )
   FRAME_-131350_ANGLES      = ( 0.04903647, -14.97465222, 0.05110624 )

   FRAME_LISM_TC2_HEAD       =  -131370
   FRAME_-131370_NAME        = 'LISM_TC2_HEAD'
   FRAME_-131370_CLASS       =  4
   FRAME_-131370_CLASS_ID    =  -131370
   FRAME_-131370_CENTER      =  -131
   FRAME_-131370_RELATIVE    = 'LISM_LRU_TC_MI'
   FRAME_-131370_SPEC        = 'ANGLES'
   FRAME_-131370_UNITS       = 'DEGREES'
   FRAME_-131370_AXES        = ( 3,   2,   1   )
   FRAME_-131370_ANGLES      = ( -0.04379912, 15.01058661, 0.09222246 )
   
   FRAME_SELENE_LALT         =  -131501
   FRAME_-131501_NAME        = 'SELENE_LALT'
   FRAME_-131501_CLASS       =  4
   FRAME_-131501_CLASS_ID    =  -131501
   FRAME_-131501_CENTER      =  -131
   FRAME_-131501_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131501_SPEC        = 'ANGLES'
   FRAME_-131501_UNITS       = 'DEGREES'
   FRAME_-131501_AXES        = ( 1,    1,    1   )
   FRAME_-131501_ANGLES      = ( 0.0,  0.0,  0.0 )
   
   FRAME_SELENE_HDTV         = -131800
   FRAME_-131800_NAME        = 'SELENE_HDTV'
   FRAME_-131800_CLASS       = 4
   FRAME_-131800_CLASS_ID    = -131800
   FRAME_-131800_CENTER      = -131
   FRAME_-131800_SPEC        = 'ANGLES'
   FRAME_-131800_RELATIVE    = 'SELENE_M_SPACECRAFT'
   FRAME_-131800_ANGLES      = (   0.0,   0.0  , 0.0 )
   FRAME_-131800_AXES        = (     3,     2,     3 )
   FRAME_-131800_UNITS       = 'DEGREES'
   
   FRAME_SELENE_HDTV_TELE    = -131810
   FRAME_-131810_NAME        = 'SELENE_HDTV_TELE'
   FRAME_-131810_CLASS       = 4
   FRAME_-131810_CLASS_ID    = -131810
   FRAME_-131810_CENTER      = -131
   FRAME_-131810_SPEC        = 'ANGLES'
   FRAME_-131810_RELATIVE    = 'SELENE_HDTV'
   FRAME_-131810_ANGLES      = (   0.0, -71.5, -90.0 )
   FRAME_-131810_AXES        = (     3,     2,     3 )
   FRAME_-131810_UNITS       = 'DEGREES'
   
   FRAME_SELENE_HDTV_WIDE    = -131820
   FRAME_-131820_NAME        = 'SELENE_HDTV_WIDE'
   FRAME_-131820_CLASS       = 4
   FRAME_-131820_CLASS_ID    = -131820
   FRAME_-131820_CENTER      = -131
   FRAME_-131820_SPEC        = 'ANGLES'
   FRAME_-131820_RELATIVE    = 'SELENE_HDTV'
   FRAME_-131820_ANGLES      = (   0.0,  67.5, 90.0 )
   FRAME_-131820_AXES        = (     3,     2,    3 )
   FRAME_-131820_UNITS       = 'DEGREES'

\begintext

SELENE NAIF ID Codes -- Definitions
------------------------------------------
  
  This section contains name to NAIF ID mappings for the SELENE mission.
  Once the contents of this file is loaded into the KERNEL POOL,
  these mappings become available within SPICE, making it possible to use
  names insted of ID code in the high level SPICE routine calls.
  
  The set of codes below is not complete. Additional ID codes for
  some SELENE instruments are defined in the IK files.

  Spacecraft:
  -----------
    SELENE                       -131
    SELENE_M                     -131
    SELENE_M_SPACECRAFT          -131000

  Science Instruments:
  --------------------
    XRS_XRFA                     -131110
    XRS_SOLB_S0                  -131120
    XRS_SOLB_S1                  -131121
    XRS_SOLC                     -131130    
    LISM_LRU_SP                  -131310
    LISM_LRU_TC_MI               -131320
    LISM_MI_V_HEAD               -131330
    LISM_MI_N_HEAD               -131340
    LISM_TC1_HEAD                -131350
    LISM_TC2_HEAD                -131360
    SELENE_LALT                  -131501
    SELENE_HDTV                  -131800
    SELENE_HDTV_TELE             -131810
    SELENE_HDTV_WIDE             -131820

  The mappings summarized in this table are implemented by the keywors below.

\begindata
   
   NAIF_BODY_NAME                += ( 'SELENE' )
   NAIF_BODY_CODE                += ( -131 )

   NAIF_BODY_NAME                += ( 'SELENE_M' )
   NAIF_BODY_CODE                += ( -131 )

   NAIF_BODY_NAME                += ( 'SELENE_M_SPACECRAFT' )
   NAIF_BODY_CODE                += ( -131000 )

   NAIF_BODY_NAME                += ( 'XRS_XRFA' )
   NAIF_BODY_CODE                += ( -131110 )

   NAIF_BODY_NAME                += ( 'XRS_SOLB_S0' )
   NAIF_BODY_CODE                += ( -131120 )

   NAIF_BODY_NAME                += ( 'XRS_SOLB_S1' )
   NAIF_BODY_CODE                += ( -131121 )

   NAIF_BODY_NAME                += ( 'XRS_SOLC' )
   NAIF_BODY_CODE                += ( -131130 )

   NAIF_BODY_NAME                += ( 'LISM_LRU_SP' )
   NAIF_BODY_CODE                += ( -131310 )

   NAIF_BODY_NAME                += ( 'LISM_LRU_TC_MI' )
   NAIF_BODY_CODE                += ( -131320 )

   NAIF_BODY_NAME                += ( 'LISM_MI_V_HEAD' )
   NAIF_BODY_CODE                += ( -131330 )

   NAIF_BODY_NAME                += ( 'LISM_MI_N_HEAD' )
   NAIF_BODY_CODE                += ( -131340 )

   NAIF_BODY_NAME                += ( 'LISM_TC1_HEAD' )
   NAIF_BODY_CODE                += ( -131350 )

   NAIF_BODY_NAME                += ( 'LISM_TC2_HEAD' )
   NAIF_BODY_CODE                += ( -131370 )

   NAIF_BODY_NAME                += ( 'SELENE_LALT' )
   NAIF_BODY_CODE                += ( -131501 )

   NAIF_BODY_NAME                += ( 'SELENE_HDTV' )
   NAIF_BODY_CODE                += ( -131800 )

   NAIF_BODY_NAME                += ( 'SELENE_HDTV_TELE' )
   NAIF_BODY_CODE                += ( -131810 )

   NAIF_BODY_NAME                += ( 'SELENE_HDTV_WIDE' )
   NAIF_BODY_CODE                += ( -131820 )

\begintext

End of FK file.
