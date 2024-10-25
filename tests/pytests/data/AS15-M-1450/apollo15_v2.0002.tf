KPL/FK

Apollo 15 Frames Kernel
===============================================================================

   This frame kernel contains the complete set of frame definitions for
   the Apollo 15 spacecraft including definitions for the
   spacecraft frame, instrument platform frame, and science instrument
   frames.


Version and Date
-------------------------------------------------------------------------------
   
   Version 2.0 -- Februrary 23, 2013 - Ken Edmundson
      Modified definition of Nadir Frame such that the secondary axis
      (FRAME_1400015_SEC_AXIS) is 'Y'. In Version 1.0 it was 'X'. Also changed
      the OBSERVER_TARGET_POSITION such that the observer is defined as the
      Moon and the target as the Apollo 15 spacecraft. It was the opposite in
      Version 1.0.
   
   Version 1.0 -- December 13, 2006 - Jacob Danton



References
-------------------------------------------------------------------------------

   1. 


A15 Frames
-------------------------------------------------------------------------------

   The following A15 frames are defined in this kernel file:

           Name                  Relative to           Type       NAIF ID
      ======================  ===================  ============   =======

   Spacecraft frame:
   -----------------
      A15_SPACECRAFT          rel.to B1950         CK             -915000

   Instrument Platform frame:
   --------------------------
      A15_PLATFORM            rel.to B1950         CK             -915001

   Science Instrument frames:
   --------------------------
      A15_METRIC              rel.to PLATFORM      FIXED          -915240
      A15_PANORAMIC           rel.to PLATFORM      FIXED          -915230


A15 Frames Hierarchy
-------------------------------------------------------------------------------

   The diagram below shows A15 frames hierarchy:


                               "B1950" INERTIAL
        +-----------------------------------------------------------+
        |                   |                   |                   |
        | <--pck            | <--ck             | <--ck             | <--pck
        |                   |                   |                   | 
        V                   V                   |                   V
    "IAU_MOON"      "A15_SPACECRAFT"            |              "IAU_EARTH"
    -----------     ----------------            |              ------------
                                                |
                                                |
                                                V
                                          "A15_PLATFORM"                       
                                     +---------------------+
                                     |                     |
                                     | <--fixed            | <--fixed
                                     |                     |
                                     V                     V
                                "A15_METRIC"        "A15_PANORAMIC"
                                 ----------            ----------



Spacecraft Bus Frame
-------------------------------------------------------------------------------

   \begindata

      FRAME_A15_SPACECRAFT     = -915000
      FRAME_-915000_NAME        = 'A15_SPACECRAFT'
      FRAME_-915000_CLASS       = 3
      FRAME_-915000_CLASS_ID    = -915000
      FRAME_-915000_CENTER      = -915
      CK_-915000_SCLK           = -915
      CK_-915000_SPK            = -915

   \begintext

Metric and Panoramic Frames
-------------------------------------------------------------------------------
   
   \begindata

      FRAME_A15_METRIC         = -915240
      FRAME_-915240_NAME        = 'A15_METRIC'
      FRAME_-915240_CLASS       = 3
      FRAME_-915240_CLASS_ID    = -915240
      FRAME_-915240_CENTER      = -915
      CK_-915240_SCLK           = -915
      CK_-915240_SPK            = -915

      FRAME_A15_PANORAMIC      = -915230
      FRAME_-915230_NAME        = 'A15_PANORAMIC'
      FRAME_-915230_CLASS       = 4
      FRAME_-915230_CLASS_ID    = -915230
      FRAME_-915230_CENTER      = -915
      TKFRAME_-915230_SPEC      = 'ANGLES'
      TKFRAME_-915230_RELATIVE  = 'A15_SPACECRAFT'
      TKFRAME_-915230_ANGLES    = ( 0.0, 0.0, 0.0 )
      TKFRAME_-915230_AXES      = (   1,   2,   3 )
      TKFRAME_-915230_UNITS     = 'DEGREES'

   \begintext


Apollo 15 NAIF ID Codes -- Definition Section
========================================================================

   This section contains name to NAIF ID mappings for the A15. Once the
   contents of this file is loaded into the KERNEL POOL, these mappings
   become available within SPICE, making it possible to use names
   instead of ID code in the high level SPICE routine calls.

   Spacecraft and its structures: 
   ------------------------------

      APOLLO 15                  -915
      A15                        -915

      A15_SPACECRAFT_BUS         -915000
      A15_SC_BUS                 -915000
      A15_SPACECRAFT             -915000

      A15_INSTRUMENT_PLATFORM    -915000
      A15_PLATFORM               -915000

   Science Instruments:
   --------------------

      A15_METRIC                 -915240
      A15_PANORAMIC              -915230

   The mappings summarized in this table are implemented by the keywords 
   below.

   \begindata

      NAIF_BODY_NAME += ( 'APOLLO 15'               )
      NAIF_BODY_CODE += ( -915                       )

      NAIF_BODY_NAME += ( 'A15'                     )
      NAIF_BODY_CODE += ( -915                       )

      NAIF_BODY_NAME += ( 'A15_SPACECRAFT_BUS'      )
      NAIF_BODY_CODE += ( -915000                   )

      NAIF_BODY_NAME += ( 'A15_SC_BUS'              )
      NAIF_BODY_CODE += ( -915000                   )

      NAIF_BODY_NAME += ( 'A15_SPACECRAFT'          )
      NAIF_BODY_CODE += ( -915000                   )

      NAIF_BODY_NAME += ( 'A15_METRIC'              )
      NAIF_BODY_CODE += ( -915240                   )

      NAIF_BODY_NAME += ( 'A15_PANORAMIC'           )
      NAIF_BODY_CODE += ( -915230                   )

\begintext

Definition of the nadir frame
-----------------------------
  All vectors are geometric: no aberration corrections are used.

  The +Z axis points from the center of the Moon to the spacecraft.

  The component of the Moon's north pole direction orthogonal to
  to Z is aligned with the +Y axis; the +Y axis points North.

  The +X axis is the cross product of the +Y axis and the +Z axis;
  the +X axis points East.


Associated Kernels
------------------

  In order to use this kernel, the following kernels must
  be loaded:

      - An SPK file giving the position of the Apollo 15
        spacecraft with respect to the Moon

      - A text PCK containing orientation data for the Moon

  For most geometry computations, the following additional kernels
  will be needed:

      - A planetary ephemeris SPK, for example de405.bsp

      - A leapseconds kernel


Restrictions
------------

1) The geometric frame specification used in this kernel is
  not usable for spacecraft latitudes of +/- 90 degrees. The frame
  implementation is subject to severe loss of precision
  for spacecraft latitudes near these limits.

2) This kernel cannot be used with SPICE Toolkit versions preceding
  version N0058.


Frame specification data
------------------------

\begindata

  FRAME_APOLLO_15_NADIR        = 1400015
  FRAME_1400015_NAME           = 'APOLLO_15_NADIR'
  FRAME_1400015_CLASS          = 5
  FRAME_1400015_CLASS_ID       = 1400015
  FRAME_1400015_CENTER         = -915
  FRAME_1400015_RELATIVE       = 'B1950'
  FRAME_1400015_DEF_STYLE      = 'PARAMETERIZED'
  FRAME_1400015_FAMILY         = 'TWO-VECTOR'
  FRAME_1400015_PRI_AXIS       = 'Z'
  FRAME_1400015_PRI_VECTOR_DEF = 'OBSERVER_TARGET_POSITION'
  FRAME_1400015_PRI_OBSERVER   = 'MOON'
  FRAME_1400015_PRI_TARGET     = -915
  FRAME_1400015_PRI_ABCORR     = 'NONE'
  FRAME_1400015_SEC_AXIS       = 'Y'
  FRAME_1400015_SEC_VECTOR_DEF = 'CONSTANT'
  FRAME_1400015_SEC_SPEC       = 'RECTANGULAR'
  FRAME_1400015_SEC_VECTOR     = ( 0, 0, 1 )
  FRAME_1400015_SEC_FRAME      = 'MOON_ME'
  FRAME_1400015_SEC_ABCORR     = 'NONE'

\begintext

