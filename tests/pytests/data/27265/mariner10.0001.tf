KPL/FK

MARINER 10 NAIF Name/ID Definitions Kernel
===============================================================================

   This text kernel contains name-to-NAIF ID mappings for MARINER 10
   (M10) mission.


Version and Date
--------------------------------------------------------

   Version 1.0 -- June 4, 2008 -- Boris Semenov, NAIF


References
--------------------------------------------------------

   1. ``SP-424 The Voyage of Mariner 10'', 
      http://history.nasa.gov/SP-424/sp424.htm


M10 NAIF Name-ID Mappings
--------------------------------------------------------

   This section contains name to NAIF ID mappings for the M10 mission.
   Once the contents of this file is loaded into the KERNEL POOL, these
   mappings become available within SPICE, making it possible to use
   names instead of ID code in the high level SPICE routine calls.

   The tables below summarize the mappings; the actual definitions are
   provided after the last summary table.

   Spacecraft and Spacecraft Bus
   -----------------------------

      MARINER-10		-76
      MARINER 10		-76
      MARINER_10		-76
      MARINER10			-76
      M10			-76

      M10_SC_BUS		-76000
      M10_SPACECRAFT_BUS	-76000
      M10_SPACECRAFT		-76000

      M10_INSTRUMENT_PLATFORM	-76000
      M10_PLATFORM		-76000

   Instruments and Sensors Mounted on Spacecraft Bus
   -------------------------------------------------

      M10_IR              -76010

      M10_PLASMA_SEA      -76021
      M10_PLASMA_SES      -76022

      M10_CPT_MAIN        -76041
      M10_CPT_LOW_ENERGY  -76042

      M10_EUV_OCCULTATION -76051

      M10_SUN_SENSOR      -76060

      M10_STAR_TRACKER    -76070

   Scan Platform
   -------------

      M10_SCAN_PLATFORM   -76100

   Instruments Mounted on Scan Platform
   ------------------------------------

      M10_VIDICON_A       -76110
      A       		  -76110
      M10_VIDICON_B       -76120
      B       		  -76120

      M10_EUV_AIRGLOW     -76152

   Magnetometer Boom and Sensors 
   ----------------------------- 

      M10_MAG_BOOM        -76200
      M10_MAG_INBOARD     -76211 
      M10_MAG_OUTBOARD    -76212

   Solar Arrays
   ------------

      M10_SA+X            -76310 
      M10_SA-X            -76320 

   High Gain Antenna
   -----------------

      M10_HGA             -76400

   Low Gain Antenna
   ----------------

      M10_LGA             -76500


   Spacecraft Bus Frame
   -------------------------------------------------------------------------------
   
      \begindata
   
         FRAME_M10_SPACECRAFT     = -76000
         FRAME_-76000_NAME        = 'M10_SPACECRAFT'
         FRAME_-76000_CLASS       = 3
         FRAME_-76000_CLASS_ID    = -76000
         FRAME_-76000_CENTER      = -76
         CK_-76000_SCLK           = -76
         CK_-76000_SPK            = -76
   
      \begintext
   
   Metric Frames
   -------------------------------------------------------------------------------
   
      \begindata
   
         FRAME_VIDICON_A          = -76110
         FRAME_-76110_NAME        = 'VIDICON_A'
         FRAME_-76110_CLASS       = 3
         FRAME_-76110_CLASS_ID    = -76110
         FRAME_-76110_CENTER      = -76
         CK_-76110_SCLK           = -76
         CK_-76110_SPK            = -76
   
         FRAME_VIDICON_B          = -76120
         FRAME_-76120_NAME        = 'VIDICON_B'
         FRAME_-76120_CLASS       = 3
         FRAME_-76120_CLASS_ID    = -76120
         FRAME_-76120_CENTER      = -76
         CK_-76120_SCLK           = -76
         CK_-76120_SPK            = -76
   
      \begintext
  
   The keywords below implement M10 name-ID mappings summarized above.

   \begindata

      NAIF_BODY_NAME += ( 'MARINER-10' )
      NAIF_BODY_CODE += ( -76 )

      NAIF_BODY_NAME += ( 'MARINER 10' )
      NAIF_BODY_CODE += ( -76 )

      NAIF_BODY_NAME += ( 'MARINER10' )
      NAIF_BODY_CODE += ( -76 )

      NAIF_BODY_NAME += ( 'M-10' )
      NAIF_BODY_CODE += ( -76 )

      NAIF_BODY_NAME += ( 'M10' )
      NAIF_BODY_CODE += ( -76 )

      NAIF_BODY_NAME += ( 'M10_SC_BUS' )
      NAIF_BODY_CODE += ( -76000 )

      NAIF_BODY_NAME += ( 'M10_SPACECRAFT_BUS' )
      NAIF_BODY_CODE += ( -76000 )

      NAIF_BODY_NAME += ( 'M10_SPACECRAFT' )
      NAIF_BODY_CODE += ( -76000 )

      NAIF_BODY_NAME += ( 'M10_INSTRUMENT_PLATFORM' )
      NAIF_BODY_CODE += ( -76000 )

      NAIF_BODY_NAME += ( 'M10_PLATFORM' )
      NAIF_BODY_CODE += ( -76000 )

      NAIF_BODY_NAME += ( 'M10_IR' )
      NAIF_BODY_CODE += ( -76010 )

      NAIF_BODY_NAME += ( 'M10_PLASMA_SEA' )
      NAIF_BODY_CODE += ( -76021 )

      NAIF_BODY_NAME += ( 'M10_PLASMA_SES' )
      NAIF_BODY_CODE += ( -76022 )

      NAIF_BODY_NAME += ( 'M10_CPT_MAIN' )
      NAIF_BODY_CODE += ( -76041 )

      NAIF_BODY_NAME += ( 'M10_CPT_LOW_ENERGY' )
      NAIF_BODY_CODE += ( -76042 )

      NAIF_BODY_NAME += ( 'M10_EUV_OCCULTATION' )
      NAIF_BODY_CODE += ( -76051 )

      NAIF_BODY_NAME += ( 'M10_SUN_SENSOR' )
      NAIF_BODY_CODE += ( -76060 )

      NAIF_BODY_NAME += ( 'M10_STAR_TRACKER' )
      NAIF_BODY_CODE += ( -76070 )

      NAIF_BODY_NAME += ( 'M10_SCAN_PLATFORM' )
      NAIF_BODY_CODE += ( -76100 )

      NAIF_BODY_NAME += ( 'M10_VIDICON_A' )
      NAIF_BODY_CODE += ( -76110 )

      NAIF_BODY_NAME += ( 'A' )
      NAIF_BODY_CODE += ( -76110 )

      NAIF_BODY_NAME += ( 'M10_VIDICON_B' )
      NAIF_BODY_CODE += ( -76120 )

      NAIF_BODY_NAME += ( 'B' )
      NAIF_BODY_CODE += ( -76120 )

      NAIF_BODY_NAME += ( 'M10_EUV_AIRGLOW' )
      NAIF_BODY_CODE += ( -76152 )

      NAIF_BODY_NAME += ( 'M10_MAG_BOOM' )
      NAIF_BODY_CODE += ( -76200 )

      NAIF_BODY_NAME += ( 'M10_MAG_INBOARD' )
      NAIF_BODY_CODE += ( -76211 )

      NAIF_BODY_NAME += ( 'M10_MAG_OUTBOARD' )
      NAIF_BODY_CODE += ( -76212 )

      NAIF_BODY_NAME += ( 'M10_SA+X' )
      NAIF_BODY_CODE += ( -76310 )

      NAIF_BODY_NAME += ( 'M10_SA-X' )
      NAIF_BODY_CODE += ( -76320 )

      NAIF_BODY_NAME += ( 'M10_HGA' )
      NAIF_BODY_CODE += ( -76400 )

      NAIF_BODY_NAME += ( 'M10_LGA' )
      NAIF_BODY_CODE += ( -76500 )

   \begintext
