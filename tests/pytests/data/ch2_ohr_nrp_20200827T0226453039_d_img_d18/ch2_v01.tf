KPL/FK

Chandrayaan-2 Orbiter Frames Kernel
========================================================================

   This frame kernel contains complete set of frame definitions for the
   Chandrayaan-2 Orbiter Spacecraft and Chandrayaan Orbiter science instruments.
   This kernel also contains NAIF ID/name mapping for the Chandrayaan-2 instruments.


Version and Date
========================================================================
 
   Version 1.0 -- June 19, 2023 -- 
   SAC Optical Payload DP Team, SAC, Ahmedabad
   Flight Dynamics Group, URSC-ISRO, Bangalore

   Information provided by the Payload team - SEDA/SAC.
   Information integration and kernel verification by FDG team - URSC


References
========================================================================

   1. ``Frames Required Reading''

   2. ``Kernel Pool Required Reading''

   3. ``C-Kernel Required Reading''


Implementation Notes
========================================================================

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must `load' the kernel, normally during
   program initialization. The SPICELIB routine FURNSH and CSPICE
   function furnsh_c load a kernel file into the kernel pool as shown
   below.

      CALL FURNSH ( 'frame_kernel_name' )
      furnsh_c    ( "frame_kernel_name" );

   This file was created and may be updated with a text editor or word
   processor.



Chandrayaan-2 Mission NAIF ID Codes -- Summary Section
========================================================================
 
   The following names and NAIF ID codes are assigned to the Chandrayaan-2
   spacecraft, its structures and science instruments (the keywords
   implementing these definitions are located in the section "Chandrayaan-2
   Mission NAIF ID Codes -- Definition Section" at the end of this
   file):

   Chandrayaan-2 Spacecraft and Spacecraft Structures names/IDs:

            CHANDRAYAAN-2         -152
            CH2                   -152
            CH2_ORBITER         -152001

            CH2_TMC_NADIR       -152210
            CH2_TMC_FORE        -152211
            CH2_TMC_AFT         -152212
            CH2_OHRC            -152270
            CH2_IIR_I           -152240



Chandrayaan-2 Frames
=====================================================================================================

   The following Chandrayaan-2 frames are defined in this kernel file:

           Name                  Relative to           Type       NAIF ID
      ======================  ===================  ============   =======

            CH2_SPACECRAFT        J2000               CK          -152000
            CH2_TMC_NADIR         CH2                FIXED        -152210
            CH2_TMC_FORE          CH2                FIXED        -152211
            CH2_TMC_AFT           CH2                FIXED        -152212
            CH2_OHRC              CH2                FIXED        -152270
            CH2_IIR_I             CH2                FIXED        -152240



Frame Tree
=====================================================================================================

   The diagram below shows the frame hierarchy for the Chandrayaan-2 Orbiter spacecraft
   and its structure frame.


                               "J2000" INERTIAL
           +-----------------------------------------------------+
           |                          |                          |
           |<-pck                     |<-ck                      |<-pck
           |                          |                          |
           V                          |                          V
      "IAU_MOON"                      |                    "MOON_PA"
       MOON BFXD                      |                    MOON HIGH-PREC BFXD
      -------------                   |                    -------------------
                                      |
                                      |
                                      |
                                      |
                                      |
      "CH2_SN1/2"                     |
      -----------                     |
           ^                          |
           |                          |
           |<-fixed                   |                         
           |                          V
           |                 "CH2_SPACECRAFT"                
           +---------------------------------------------------------------------------+
           |                |                       |            |                     |
           |<-ck            |<-ck                   |<-fixed     |<-fixed              |<-fixed
           |                |                       |            |                     |
           V                V                       V            V                     V
       "CH2_SA_GIMBAL" "CH2_HGA_GIMBAL"        "CH2_TMC"    "CH2_IIRS"              "CH2_OHRC"
       --------------- ---------------         ------------ ----------              ----------
           |                |                       |            |                     |
           |<-fixed         |<-fixed                |<-fixed     |<-fixed              |<-fixed
           |                |                       |            |                     |
           V                V                       V            V                     V
       "CH2_SA"         "CH2_HGA"       "AFT" "NADIR" "FORE"   "Infrared Spectrometer" "PAN"
       --------         ---------       ----- ------- ------   ---------------------- -----



CH2 Spacecraft Frame
===================================================================================================

            CH2_ORBITER    J2000                CK             -152001


CH2 Spacecraft Frame
--------------------------------------

   The CH2 spacecraft frame is defined as follows:

      -  +Z towards solar array side -- positive or negative normal to 
         orbit plane

      -  +X is along instrument boresights - towards Moon

      -  +Y  -- along velocity or along anti-velocity

      -  the origin of this frame is the launch vehicle interface point.

   Since the orientation of the CH2_SPACECRAFT frame is computed
   on-board, sent down in telemetry, and stored in the s/c CK files, it
   is defined as a CK-based frame.
 
   \begindata

      FRAME_CH2_ORBITER             = -152001
      FRAME_-152001_NAME               = 'CH2_ORBITER'
      FRAME_-152001_CLASS              =  3
      FRAME_-152001_CLASS_ID           = -152001
      FRAME_-152001_CENTER             = -152
      CK_-152001_SCLK                  = -152
      CK_-152001_SPK                   = -152
      OBJECT_-152_FRAME                = 'CH2_ORBITER'
   \begintext



Star Sensor Frames
========================================================================

            CH2_SN1           CH2_SPACECRAFT       FIXED          -152051
            CH2_SN2           CH2_SPACECRAFT       FIXED          -152052


   The orienation of the star sensor frame is given by the following
   sensor-to-body quaternion :

      [0.3099752106 -0.0123421249 0.6355433650 0.7069990854]   


   \begindata

      FRAME_CH2_SN1                   =  -152051
      FRAME_-152051_NAME               = 'CH2_SN1'
      FRAME_-152051_CLASS              =  4
      FRAME_-152051_CLASS_ID           =  -152051
      FRAME_-152051_CENTER             =  -152
      TKFRAME_-152051_RELATIVE         = 'CH2_ORBITER'
      TKFRAME_-152051_SPEC             = 'QUATERNION'
      TKFRAME_-152051_Q                = (

                  0.7069990854 -0.3099752106 0.0123421249 -0.6355433650

                                        )

   \begintext



OHRC Frames
========================================================================

   \begintext

   The OHRC camera detector frames, CH2_OHRC
   are defined as follows:

      -  +X axis points along the detector boresight;

      -  +Z axis is parallel to the detecor lines.

      -  +Y axis completes the right handed frame

      -  the origin of the frame is located at the camera focal point.

   \begindata

      FRAME_CH2_OHRC               =  -152270
      FRAME_-152270_NAME               = 'CH2_OHRC'
      FRAME_-152270_CLASS              =  4
      FRAME_-152270_CLASS_ID           =  -152270
      FRAME_-152270_CENTER             =  -152
      TKFRAME_-152270_RELATIVE         = 'CH2_ORBITER'
      TKFRAME_-152270_SPEC             = 'ANGLES'
      TKFRAME_-152270_UNITS            = 'DEGREES'
      TKFRAME_-152270_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-152270_AXES             = ( 1,   2,   3   )


   \begintext



TMC Frames
========================================================================

   \begintext

   The TMC camera detector frames, CH2_TMC_NADIR, CH2_TMC_FORE, CH2_TMC_AFT
   are defined as follows:

      -  +X axis points along the detector boresight;

      -  +Z axis is parallel to the detecor lines.

      -  +Y axis completes the right handed frame

      -  the origin of the frame is located at the camera focal point.

   The NADIR detector frame is co-aligned with the camera frame. The FORE
   detector frame is rotated 25 degrees about +Z relative to the camera
   frame. The AFT detector frame is rotated -25 degrees about +Z relative 
   to the camera frame.

   \begindata

      FRAME_CH2_TMC_AFT               =  -152212
      FRAME_-152212_NAME               = 'CH2_TMC_AFT'
      FRAME_-152212_CLASS              =  4
      FRAME_-152212_CLASS_ID           =  -152212
      FRAME_-152212_CENTER             =  -152
      TKFRAME_-152212_RELATIVE         = 'CH2_ORBITER'
      TKFRAME_-152212_SPEC             = 'ANGLES'
      TKFRAME_-152212_UNITS            = 'DEGREES'
      TKFRAME_-152212_ANGLES           = ( 0.0, 0.0, 25.0 )
      TKFRAME_-152212_AXES             = ( 1,   2,   3   )

      FRAME_CH2_TMC_NADIR             =  -152210
      FRAME_-152210_NAME               = 'CH2_TMC_NADIR'
      FRAME_-152210_CLASS              =  4
      FRAME_-152210_CLASS_ID           =  -152210
      FRAME_-152210_CENTER             =  -152
      TKFRAME_-152210_RELATIVE         = 'CH2_ORBITER'
      TKFRAME_-152210_SPEC             = 'ANGLES'
      TKFRAME_-152210_UNITS            = 'DEGREES'
      TKFRAME_-152210_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-152210_AXES             = ( 1,   2,   3   )

      FRAME_CH2_TMC_FORE              =  -152211
      FRAME_-152211_NAME               = 'CH2_TMC_FORE'
      FRAME_-152211_CLASS              =  4
      FRAME_-152211_CLASS_ID           =  -152211
      FRAME_-152211_CENTER             =  -152
      TKFRAME_-152211_RELATIVE         = 'CH2_ORBITER'
      TKFRAME_-152211_SPEC             = 'ANGLES'
      TKFRAME_-152211_UNITS            = 'DEGREES'
      TKFRAME_-152211_ANGLES           = ( 0.0, 0.0, -25.0 )
      TKFRAME_-152211_AXES             = ( 1,   2,   3   )


   \begintext


IIRS Frames
========================================================================

   \begintext

   The IIRS camera detector frames, CH2_IIR_I
   are defined as follows:

      -  +X axis points along the detector boresight;

      -  +Z axis is parallel to the detecor lines.

      -  +Y axis completes the right handed frame

      -  the origin of the frame is located at the camera focal point.

   \begindata

      FRAME_CH2_IIR_I                  =  -152240
      FRAME_-152240_NAME               = 'CH2_IIR_I'
      FRAME_-152240_CLASS              =  4
      FRAME_-152240_CLASS_ID           =  -152240
      FRAME_-152240_CENTER             =  -152
      TKFRAME_-152240_RELATIVE         = 'CH2_ORBITER'
      TKFRAME_-152240_SPEC             = 'ANGLES'
      TKFRAME_-152240_UNITS            = 'DEGREES'
      TKFRAME_-152240_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-152240_AXES             = ( 1,   2,   3   )




   \begintext

CH2 Mission NAIF ID Codes -- Definition Section
========================================================================

   This section contains name to NAIF ID mappings for the CH2 mission.

         \begindata

            NAIF_BODY_NAME += ( 'CHANDRAYAAN-2' )         
            NAIF_BODY_CODE += ( -152 )

            NAIF_BODY_NAME += ( 'CH2' )                   
            NAIF_BODY_CODE += ( -152 )

            NAIF_BODY_NAME += ( 'CH2_ORBITER' )        
            NAIF_BODY_CODE += ( -152001 )

            NAIF_BODY_NAME += ( 'CH2_OHRC' )           
            NAIF_BODY_CODE += ( -152270 )

            NAIF_BODY_NAME += ( 'CH2_TMC_AFT' )           
            NAIF_BODY_CODE += ( -152212 )

            NAIF_BODY_NAME += ( 'CH2_TMC_NADIR' )         
            NAIF_BODY_CODE += ( -152210 )

            NAIF_BODY_NAME += ( 'CH2_TMC_FORE' )          
            NAIF_BODY_CODE += ( -152211 )

            NAIF_BODY_NAME += ( 'CH2_IIR_I' )           
            NAIF_BODY_CODE += ( -152240 )


         \begintext

End of CH2 FK File.
