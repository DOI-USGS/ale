KPL/FK

MSL Frames Kernel
========================================================================

   This frame kernel contains complete set of frame definitions for the
   MSL including definitions for the MSL cruise, descent, and rover
   frames, local level, topocentric and surface-fixed frames, appendage 
   frames, and science instrument frames.


Version and Date
========================================================================

   Version 0.8 -- May 30, 2013 -- Boris Semenov, NAIF

      Updated alignments of MSL_NAVCAM_LEFT_B and MSL_NAVCAM_RIGHT_B
      based on the updated CAHVOR models for the -5C temperature point
      (MSL_CAL_005_SN_0215-NAVL-FLIGHT-RCE-B.cahvor and 
      MSL_CAL_005_SN_0218-NAVR-FLIGHT-RCE-B.cahvor).

   Version 0.7 -- April 12, 2013 -- Boris Semenov, NAIF

      Added name/ID mappings for DIMU_A.

   Version 0.6 -- November 21, 2012 -- Boris Semenov, NAIF

      Added name/ID mappings for DAN modules.

   Version 0.5 -- August 8, 2012 -- Boris Semenov, NAIF

      Redefined HGA, RSM, and RA frame chains. Added name/ID mappings
      for calibration targets, fiducials, etc. Incorporated actual
      frame alignments based on camera models and flight parameters.
      Incorporated the topocentric frame definition based on the actual
      landing site location.

   Version 0.4 -- December 13, 2011 -- Boris Semenov, NAIF

      Changed antenna frame ID codes as follows:

         MSL_PLGA        -76810  -->  -76060
         MSL_TLGA        -76811  -->  -76061
         MSL_PUHF        -76812  -->  -76062
         MSL_MGA         -76813  -->  -76063

         MSL_DLGA        -76820  -->  -76064
         MSL_DUHF        -76821  -->  -76065

         MSL_RLGA        -76830  -->  -76110
         MSL_RUHF        -76840  -->  -76111
         MSL_HGA_BASE    -76850  -->  -76120
         MSL_HGA_GIMBAL  -76851  -->  -76121
         MSL_HGA         -76852  -->  -76122
         MSL_HGA_EB      -76853  -->  -76123

     Added name-ID mapping for all instruments, structures and sites.

   Version 0.3 -- May 4, 2011 -- Boris Semenov, NAIF

      Preliminary version. Changed MSL_TLGA tilt angle to 17.5 
      degrees.

   Version 0.2 -- May 3, 2011 -- Boris Semenov, NAIF

      Preliminary version. Added CACS and antenna frames.

   Version 0.1 -- March 3, 2011 -- Boris Semenov, NAIF

      Preliminary version. Added rover and cruise/descent frames.

   Version 0.0 -- December 4, 2007 -- Boris Semenov, NAIF

      Very preliminary version. Contains definitions only for the 
      LOCAL_LEVEL and SURFACE_FIXED frames.
 

References
========================================================================

   1. ``Frames Required Reading''

   2. ``Kernel Pool Required Reading''

   3. ``C-Kernel Required Reading''

   4. MSL 3PCS document, latest version

   5. RSVP's MSL_kinematics_tree.xml, 2012-07-06

   6. FSW parameter set, July 2012

   7. 'MSL_Surface_Cruise DataRate_Analysis_Rev-A_5-19-11.pdf', 
      Section E.7, May 10, 2011

   8. MSL camera CAHVORE Models and reference positions/quaternions,
      July 2012


Contact Information
========================================================================

   Boris V. Semenov, NAIF/JPL, (818)-354-8136, Boris.Semenov@jpl.nasa.gov


Implementation Notes
========================================================================

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must `load' the kernel using SPICE routine
   FURNSH, normally during program initialization.

   This file was created and may be updated with a text editor or word
   processor.


MSL NAIF ID Codes
========================================================================

   The following names and NAIF ID codes are assigned to the MSL rover,
   its structures and science instruments (the keywords implementing
   these definitions are located in the section "MSL Mission NAIF ID
   Codes -- Definition Section" at the end of this file):
 
   Landing site and sites:
   -----------------------

      MSL_LANDING_SITE         -76900

      MSL_SITE_1...399         -76501...-76899

   Cruise and descent stages and the rover:
   ----------------------------------------

      MSL                      -76

      MSL_ROVER                -76000

      MSL_SPACECRAFT           -76010
      MSL_CRUISE_STAGE         -76020
      MSL_DESCENT_STAGE        -76030
      MSL_ROVER_MECH           -76040
      MSL_CACS                 -76050

      MSL_DIMU_A               -76031


   Instruments and structures on cruise and descent modules:
   ---------------------------------------------------------

      MSL_PLGA                 -76060
      MSL_TLGA                 -76061
      MSL_PUHF                 -76062
      MSL_MGA                  -76063

      MSL_DLGA                 -76064
      MSL_DUHF                 -76065

      MSL_MEDLI_MISP_T1        -76071
      MSL_MEDLI_MISP_T2        -76072
      MSL_MEDLI_MISP_T3        -76073
      MSL_MEDLI_MISP_T4        -76074
      MSL_MEDLI_MISP_T5        -76075
      MSL_MEDLI_MISP_T6        -76076
      MSL_MEDLI_MISP_T7        -76077

      MSL_MEDLI_MEADS_P1       -76081
      MSL_MEDLI_MEADS_P2       -76082
      MSL_MEDLI_MEADS_P3       -76083
      MSL_MEDLI_MEADS_P4       -76084
      MSL_MEDLI_MEADS_P5       -76085
      MSL_MEDLI_MEADS_P6       -76086
      MSL_MEDLI_MEADS_P7       -76087

   Instruments and devices on the rover:
   --------------------------------------

      MSL_RLGA                 -76110
      MSL_RUHF                 -76111

      MSL_HGA_ZERO_AZ          -76121
      MSL_HGA_AZ               -76122
      MSL_HGA_ZERO_EL          -76123
      MSL_HGA_EL               -76124
      MSL_HGA                  -76125
      MSL_HGA_EB               -76126

      MSL_HAZCAM_FRONT_LEFT_A  -76131
      MSL_HAZCAM_FRONT_RIGHT_A -76132
      MSL_HAZCAM_FRONT_LEFT_B  -76133
      MSL_HAZCAM_FRONT_RIGHT_B -76134
      MSL_HAZCAM_BACK_LEFT_A   -76141
      MSL_HAZCAM_BACK_RIGHT_A  -76142
      MSL_HAZCAM_BACK_LEFT_B   -76143
      MSL_HAZCAM_BACK_RIGHT_B  -76144

      MSL_RAD                  -76150

      MSL_MARDI                -76160

      MSL_REMS_UVS             -76170

      MSL_DAN                  -76180
      MSL_DAN_PNG              -76191
      MSL_DAN_DE               -76192
      MSL_DAN_DE_CTN           -76193
      MSL_DAN_DE_CETN          -76194
      MSL_REMS_PS              -76181
      MSL_SAM                  -76182
      MSL_SAM_1_INLET          -76184
      MSL_SAM_2_INLET          -76185
      MSL_CHEMIN               -76183
      MSL_CHEMIN_INLET         -76186

   Instruments and devices on RSM:
   --------------------------------

      MSL_RSM_ZERO_AZ          -76201
      MSL_RSM_AZ               -76202          
      MSL_RSM_ZERO_EL          -76203
      MSL_RSM_EL               -76204
      MSL_RSM_HEAD             -76205

      MSL_MASTCAM_LEFT         -76210
      MSL_MASTCAM_LEFT_F1      -76211
      MSL_MASTCAM_LEFT_F2      -76212
      MSL_MASTCAM_LEFT_F3      -76213
      MSL_MASTCAM_LEFT_F4      -76214
      MSL_MASTCAM_LEFT_F5      -76215
      MSL_MASTCAM_LEFT_F6      -76216
      MSL_MASTCAM_LEFT_F7      -76217
      MSL_MASTCAM_LEFT_F8      -76218
      MSL_MASTCAM_RIGHT        -76220
      MSL_MASTCAM_RIGHT_F1     -76221
      MSL_MASTCAM_RIGHT_F2     -76222
      MSL_MASTCAM_RIGHT_F3     -76223
      MSL_MASTCAM_RIGHT_F4     -76224
      MSL_MASTCAM_RIGHT_F5     -76225
      MSL_MASTCAM_RIGHT_F6     -76226
      MSL_MASTCAM_RIGHT_F7     -76227
      MSL_MASTCAM_RIGHT_F8     -76228

      MSL_NAVCAM_LEFT_A        -76231
      MSL_NAVCAM_RIGHT_A       -76232
      MSL_NAVCAM_LEFT_B        -76233
      MSL_NAVCAM_RIGHT_B       -76234

      MSL_CHEMCAM              -76240
      MSL_CHEMCAM_LIBS_LASER   -76241
      MSL_CHEMCAM_LIBS_CAM     -76242
      MSL_CHEMCAM_RMI          -76243

      MSL_REMS_BOOM1           -76250
      MSL_REMS_BOOM1_WS1       -76251
      MSL_REMS_BOOM1_WS2       -76252
      MSL_REMS_BOOM1_WS3       -76253
      MSL_REMS_BOOM1_ATS       -76254
      MSL_REMS_BOOM1_GTS       -76255
      MSL_REMS_BOOM1_TIP       -76256

      MSL_REMS_BOOM2           -76260
      MSL_REMS_BOOM2_WS1       -76261
      MSL_REMS_BOOM2_WS2       -76262
      MSL_REMS_BOOM2_WS3       -76263
      MSL_REMS_BOOM2_ATS       -76264
      MSL_REMS_BOOM2_HS        -76265
      MSL_REMS_BOOM2_TIP       -76266

   Instruments and devices on RA:
   ------------------------------

      MSL_RA_BASE              -76300
      MSL_RA_SHOULDER_AZ       -76301
      MSL_RA_SHOULDER_EL       -76302
      MSL_RA_ELBOW             -76303
      MSL_RA_WRIST             -76304
      MSL_RA_TURRET            -76305
      MSL_RA_TURRET_HEAD       -76306

      MSL_MAHLI_REF            -76310
      MSL_MAHLI                -76311

      MSL_APXS_REF             -76320
      MSL_APXS                 -76321

      MSL_PADS_REF             -76330
      MSL_PADS                 -76331

      MSL_DRT_REF              -76340
      MSL_DRT                  -76341

      MSL_CHIMRA_REF           -76350
      MSL_CHIMRA               -76351

   Fiducials, Calibration targets, etc.
   ------------------------------------

      MSL_FD_RESTRAINT_PORT    -76401
      MSL_FD_RESTRAINT_SBRD    -76402
      MSL_FD_DECK_SIDE_PORT    -76403
      MSL_FD_DECK_FRONT_PORT   -76404
      MSL_FD_DECK_CENTER_PORT  -76405
      MSL_FD_DECK_CENTER_SBRD  -76406
      MSL_FD_DECK_FRONT_SBRD   -76407

      MSL_FD_OCM_PORT          -76408
      MSL_FD_OCM_CENTER        -76409
      MSL_FD_OCM_STARBOARD     -76410

      MSL_SCI_OBS_TRAY         -76411
      MSL_ENG_OBS_TRAY         -76412

      MSL_BIT_BOX_1_TARGET     -76413
      MSL_BIT_BOX_2_TARGET     -76414

      MSL_OCM_LOCATION_1       -76415
      MSL_OCM_LOCATION_2       -76416
      MSL_OCM_LOCATION_3       -76417
      MSL_OCM_LOCATION_4       -76418
      MSL_OCM_LOCATION_5       -76419
      MSL_OCM_LOCATION_6       -76420

      MSL_APXS_CALTARGET       -76421
      MSL_MAHLI_CALTARGET      -76422

      MSL_CCAM_CAL1            -76423
      MSL_CCAM_CAL2            -76424
      MSL_CCAM_CAL3            -76425
      MSL_CCAM_CAL4            -76426
      MSL_CCAM_CAL5            -76427
      MSL_CCAM_CAL6            -76428
      MSL_CCAM_CAL7            -76429
      MSL_CCAM_CAL8            -76430
      MSL_CCAM_CAL9            -76431
      MSL_CCAM_CAL10           -76432

      MSL_MCAM_CAL             -76433


MSL Frames
========================================================================

   The following MSL frames are defined in this kernel file:

           Name                      Relative to         Type    NAIF ID
      ======================     ===================     =====   =======

   Surface frames:
   ---------------

      MSL_TOPO                   IAU_MARS                FIXED   -76900
      MSL_LOCAL_LEVEL            MSL_TOPO                FIXED   -76910
      MSL_SURFACE_FIXED          MSL_LOCAL_LEVEL         FIXED   -76920

   Rover frames:
   -------------

      MSL_ROVER                  J2000, MSL_LOCAL_LEVEL  CK      -76000
      MSL_ROVER_MECH             MSL_ROVER               FIXED   -76040

   Cruise and Descent frames:
   --------------------------

      MSL_SPACECRAFT             MSL_ROVER               FIXED   -76010
      MSL_CRUISE_STAGE           MSL_ROVER               FIXED   -76020
      MSL_DESCENT_STAGE          MSL_ROVER               FIXED   -76030
      MSL_CACS                   J2000, MSL_ROVER        CK      -76050

   Cruise and Descent Antenna frames:
   ----------------------------------

      MSL_PLGA                   MSL_CRUISE_STAGE        FIXED   -76060
      MSL_TLGA                   MSL_CRUISE_STAGE        FIXED   -76061
      MSL_PUHF                   MSL_CRUISE_STAGE        FIXED   -76062
      MSL_MGA                    MSL_CRUISE_STAGE        FIXED   -76063

      MSL_DLGA                   MSL_DESCENT_STAGE       FIXED   -76064
      MSL_DUHF                   MSL_DESCENT_STAGE       FIXED   -76065

   Rover instrument and structures frames:
   ---------------------------------------

      MSL_RLGA                   MSL_ROVER               FIXED   -76110
      MSL_RUHF                   MSL_ROVER               FIXED   -76111

      MSL_HGA_ZERO_AZ            MSL_ROVER               FIXED   -76121
      MSL_HGA_AZ                 MSL_HGA_ZERO_AZ         CK      -76122
      MSL_HGA_ZERO_EL            MSL_HGA_AZ              FIXED   -76123
      MSL_HGA_EL                 MSL_HGA_ZERO_EL         CK      -76124
      MSL_HGA                    MSL_HGA_EL              FIXED   -76125
      MSL_HGA_EB                 MSL_HGA                 FIXED   -76126

      MSL_HAZCAM_FRONT_LEFT_A    MSL_ROVER               FIXED   -76131
      MSL_HAZCAM_FRONT_RIGHT_A   MSL_ROVER               FIXED   -76132
      MSL_HAZCAM_FRONT_LEFT_B    MSL_ROVER               FIXED   -76133
      MSL_HAZCAM_FRONT_RIGHT_B   MSL_ROVER               FIXED   -76134
      MSL_HAZCAM_BACK_LEFT_A     MSL_ROVER               FIXED   -76141
      MSL_HAZCAM_BACK_RIGHT_A    MSL_ROVER               FIXED   -76142
      MSL_HAZCAM_BACK_LEFT_B     MSL_ROVER               FIXED   -76143
      MSL_HAZCAM_BACK_RIGHT_B    MSL_ROVER               FIXED   -76144

      MSL_RAD                    MSL_ROVER               FIXED   -76150

      MSL_MARDI                  MSL_ROVER               FIXED   -76160

      MSL_REMS_UVS               MSL_ROVER               FIXED   -76170

      MSL_DAN                    MSL_ROVER               FIXED   -76180
      MSL_SAM                    MSL_ROVER               FIXED   -76182
      MSL_CHEMIN                 MSL_ROVER               FIXED   -76183

   RSM instrument and structures frames:
   -------------------------------------

      MSL_RSM_ZERO_AZ            MSL_ROVER               FIXED   -76201
      MSL_RSM_AZ                 MSL_RSM_ZERO_AZ         CK      -76202
      MSL_RSM_ZERO_EL            MSL_RSM_AZ              FIXED   -76203
      MSL_RSM_EL                 MSL_RSM_ZERO_EL         CK      -76204
      MSL_RSM_HEAD               MSL_RSM_EL              FIXED   -76205

      MSL_MASTCAM_LEFT           MSL_RSM_HEAD            FIXED   -76210
      MSL_MASTCAM_RIGHT          MSL_RSM_HEAD            FIXED   -76220

      MSL_NAVCAM_LEFT_A          MSL_RSM_HEAD            FIXED   -76231
      MSL_NAVCAM_RIGHT_A         MSL_RSM_HEAD            FIXED   -76232
      MSL_NAVCAM_LEFT_B          MSL_RSM_HEAD            FIXED   -76233
      MSL_NAVCAM_RIGHT_B         MSL_RSM_HEAD            FIXED   -76234

      MSL_CHEMCAM_LIBS_LASER     MSL_RSM_HEAD            FIXED   -76241
      MSL_CHEMCAM_LIBS_CAM       MSL_RSM_HEAD            FIXED   -76242
      MSL_CHEMCAM_RMI            MSL_RSM_HEAD            FIXED   -76243

      MSL_REMS_BOOM1             MSL_ROVER               FIXED   -76250
      MSL_REMS_BOOM1_WS1         MSL_REMS_BOOM1          FIXED   -76251
      MSL_REMS_BOOM1_WS2         MSL_REMS_BOOM1          FIXED   -76252
      MSL_REMS_BOOM1_WS3         MSL_REMS_BOOM1          FIXED   -76253
      MSL_REMS_BOOM1_ATS         MSL_REMS_BOOM1          FIXED   -76254
      MSL_REMS_BOOM1_GTS         MSL_REMS_BOOM1          FIXED   -76255

      MSL_REMS_BOOM2             MSL_ROVER               FIXED   -76260
      MSL_REMS_BOOM2_WS1         MSL_REMS_BOOM2          FIXED   -76261
      MSL_REMS_BOOM2_WS2         MSL_REMS_BOOM2          FIXED   -76262
      MSL_REMS_BOOM2_WS3         MSL_REMS_BOOM2          FIXED   -76263
      MSL_REMS_BOOM2_ATS         MSL_REMS_BOOM2          FIXED   -76264
      MSL_REMS_BOOM2_HS          MSL_REMS_BOOM2          FIXED   -76265

   RA instrument and structures frames:
   -------------------------------------

      MSL_RA_BASE                MSL_ROVER               FIXED   -76300
      MSL_RA_SHOULDER_AZ         MSL_RA_BASE             CK      -76301
      MSL_RA_SHOULDER_EL         MSL_RA_SHOULDER_AZ      CK      -76302
      MSL_RA_ELBOW               MSL_RA_SHOULDER_EL      CK      -76303
      MSL_RA_WRIST               MSL_RA_ELBOW            CK      -76304
      MSL_RA_TURRET              MSL_RA_WRIST            CK      -76305

      MSL_MAHLI_REF              MSL_RA_TURRET           CK      -76310
      MSL_MAHLI                  MSL_MAHLI_REF           FIXED   -76311

      MSL_APXS_REF               MSL_RA_TURRET           CK      -76320
      MSL_APXS                   MSL_APXS_REF            FIXED   -76321

      MSL_PADS_REF               MSL_RA_TURRET           CK      -76330
      MSL_PADS                   MSL_PADS_REF            FIXED   -76331

      MSL_DRT_REF                MSL_RA_TURRET           CK      -76340
      MSL_DRT                    MSL_DRT_REF             FIXED   -76341

      MSL_CHIMRA_REF             MSL_RA_TURRET           CK      -76350
      MSL_CHIMRA                 MSL_CHIMRA_REF          FIXED   -76351


MSL Frame Tree
========================================================================

   The diagram below shows the MSL frame hierarchy:


                                   "J2000" 
                      +---------------------------------+
                      |               |<-pck            |<-pck
                      |               v                 v
                      |          "IAU_MARS"       "IAU_EARTH"
                      |          ----------       -----------
                      |               |<-fixed      
                      |               v             
                      |          "MSL_TOPO"      "MSL_SURFACE_FIXED"
                      |          ----------      ---------------------
                      |               |<-fixed          ^<-fixed
                      |               v                 |
                      |       "MSL_LOCAL_LEVEL"         |
                      |       --------------------------+
                      |               |                      
                      |               |                   "MSL_DLGA/DUHF"
                      |               |                   ---------------
                      |               |                     fixed-> ^
                      |               |                             |
                      |               |  "MSL_PLGA/TLGA/PUHF/MGA"   |
                      |               |  ------------------------   |
                      |               |               fixed-> ^     |
                      |               |                       |     |
                      |               |  "MSL_SPACECRAFT"     |     |
                      |               |  ----------------     |     |
                      |               |     ^ <-fixed         |     |
                      |               |     |                 |     | 
                      |               |     |   "MSL_CRUISE_STAGE"  |
                      |               |     |   ------------------  |
                      |               |     |     ^ <-fixed         |
                      |               |     |     |                 |
                      |               |     |     |   "MSL_DESCENT_STAGE"
                      |               |     |     |   ------------------
                 ck-> |               |     |     |     ^ <-fixed
                      |               |     |     |     |
                 "MSL_CACS"           |     |     |     |  "MSL_ROVER_MECH"
                 ----------           |     |     |     |  ----------------
                      ^               |     |     |     |     ^ <-fixed
                 ck-> |          ck-> |     |     |     |     |             
                      v               v                      
                                 "MSL_ROVER"
     +--------------------------------------------------------------+
     |     |      |   |   |   |       |       |    | | |      |     | 
     |     |<-fxd |   |   |   |<-fxd  |  fxd->|    | | | fxd->|     | 
     |     v      |   |   |   V       |       V    | | |      v     |
     | "MLS_RUHF" |   |   | "MSL_DAN" | "MSL_RAD"  | | | "MLS_RLGA" |
     | ---------- |   |   | --------- | ---------  | | | ---------- |
     |            |   |   |           |            | | |            | 
     |    fixed-> |   |   | <-fixed   |    fixed-> | | | <-fixed    | 
     |            V   |   V           |            V | V            | 
     | "MSL_REMS_UVS" | "MSL_HAZCAM*" |  "MSL_MARDI" | "MSL_SAM"    | 
     | -------------- | ------------- |  ----------- | ---------    | 
     |                |               |              |              | 
     |        fixed-> |               |              | <-fixed      | 
     |                V               |              V              | 
     |        "MSL_REMS_BOOM1/2"      |         "MSL_CHEMIN"        |
     |        ----------------        |         ------------        | 
     |                |               |                             | 
     |        fixed-> |               |                             | 
     |                V               |                             | 
     |      "MSL_REMS_BOOM1/2_*"      |                             |
     |      --------------------      |                             | 
     |                                |                             | 
     | <-fixed                        | <-fixed                     | <-fixed
     V                                V                             V 
  "MSL_RA_BASE"               "MSL_RSM_ZERO_AZ"              "MSL_HGA_ZERO_AZ"
  -------------                ----------------              -----------------
     |                                |                             | 
     | <-ck                           | <-ck                        | <-ck
     V                                V                             V 
  "MSL_RA_SHOULDER_AZ"          "MSL_RSM_AZ"                  "MSL_HGA_AZ"
  --------------------          ------------                  ------------
     |                                |                             | 
     | <-ck                           | <-fixed                     | <-fixed
     V                                V                             V 
  "MSL_RA_SHOULDER_EL"         "MSL_RSM_ZERO_EL"             "MSL_HGA_ZERO_EL"
  --------------------         -----------------             -----------------
     |                                |                             | 
     | <-ck                           | <-ck                        | <-ck
     V                                V                             V 
  "MSL_RA_ELBOW"                "MSL_RSM_EL"                  "MSL_HGA_EL"
  --------------                ------------                  ------------
     |                                |                             | 
     | <-ck                           |                             | <-fixed
     V                                |                             V 
  "MSL_RA_WRIST"                      |                         "MSL_HGA"
  --------------                      |                         ---------
     |                                |                             | 
     |                                |                             | <-fixed
     |                                |                             V
     |                                |                       "MSL_HGA_EB"
     |                                |                       ------------
     |                                |
     |                                | <-fixed
     |                                V
     |                           "MSL_RSM_HEAD"
     |                +-------------------------------+
     |                |               |               |
     |                | <-fixed       | <-fixed       | <-fixed
     |                V               V               V
     |       "MSL_MASTCAM_*"    "MSL_NAVCAM_*"    "MSL_CHEMCAM_RMI"
     |       ---------------    --------------   +------------------+
     |                                           |                  |
     |                                           | <-fixed          | <-fixed
     |                                           V                  V
     |                       "MSL_CHEMCAM_LIBS_LASER"   "MSL_CHEMCAM_LIBS_CAM"
     |                       ------------------------   ----------------------
     |              
     | <-ck         
     V              
  "MSL_RA_TURRET"
  ---------------------------------------------------------------------+
     |                |               |               |                |
     | <-ck           | <-ck          | <-ck          | <-ck           | <-ck
     V                V               V               V                V
  "MSL_MAHLI_REF" "MSL_APXS_REF" "MSL_PADS_REF" "MSL_DRT_REF" "MSL_CHIMRA_REF"
  --------------- -------------- -------------- ------------- ----------------
     |                |               |               |                |
     | <-fixed        | <-fixed       | <-fixed       | <-fixed        | <-fxd
     V                V               V               V                V
  "MSL_MAHLI"      "MSL_APXS"      "MSL_PADS"       "MSL_DRT"    "MSL_CHIMRA"
  -----------      ----------      ----------       ---------    ------------


   
MSL Surface Frames
========================================================================

   The surface frames layout in this version of the FK is based on the
   assumption that the total traverse distance during the mission will
   be relatively short (hundreds of meters, not kilometers) and,
   therefore, the local north and nadir directions, defining surface
   frame orientations, will be approximately the same at any point
   along the traverse path. This assumption allows defining surface
   frames as fixed offset frames with respect to each other and/or to
   Mars body-fixed frame, IAU_MARS.


Topocentric Frame
-------------------------------------------------

   MSL topocentric frame, MSL_TOPO, is defined as follows:

      -- +Z axis is along the outward normal at the landing site ("zenith");

      -- +X axis is along the local north direction ("north");

      -- +Y axis completes the right hand frame ("west");

      -- the origin of this frame is located at the landing site.

   Orientation of the frame is given relative to the body fixed
   rotating frame 'IAU_MARS' (x - along the line of zero longitude
   intersecting the equator, z - along the spin axis, y - completing
   the right hand coordinate frame.)

   The transformation from 'MSL_TOPO' frame to 'IAU_MARS' frame is a
   3-2-3 rotation with defined angles as the negative of the site
   longitude, the negative of the site co-latitude, 180 degrees.

   The landing site Gaussian longitude and latitude upon which the
   definition is built are:

      Lon = 137.441700 degrees East
      Lat =  -4.643851 degrees North

   The coordinates specified above are given with respect to the
   'IAU_MARS' instance defined by the rotation/shape model from the the
   PCK file 'pck00008.tpc'.

   These keywords implement the frame definition.

   \begindata

       FRAME_MSL_TOPO             = -76900
       FRAME_-76900_NAME           = 'MSL_TOPO'
       FRAME_-76900_CLASS          =  4
       FRAME_-76900_CLASS_ID       =  -76900
       FRAME_-76900_CENTER         =  -76900

       TKFRAME_-76900_RELATIVE     = 'IAU_MARS'
       TKFRAME_-76900_SPEC         = 'ANGLES'
       TKFRAME_-76900_UNITS        = 'DEGREES'
       TKFRAME_-76900_AXES         = ( 3, 2, 3 )
       TKFRAME_-76900_ANGLES       = ( -137.4417, -94.643851, 180.000  )

   \begintext


Local Level Frame
-------------------------------------------------

   MSL local level frame, MSL_LOCAL_LEVEL, is defined as follows:

      -- +Z axis is along the downward normal at the landing site ("nadir");

      -- +X axis is along the local north direction ("north");

      -- +Y axis completes the right hand frame ("east");

      -- the origin of this frame is located between the rover's middle
         wheels and moves with the rover.
 
   Since this frame is essentially the MSL_TOPO frame flipped by 180
   degrees about +X ("north") to point +Z down, this frame is defined
   as a fixed offset frame with respect to the MSL_TOPO frame.

   \begindata

      FRAME_MSL_LOCAL_LEVEL           = -76910
      FRAME_-76910_NAME               = 'MSL_LOCAL_LEVEL'
      FRAME_-76910_CLASS              = 4
      FRAME_-76910_CLASS_ID           = -76910
      FRAME_-76910_CENTER             = -76900
      TKFRAME_-76910_RELATIVE         = 'MSL_TOPO'
      TKFRAME_-76910_SPEC             = 'ANGLES'
      TKFRAME_-76910_UNITS            = 'DEGREES'
      TKFRAME_-76910_AXES             = (   1,       2,       3     )
      TKFRAME_-76910_ANGLES           = ( 180.000,   0.000,   0.000 )

   \begintext


Surface Fixed Frame
-------------------------------------------------

   MSL surface fixed frame -- MSL_SURFACE_FIXED -- is nominally
   co-aligned in orientation with the MSL_LOCAL_LEVEL frame but its
   origin does not move during the mission. Therefore, this frame is
   defined as a zero-offset, fixed frame with respect to the MSL_LOCAL_LEVEL
   frame.

   \begindata

      FRAME_MSL_SURFACE_FIXED         = -76920
      FRAME_-76920_NAME               = 'MSL_SURFACE_FIXED'
      FRAME_-76920_CLASS              = 4
      FRAME_-76920_CLASS_ID           = -76920
      FRAME_-76920_CENTER             = -76900
      TKFRAME_-76920_RELATIVE         = 'MSL_LOCAL_LEVEL'
      TKFRAME_-76920_SPEC             = 'ANGLES'
      TKFRAME_-76920_UNITS            = 'DEGREES'
      TKFRAME_-76920_AXES             = (   1,       2,       3     )
      TKFRAME_-76920_ANGLES           = (   0.000,   0.000,   0.000 )

   \begintext


MSL Rover Frames
========================================================================

   The MSL rover NAV frame, MSL_ROVER, is defined as follows:

      -  +X points to the front of the rover, away from RTG
  
      -  +Z points down

      -  +Y completes the right handed frame

      -  the origin on this frame is between the rover middle wheels
         (midpoint between and on the rotation axis of the middle
         wheels for deployed rover and suspension system on flat plane.

   The MSL rover NAV frame is shown on these diagrams:

      Rover -Y side view:
      -------------------
                               _
                              | | RSM
                              `-'
                               |              HGA
                               |          .```.         .
                               |         |  o  ---   .-' \ RTG
          RA                   |          `._.' | .-'     \
      -|-                  |-.-------------------'       .- 
       o---------o--------o| |                   |    .-' 
                             |.-------o----.     |-.-'
                           .-`--------------`-.--' 
                           |             .-----`o------.
                         .-|-.         .-|-.         .-|-.
                        |  o  |       |  o  |       |  o  |
                         `._.'   <-------x.'         `._.'
                                Xr       |
                                         |
                                         |
                                      Zr v     Yr is into the page.


      Rover -Z side ("top") view:
      ---------------------------

                        .-----.       .-----.       .-----.
                        |     |       |     |       |     |
                        |     |       |             |     |
                        `--|--'       `-- Yr        `--|--'
                           `----------o- ^ -----o------'
                             ..-.------- | ------.
                         RSM || |        |       |--------.
                             || |        |       |-------.|
          RA                 |`- <-------x       |       || RTG 
       |                     |  Xr           HGA |-------'|
      -o---------|--------|o-|            =====-o---------'
       |                     `-------------------' 
                           .----------o---------o------.
                        .--|--.       .--|--.       .--|--.
                        |     |       |     |       |     |
                        |     |       |     |       |     |
                        `-----'       `-----'       `-----'

                                               Zr is into the page.


   The orientation of this frame relative to other frames (J2000,
   MSL_LOCAL_LEVEL) changes in time and is provided in CK files.
   Therefore the MSL_ROVER frame is defined as a CK-based frame.

   \begindata

      FRAME_MSL_ROVER                 = -76000
      FRAME_-76000_NAME               = 'MSL_ROVER'
      FRAME_-76000_CLASS              = 3
      FRAME_-76000_CLASS_ID           = -76000
      FRAME_-76000_CENTER             = -76
      CK_-76000_SCLK                  = -76
      CK_-76000_SPK                   = -76

   \begintext

   The MSL rover mechanical frame -- MSL_ROVER_MECH -- is nominally
   co-aligned in orientation with the rover NAV frame, MSL_ROVER. The
   origin of this frame is different from the rover NAV frame origin
   and is translated from it by a fixed offset along the Z axis,
   provided in the MSL structures SPK file.

   The MSL rover and rover mechanical frames are shown on this diagram:

                               _
                              | | RSM
                              `-'
                               |              HGA
                               |          .```.         .
                               |         |  o  ---   .-' \ RTG
          RA                   |          `._.' | .-'     \
      -|-                  |-.-- <-------x ------'       .- 
       o---------o--------o| |  Xrm      |       |    .-' 
                             |.-------o- | .     |-.-'
                           .-`---------- | -`-.--' 
                           |         Zrm v ----`o------.
                         .-|-.         .- -.         .-|-.
                        |  o  |       |  o  |       |  o  |
                         `._.'   <-------x.'         `._.'
                                Xr       |
                                         |
                                         |
                                      Zr v     Yr, Yrm are into the page.


   The MSL_ROVER_MECH frame is defined below as fixed, zero-offset
   frame relative to the MSL_ROVER frame.
 
   \begindata

      FRAME_MSL_ROVER_MECH            = -76040
      FRAME_-76040_NAME               = 'MSL_ROVER_MECH'
      FRAME_-76040_CLASS              = 4
      FRAME_-76040_CLASS_ID           = -76040
      FRAME_-76040_CENTER             = -76
      TKFRAME_-76040_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76040_SPEC             = 'ANGLES'
      TKFRAME_-76040_UNITS            = 'DEGREES'
      TKFRAME_-76040_AXES             = (   1,       2,       3     )
      TKFRAME_-76040_ANGLES           = (   0.000,   0.000,   0.000 )

   \begintext


MSL Cruise and Descent Frames
========================================================================

   The following three MSL cruise and descent frames -- MSL_SPACECRAFT,
   MSL_CRUISE_STAGE, and MSL_DESCENT_STAGE -- are nominally co-aligned
   in orientation with the rover NAV frame, MSL_ROVER. The origins of
   these frames are different from the rover NAV frame origin and are
   translated from it fixed offsets along the Z axis, provided in the
   MSL structures SPK file.

   These frames are defined below as fixed, zero-offset frames
   relative to the MSL_ROVER frame.

   \begindata

      FRAME_MSL_SPACECRAFT            = -76010
      FRAME_-76010_NAME               = 'MSL_SPACECRAFT'
      FRAME_-76010_CLASS              = 4
      FRAME_-76010_CLASS_ID           = -76010
      FRAME_-76010_CENTER             = -76
      TKFRAME_-76010_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76010_SPEC             = 'ANGLES'
      TKFRAME_-76010_UNITS            = 'DEGREES'
      TKFRAME_-76010_AXES             = (   1,       2,       3     )
      TKFRAME_-76010_ANGLES           = (   0.000,   0.000,   0.000 )

      FRAME_MSL_CRUISE_STAGE          = -76020
      FRAME_-76020_NAME               = 'MSL_CRUISE_STAGE'
      FRAME_-76020_CLASS              = 4
      FRAME_-76020_CLASS_ID           = -76020
      FRAME_-76020_CENTER             = -76
      TKFRAME_-76020_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76020_SPEC             = 'ANGLES'
      TKFRAME_-76020_UNITS            = 'DEGREES'
      TKFRAME_-76020_AXES             = (   1,       2,       3     )
      TKFRAME_-76020_ANGLES           = (   0.000,   0.000,   0.000 )

      FRAME_MSL_DESCENT_STAGE         = -76030
      FRAME_-76030_NAME               = 'MSL_DESCENT_STAGE'
      FRAME_-76030_CLASS              = 4
      FRAME_-76030_CLASS_ID           = -76030
      FRAME_-76030_CENTER             = -76
      TKFRAME_-76030_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76030_SPEC             = 'ANGLES'
      TKFRAME_-76030_UNITS            = 'DEGREES'
      TKFRAME_-76030_AXES             = (   1,       2,       3     )
      TKFRAME_-76030_ANGLES           = (   0.000,   0.000,   0.000 )

   \begintext

   The MSL Cruise ACS frame, MSL_CACS, is defined as follows:

      -  +Z is co-aligned with the +Z axis of the MSL_ROVER frame
  
      -  +Y is directly over the location of the cruise stage star
         scanner

      -  +X completes the right handed frame and is directly over the
         B-string thruster cluster

      -  the origin on this frame is the same as of the MSL_SPACECRAFT
         frame.

   Nominally this frame is rotated -135 degrees about +Z from the
   MSL_ROVER frame. 

   The relation ship between MSL rover and CACS frames is shown on
   this diagram:

                              Xcacs
                            ^
                             \         .> Ycacs 
                              \     .-' 
                               \ .-'
                                x-------> Xr
                                |
                                |
                                |
                                v
                                 Yr       Zr, Zcacs are into the page.


   Because during cruise the orientation of this frame
   relative the J2000 frame comes in telemetry and is provided in CK
   files, this frame is defined as a CK-based frame.

   \begindata

      FRAME_MSL_CACS                  = -76050
      FRAME_-76050_NAME               = 'MSL_CACS'
      FRAME_-76050_CLASS              = 3
      FRAME_-76050_CLASS_ID           = -76050
      FRAME_-76050_CENTER             = -76
      CK_-76050_SCLK                  = -76
      CK_-76050_SPK                   = -76

   \begintext


MSL Antenna Frames
========================================================================

   This section defines frames for the MSL antennas -- Cruise X-band
   Parachute Cone Low Gain Antenna (PLGA), Cruise X-band Tilted Low
   Gain Antenna (TLGA), Cruise Parachute Cone UHF Antenna (PUHF),
   Cruise Medium Gain Antenna (MGA), X-band Descent Stage Low Gain
   Antenna (DLGA), Descent Stage UHF Antenna (DUHF), X-band Rover Low
   Gain Antenna (RLGA), Rover UHF Antenna (RUHF), X-band High Gain
   Antenna (HGA).

Cruise Antennas
-------------------------------------------------

   The frames for antennas mounted on the MSL cruise stage and
   parachute capsule -- MSL_PLGA, MSL_TLGA, MSL_PUHF, MSL_MGA -- are
   fixed with respect to the cruise stage frame, MSL_CRUISE_STAGE, and
   defined as follows:

      -  +Z is the antenna boresight, which is nominally along the
         cruise stage -Z except for TLGA
  
      -  +X is nominally co-aligned with the the cruise stage +X,
         except for TLGA

      -  +Y completes the right handed frame

      -  the origin is at the center of the antenna side farthest from
         the mounting plate (rim, tip, etc).

   The MSL_PLGA, MSL_PUHF, and MSL_MGA frames are rotated by 180 degrees 
   about +X from the MSL_CRUISE_STAGE frame.

   The MSL_TLGA frame is first rotated by 180 degrees about +X, then by
   -17.5 (TBD) degrees about +Y from the MSL_CRUISE_STAGE frame.

   These frames are defined below as fixed frames relative to the
   MSL_CRUISE_STAGE frame.

   \begindata

      FRAME_MSL_PLGA                  = -76060
      FRAME_-76060_NAME               = 'MSL_PLGA'
      FRAME_-76060_CLASS              = 4
      FRAME_-76060_CLASS_ID           = -76060
      FRAME_-76060_CENTER             = -76
      TKFRAME_-76060_RELATIVE         = 'MSL_CRUISE_STAGE'
      TKFRAME_-76060_SPEC             = 'ANGLES'
      TKFRAME_-76060_UNITS            = 'DEGREES'
      TKFRAME_-76060_AXES             = (   1,       2,       3     )
      TKFRAME_-76060_ANGLES           = ( 180.000,   0.000,   0.000 )

      FRAME_MSL_TLGA                  = -76061
      FRAME_-76061_NAME               = 'MSL_TLGA'
      FRAME_-76061_CLASS              = 4
      FRAME_-76061_CLASS_ID           = -76061
      FRAME_-76061_CENTER             = -76
      TKFRAME_-76061_RELATIVE         = 'MSL_CRUISE_STAGE'
      TKFRAME_-76061_SPEC             = 'ANGLES'
      TKFRAME_-76061_UNITS            = 'DEGREES'
      TKFRAME_-76061_AXES             = (   1,       2,       3     )
      TKFRAME_-76061_ANGLES           = ( 180.000,  17.500,   0.000 )

      FRAME_MSL_PUHF                  = -76062
      FRAME_-76062_NAME               = 'MSL_PUHF'
      FRAME_-76062_CLASS              = 4
      FRAME_-76062_CLASS_ID           = -76062
      FRAME_-76062_CENTER             = -76
      TKFRAME_-76062_RELATIVE         = 'MSL_CRUISE_STAGE'
      TKFRAME_-76062_SPEC             = 'ANGLES'
      TKFRAME_-76062_UNITS            = 'DEGREES'
      TKFRAME_-76062_AXES             = (   1,       2,       3     )
      TKFRAME_-76062_ANGLES           = ( 180.000,   0.000,   0.000 )

      FRAME_MSL_MGA                   = -76063
      FRAME_-76063_NAME               = 'MSL_MGA'
      FRAME_-76063_CLASS              = 4
      FRAME_-76063_CLASS_ID           = -76063
      FRAME_-76063_CENTER             = -76
      TKFRAME_-76063_RELATIVE         = 'MSL_CRUISE_STAGE'
      TKFRAME_-76063_SPEC             = 'ANGLES'
      TKFRAME_-76063_UNITS            = 'DEGREES'
      TKFRAME_-76063_AXES             = (   1,       2,       3     )
      TKFRAME_-76063_ANGLES           = ( 180.000,   0.000,   0.000 )

   \begintext


EDL Antennas
-------------------------------------------------

   The frames for the antennas mounted on the MSL descent stage --
   MSL_DLGA, MSL_DUHF -- are fixed with respect to the descent stage
   frame, MSL_DESCENT_STAGE, and defined as follows:

      -  +Z is the antenna boresight, nominally along the descent stage
         -Z
  
      -  +X is nominally co-aligned with the the descent stage +X

      -  +Y completes the right handed frame

      -  the origin is at the center of the antenna side farthest from
         the mounting plate (rim, tip, etc).

   The MSL_DLGA and MSL_DUHF frames are rotated by 180 degrees about +X
   from the MSL_DESCENT_STAGE frame.

   These frames are defined below as fixed frames relative to the
   MSL_DESCENT_STAGE frame.

   \begindata

      FRAME_MSL_DLGA                  = -76064
      FRAME_-76064_NAME               = 'MSL_DLGA'
      FRAME_-76064_CLASS              = 4
      FRAME_-76064_CLASS_ID           = -76064
      FRAME_-76064_CENTER             = -76
      TKFRAME_-76064_RELATIVE         = 'MSL_DESCENT_STAGE'
      TKFRAME_-76064_SPEC             = 'ANGLES'
      TKFRAME_-76064_UNITS            = 'DEGREES'
      TKFRAME_-76064_AXES             = (   1,       2,       3     )
      TKFRAME_-76064_ANGLES           = ( 180.000,   0.000,   0.000 )

      FRAME_MSL_DUHF                  = -76065
      FRAME_-76065_NAME               = 'MSL_DUHF'
      FRAME_-76065_CLASS              = 4
      FRAME_-76065_CLASS_ID           = -76065
      FRAME_-76065_CENTER             = -76
      TKFRAME_-76065_RELATIVE         = 'MSL_DESCENT_STAGE'
      TKFRAME_-76065_SPEC             = 'ANGLES'
      TKFRAME_-76065_UNITS            = 'DEGREES'
      TKFRAME_-76065_AXES             = (   1,       2,       3     )
      TKFRAME_-76065_ANGLES           = ( 180.000,   0.000,   0.000 )

   \begintext


Rover Antennas
-------------------------------------------------

   The frames for the two MSL non-articulating antennas mounted on the
   rover body -- MSL_RLGA, MSL_RUHF -- are fixed with respect to the
   rover frame, MSL_ROVER, and defined as follows:

      -  +Z is the antenna boresight, nominally along the rover -Z
  
      -  +X is nominally co-aligned with the the rover +X

      -  +Y completes the right handed frame

      -  the origin is at the center of the antenna side farthest from
         the mounting plate (rim, tip, etc).

   The MSL_RLGA and MSL_RUHF frames are rotated by 180 degrees about +X
   from the MSL_ROVER frame.

   These frames are defined below as fixed frames relative to the
   MSL_ROVER frame.

   \begindata

      FRAME_MSL_RLGA                  = -76110
      FRAME_-76110_NAME               = 'MSL_RLGA'
      FRAME_-76110_CLASS              = 4
      FRAME_-76110_CLASS_ID           = -76110
      FRAME_-76110_CENTER             = -76
      TKFRAME_-76110_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76110_SPEC             = 'ANGLES'
      TKFRAME_-76110_UNITS            = 'DEGREES'
      TKFRAME_-76110_AXES             = (   1,       2,       3     )
      TKFRAME_-76110_ANGLES           = ( 180.000,   0.000,   0.000 )

      FRAME_MSL_RUHF                  = -76111
      FRAME_-76111_NAME               = 'MSL_RUHF'
      FRAME_-76111_CLASS              = 4
      FRAME_-76111_CLASS_ID           = -76111
      FRAME_-76111_CENTER             = -76
      TKFRAME_-76111_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76111_SPEC             = 'ANGLES'
      TKFRAME_-76111_UNITS            = 'DEGREES'
      TKFRAME_-76111_AXES             = (   1,       2,       3     )
      TKFRAME_-76111_ANGLES           = ( 180.000,   0.000,   0.000 )

   \begintext


   The frame chain for the MSL articulating high gain antenna (HGA)
   includes six frames -- MSL_HGA_ZERO_AZ, MSL_HGA_AZ, MSL_HGA_ZERO_EL,
   MSL_HGA_EL, MSL_HGA, and MSL_HGA_EB.

   Only the first frame -- MSL_HGA_ZERO_AZ -- and the final two frames
   -- MSL_HGA and MSL_HGA_EB -- match the frames defined in [4] and
   used in the HGA calibration documentation. The intermediate frames
   not matching the documentation were introduced model the HGA
   pointing using a more traditional kinematics approach.

   The MSL_HGA_ZERO_AZ frame (called HGAS frame in [4]) is a fixed
   offset frame that has the +Z axis along the AZ rotation axis and the
   +X axis in the direction of the EL rotation axis for the HGA in the
   zero AZ/EL position. This frame is nominally rotated from the
   MSL_ROVER frame by +25 degrees about +Z.

   The MSL_HGA_AZ frame is a CK-based frame rotated from the
   MSL_HGA_ZERO_AZ frame by the -AZ angle about the +Z axis.

   The MSL_HGA_ZERO_EL frame is a fixed offset frame that has the +X axis
   along the EL rotation axis and the +Z axis in the direction of the AZ
   rotation axis. This frame is nominally co-aligned with the MSL_HGA_AZ 
   frame.

   The MSL_HGA_EL frame is a CK-based frame rotated from the
   MSL_HGA_ZERO_EL frame by the EL angle about the +X axis.

   The MSL_HGA frame (called ANT frame in [4]) is a fixed-offset frame
   nominally rotated from the MSL_HGA_EL frame by 180 degrees about the
   +X axis to align the +Z axis of the frame with the antenna boresight
   defined as the normal to the antenna's radiating surface.

   The MSL_HGA_EB frame (called EB frame in [4]) is a fixed offset
   frame nominally co-aligned with the MSL_HGA frame. This frame's +Z
   axis is the antenna boresight defined as the antenna electric
   boresight.

   The HGA frames for the HGA in zero AZ/EL position are shown on this
   diagram:

      Rover -Z side ("top") view, HGA in zero AZ/EL position:
      -------------------------------------------------------

                        .-----.       .-----.       .-----.
                        |     |       |     |       |     |
                        |     |       |             |     |
                        `--|--'       `-- Yr        `--|--'
                           `----------o- ^ -----o------'
                             ..-.------- | ------.
                         RSM || |        |         +Yhgaaz*,el*
                             || |        | +Xhga*  ^   --.|
          RA                 |`- <-------x<.```.  /      || RTG 
       |                     |  Xr        | `o. |/  -----'|
      -o---------|--------|o-|             `/_.'x. -------'
       |                     `-------------/-----
                           .----------o-  v  ----------.
                        .--|--.       .-  +Yhga,eb  .--|--.
                        |     |       |             |     |
                        |     |       |     |       |     |
                        `-----'       `-----'       `-----'

                     Zr, Zhgaaz0, Zhgaaz, Zhgael0, Zhgael are into the page.
                                Zhga, Zhgaeb are out of the page.


      HGA assembly "side" view, HGA in zero AZ/EL position:
      -----------------------------------------------------

                                      Azimuth
                    +Zhga,eb         rotation
                   ^                   axis
                   |                     |
       .-----------|-----------..--------|--------.-----
       | Xhga,eb   |         Xhgael0,el    Yhgael0,al   |---.     Elevation  
       |    <------o        ----  <------x   -----------------     rotation 
       |            Yhga,eb    |_/       |    |   |     |---'       axis    
       `-----------------------.         |    |   |-----'
                           .'.'          |    |   |
                     .   .'.'            v Zhgael0,el 
                     \\.'.'                   |   |
                      \.'      .-------------------.
                               `-------------------'
                                 |               |
                                  |             |
       -------------------------- <------x ---------------
         Rover deck          Xhgaaz0,az  | Yhgaaz0,az
                                         |
                                         |
                                         v Zhgaaz0,az

                                         Yhgag, Yhga are out of the page.


   The MSL_HGA_AZ, MSL_HGA_ZERO_EL, and MSL_HGA_EL frames are
   co-aligned with the MSL_HGA_ZERO_AZ frame in the zero AZ/EL
   position.

   The nominal nominal offset angles for the fixed offset frames in the
   HGA frame chain are:

      TKFRAME_-76121_ANGLES           = ( -25.0,   0.0,   0.0 )
      TKFRAME_-76121_AXES             = (   3,     1,     2   )

      TKFRAME_-76123_ANGLES           = (   0.0,   0.0,   0.0 )
      TKFRAME_-76123_AXES             = (   3,     1,     2   )

      TKFRAME_-76125_ANGLES           = (   0.0,   0.0, 180.0 )
      TKFRAME_-76125_AXES             = (   2,     3,     1   )

      TKFRAME_-76126_ANGLES           = (   0.0,   0.0,   0.0 )
      TKFRAME_-76126_AXES             = (   3,     1,     3   )

   The actual offset angles were derived from the following HGA base frame
   quaternion, directions of the HGA AZ and EL axes in the rover
   frame at "zero" AZ position and HGA calibrated maximum transmission
   gain direction (from [6] and [7]):

      base_q1        0.0015877
      base_q2       -0.0009219
      base_q3       -0.2166469
      base_q4        0.9762428
      az_axis_x     -0.0007
      az_axis_y     -0.0012
      az_axis_z      1.0000
      el_axis_x      0.9099
      el_axis_y      0.4148
      el_axis_z      0.0006
      el_azimuth     0.0

      [Tx]=          0.999955833 0.000000000 -0.009398543
                    -0.000023669 0.999996829 -0.002518213
                     0.009398513 0.002518324 0.999952662

   The frame definitions below incorporate the actual offsets.

   \begindata

      FRAME_MSL_HGA_ZERO_AZ           = -76121
      FRAME_-76121_NAME               = 'MSL_HGA_ZERO_AZ'
      FRAME_-76121_CLASS              = 4
      FRAME_-76121_CLASS_ID           = -76121
      FRAME_-76121_CENTER             = -76
      TKFRAME_-76121_SPEC             = 'ANGLES'
      TKFRAME_-76121_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76121_ANGLES           = (  -24.507049,  -0.045924,   0.065014 )
      TKFRAME_-76121_AXES             = (    3,          1,          2        )
      TKFRAME_-76121_UNITS            = 'DEGREES'

      FRAME_MSL_HGA_AZ                = -76122
      FRAME_-76122_NAME               = 'MSL_HGA_AZ'
      FRAME_-76122_CLASS              = 3
      FRAME_-76122_CLASS_ID           = -76122
      FRAME_-76122_CENTER             = -76
      CK_-76122_SCLK                  = -76
      CK_-76122_SPK                   = -76

      FRAME_MSL_HGA_ZERO_EL           = -76123
      FRAME_-76123_NAME               = 'MSL_HGA_ZERO_EL'
      FRAME_-76123_CLASS              = 4
      FRAME_-76123_CLASS_ID           = -76123
      FRAME_-76123_CENTER             = -76
      TKFRAME_-76123_SPEC             = 'ANGLES'
      TKFRAME_-76123_RELATIVE         = 'MSL_HGA_AZ'
      TKFRAME_-76123_ANGLES           = (    0.000000,  -0.000000,  -0.030636 )
      TKFRAME_-76123_AXES             = (    3,          1,          2        )
      TKFRAME_-76123_UNITS            = 'DEGREES'

      FRAME_MSL_HGA_EL                = -76124
      FRAME_-76124_NAME               = 'MSL_HGA_EL'
      FRAME_-76124_CLASS              = 3
      FRAME_-76124_CLASS_ID           = -76124
      FRAME_-76124_CENTER             = -76
      CK_-76124_SCLK                  = -76
      CK_-76124_SPK                   = -76

      FRAME_MSL_HGA                   = -76125
      FRAME_-76125_NAME               = 'MSL_HGA'
      FRAME_-76125_CLASS              = 4
      FRAME_-76125_CLASS_ID           = -76125
      FRAME_-76125_CENTER             = -76
      TKFRAME_-76125_SPEC             = 'ANGLES'
      TKFRAME_-76125_RELATIVE         = 'MSL_HGA_EL'
      TKFRAME_-76125_ANGLES           = (  -0.177348,  -0.517152, -179.798057 )
      TKFRAME_-76125_AXES             = (    2,          3,          1        )
      TKFRAME_-76125_UNITS            = 'DEGREES'

      FRAME_MSL_HGA_EB                = -76126
      FRAME_-76126_NAME               = 'MSL_HGA_EB'
      FRAME_-76126_CLASS              = 4
      FRAME_-76126_CLASS_ID           = -76126
      FRAME_-76126_CENTER             = -76
      TKFRAME_-76126_SPEC             = 'ANGLES'
      TKFRAME_-76126_RELATIVE         = 'MSL_HGA'
      TKFRAME_-76126_ANGLES           = (   75.000000,   0.557500, -75.000677 )
      TKFRAME_-76126_AXES             = (    3,          1,          3        )
      TKFRAME_-76126_UNITS            = 'DEGREES'

   \begintext


Rover-mounted Instrument Frames
========================================================================

   This section defines frames for the instruments and structures
   mounted on the rover body -- Hazard Avoidance Cameras (HAZCAMs),
   Radiation Assessment Detector (RAD), Mars Descent Imager (MARDI), UV
   and pressure sensors of the Rover Environmental Monitoring Station
   (REMS/UVS, REMS/PS), Dynamic Albedo of Neutrons experiment (DAN),
   Sample Analysis at Mars Instrument Suite (SAM), and Chemistry &
   Mineralogy X-Ray Diffraction/X-Ray Fluorescence Instrument (CHEMIN).


HAZCAMs
-------------------------------------------------

   The frame for each of the eight MSL HAZCAMs (FRONT/REAR - LEFT/RIGHT
   - A/B) is defined as follows:

      -- +Z axis is along the camera's central pixel view direction 
         ("into image");

      -- +Y axis is along the image central column and points from the
         image center toward the image top row ("up");

      -- +X completes the right hand frame and is along the image central
         row and points from the image center toward the image left 
         column ("left");

      -- the origin of the frame is located at the camera focal point.

   This diagram illustrates rover HAZCAM frames:

      Rover -Y side view:
      -------------------
                               _
                              | | RSM
                              `-'
                               |              HGA
                               |          .```.   
                               |         |  o  ---
          RA       Yhcf        |          `._.' |         Yhcr 
      -|-             <.     .-------------------'      .>   
       o---------o-     `.   |                   |    .' 
                          `. | -------o----.     |  .'
                            o`--------------`-.--'x'
                          .'             .-----`   `.   
                        .'             .-|-.         `.   
                      <'              |  o  |          `> 
                  Zhcf           <-------x.'              Zhcr
                               Xr        |
                                         |
                                         |
                                      Zr v     Xhcr are into the page.
                                               Xhcf are out of the page.


   Since all HAZCAMs are rigidly mounted on the rover body, their frames 
   are defined as fixed offset frames with orientation given with respect 
   to the rover frame.

   The nominal orientation for the HAZCAM frames are with
   boresight in the direction of the rover +X and tilted 45 degrees
   toward the ground for FRONT HAZCAMs and in the direction of -X  and
   tilted 45 degrees toward the ground for REAR HAZCAMs. The
   following sets of keywords can be incorporated into the frame
   definitions to provide this nominal orientation (provided for
   reference only):

      TKFRAME_-76131_ANGLES           = ( 0.0, -45.0,  90.0 )
      TKFRAME_-76131_AXES             = ( 1,     2,     3   )

      TKFRAME_-76132_ANGLES           = ( 0.0, -45.0,  90.0 )
      TKFRAME_-76132_AXES             = ( 1,     2,     3   )

      TKFRAME_-76133_ANGLES           = ( 0.0, -45.0,  90.0 )
      TKFRAME_-76133_AXES             = ( 1,     2,     3   )

      TKFRAME_-76134_ANGLES           = ( 0.0, -45.0,  90.0 )
      TKFRAME_-76134_AXES             = ( 1,     2,     3   )

      TKFRAME_-76141_ANGLES           = ( 0.0,  45.0, -90.0 )
      TKFRAME_-76141_AXES             = ( 1,     2,     3   )

      TKFRAME_-76142_ANGLES           = ( 0.0,  45.0, -90.0 )
      TKFRAME_-76142_AXES             = ( 1,     2,     3   )

      TKFRAME_-76143_ANGLES           = ( 0.0,  45.0, -90.0 )
      TKFRAME_-76143_AXES             = ( 1,     2,     3   )

      TKFRAME_-76144_ANGLES           = ( 0.0,  45.0, -90.0 )
      TKFRAME_-76144_AXES             = ( 1,     2,     3   )


   The actual MSL_HAZCAM_FRONT_LEFT_A frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_003_SN_0205-FHAZL-FLIGHT-RCE-A.cahvore'. According to
   this model the reference frame, MSL_ROVER, can be transformed into
   the camera frame, MSL_HAZCAM_FRONT_LEFT_A, by the following sequence
   of rotations: first by 46.52247224 degrees about Y, then by
   -0.75466490 degrees about X, and finally by -89.62067207 degrees
   about Z.

   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_FRONT_LEFT_A    = -76131
      FRAME_-76131_NAME               = 'MSL_HAZCAM_FRONT_LEFT_A'
      FRAME_-76131_CLASS              = 4
      FRAME_-76131_CLASS_ID           = -76131
      FRAME_-76131_CENTER             = -76
      TKFRAME_-76131_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76131_SPEC             = 'ANGLES'
      TKFRAME_-76131_UNITS            = 'DEGREES'
      TKFRAME_-76131_AXES             = (    2,        1,        3     )
      TKFRAME_-76131_ANGLES           = (  -46.522,    0.755,   89.621 )

   \begintext


   The actual MSL_HAZCAM_FRONT_LEFT_B frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_003_SN_0208-FHAZL-FLIGHT-RCE-B.cahvore'. According to
   this model the reference frame, MSL_ROVER, can be transformed into
   the camera frame, MSL_HAZCAM_FRONT_LEFT_B, by the following sequence
   of rotations: first by 45.49902630 degrees about Y, then by
   -1.29539651 degrees about X, and finally by -89.95051513 degrees
   about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_FRONT_LEFT_B    = -76133
      FRAME_-76133_NAME               = 'MSL_HAZCAM_FRONT_LEFT_B'
      FRAME_-76133_CLASS              = 4
      FRAME_-76133_CLASS_ID           = -76133
      FRAME_-76133_CENTER             = -76
      TKFRAME_-76133_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76133_SPEC             = 'ANGLES'
      TKFRAME_-76133_UNITS            = 'DEGREES'
      TKFRAME_-76133_AXES             = (    2,        1,        3     )
      TKFRAME_-76133_ANGLES           = (  -45.499,    1.295,   89.951 )

   \begintext


   The actual MSL_HAZCAM_FRONT_RIGHT_A frame orientation provided in
   the frame definition below was computed using the CAHVOR(E) camera
   model file, 'MSL_CAL_003_SN_0213-FHAZR-FLIGHT-RCE-A.cahvore'.
   According to this model the reference frame, MSL_ROVER, can be
   transformed into the camera frame, MSL_HAZCAM_FRONT_RIGHT_A, by the
   following sequence of rotations: first by 46.24015677 degrees about
   Y, then by 0.01873572 degrees about X, and finally by -89.99878481
   degrees about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_FRONT_RIGHT_A   = -76132
      FRAME_-76132_NAME               = 'MSL_HAZCAM_FRONT_RIGHT_A'
      FRAME_-76132_CLASS              = 4
      FRAME_-76132_CLASS_ID           = -76132
      FRAME_-76132_CENTER             = -76
      TKFRAME_-76132_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76132_SPEC             = 'ANGLES'
      TKFRAME_-76132_UNITS            = 'DEGREES'
      TKFRAME_-76132_AXES             = (    2,        1,        3     )
      TKFRAME_-76132_ANGLES           = (  -46.240,   -0.019,   89.999 )

   \begintext


   The actual MSL_HAZCAM_FRONT_RIGHT_B frame orientation provided in
   the frame definition below was computed using the CAHVOR(E) camera
   model file, 'MSL_CAL_003_SN_0209-FHAZR-FLIGHT-RCE-B.cahvore'.
   According to this model the reference frame, MSL_ROVER, can be
   transformed into the camera frame, MSL_HAZCAM_FRONT_RIGHT_B, by the
   following sequence of rotations: first by 45.15683382 degrees about
   Y, then by -0.35151197 degrees about X, and finally by -90.27295863
   degrees about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_FRONT_RIGHT_B   = -76134
      FRAME_-76134_NAME               = 'MSL_HAZCAM_FRONT_RIGHT_B'
      FRAME_-76134_CLASS              = 4
      FRAME_-76134_CLASS_ID           = -76134
      FRAME_-76134_CENTER             = -76
      TKFRAME_-76134_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76134_SPEC             = 'ANGLES'
      TKFRAME_-76134_UNITS            = 'DEGREES'
      TKFRAME_-76134_AXES             = (    2,        1,        3     )
      TKFRAME_-76134_ANGLES           = (  -45.157,    0.352,   90.273 )

   \begintext


   The actual MSL_HAZCAM_BACK_LEFT_A frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_003_SN_0211-RHAZL-FLIGHT-RCE-A.cahvore'. According to
   this model the reference frame, MSL_ROVER, can be transformed into
   the camera frame, MSL_HAZCAM_BACK_LEFT_A, by the following sequence
   of rotations: first by -45.06141007 degrees about Y, then by
   1.82024949 degrees about X, and finally by 90.60566349 degrees about
   Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_BACK_LEFT_A     = -76141
      FRAME_-76141_NAME               = 'MSL_HAZCAM_BACK_LEFT_A'
      FRAME_-76141_CLASS              = 4
      FRAME_-76141_CLASS_ID           = -76141
      FRAME_-76141_CENTER             = -76
      TKFRAME_-76141_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76141_SPEC             = 'ANGLES'
      TKFRAME_-76141_UNITS            = 'DEGREES'
      TKFRAME_-76141_AXES             = (    2,        1,        3     )
      TKFRAME_-76141_ANGLES           = (   45.061,   -1.820,  -90.606 )

   \begintext


   The actual MSL_HAZCAM_BACK_LEFT_B frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_003_SN_0212-RHAZL-FLIGHT-RCE-B.cahvore'. According to
   this model the reference frame, MSL_ROVER, can be transformed into
   the camera frame, MSL_HAZCAM_BACK_LEFT_B, by the following sequence
   of rotations: first by -45.44850801 degrees about Y, then by
   0.39620359 degrees about X, and finally by 90.00485341 degrees about
   Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_BACK_LEFT_B     = -76143
      FRAME_-76143_NAME               = 'MSL_HAZCAM_BACK_LEFT_B'
      FRAME_-76143_CLASS              = 4
      FRAME_-76143_CLASS_ID           = -76143
      FRAME_-76143_CENTER             = -76
      TKFRAME_-76143_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76143_SPEC             = 'ANGLES'
      TKFRAME_-76143_UNITS            = 'DEGREES'
      TKFRAME_-76143_AXES             = (    2,        1,        3     )
      TKFRAME_-76143_ANGLES           = (   45.449,   -0.396,  -90.005 )

   \begintext


   The actual MSL_HAZCAM_BACK_RIGHT_A frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_003_SN_0217-RHAZR-FLIGHT-RCE-A.cahvore'. According to
   this model the reference frame, MSL_ROVER, can be transformed into
   the camera frame, MSL_HAZCAM_BACK_RIGHT_A, by the following sequence
   of rotations: first by -46.45607709 degrees about Y, then by
   1.48194872 degrees about X, and finally by 90.38345774 degrees about
   Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_BACK_RIGHT_A    = -76142
      FRAME_-76142_NAME               = 'MSL_HAZCAM_BACK_RIGHT_A'
      FRAME_-76142_CLASS              = 4
      FRAME_-76142_CLASS_ID           = -76142
      FRAME_-76142_CENTER             = -76
      TKFRAME_-76142_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76142_SPEC             = 'ANGLES'
      TKFRAME_-76142_UNITS            = 'DEGREES'
      TKFRAME_-76142_AXES             = (    2,        1,        3     )
      TKFRAME_-76142_ANGLES           = (   46.456,   -1.482,  -90.383 )

   \begintext


   The actual MSL_HAZCAM_BACK_RIGHT_B frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_003_SN_0207-RHAZR-FLIGHT-RCE-B.cahvore'. According to
   this model the reference frame, MSL_ROVER, can be transformed into
   the camera frame, MSL_HAZCAM_BACK_RIGHT_B, by the following sequence
   of rotations: first by -43.65002223 degrees about Y, then by
   -0.44628107 degrees about X, and finally by 89.00921165 degrees
   about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_HAZCAM_BACK_RIGHT_B    = -76144
      FRAME_-76144_NAME               = 'MSL_HAZCAM_BACK_RIGHT_B'
      FRAME_-76144_CLASS              = 4
      FRAME_-76144_CLASS_ID           = -76144
      FRAME_-76144_CENTER             = -76
      TKFRAME_-76144_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76144_SPEC             = 'ANGLES'
      TKFRAME_-76144_UNITS            = 'DEGREES'
      TKFRAME_-76144_AXES             = (    2,        1,        3     )
      TKFRAME_-76144_ANGLES           = (   43.650,    0.446,  -89.009 )

   \begintext


RAD
-------------------------------------------------

   The RAD frame -- MSL_RAD -- is fixed with respect to the rover
   frame, MSL_ROVER, and defined as follows:

      -  +Z is the instrument boresight, nominally along the rover -Z
  
      -  +X is nominally co-aligned with the the rover +X

      -  +Y completes the right handed frame

      -  the origin is at the intersection of the instrument FOV center
         axis and the rover deck plane.

   The MSL_RAD frame is rotated by 180 degrees about +X from the
   MSL_ROVER frame.

   \begindata

      FRAME_MSL_RAD                   = -76150
      FRAME_-76150_NAME               = 'MSL_RAD'
      FRAME_-76150_CLASS              = 4
      FRAME_-76150_CLASS_ID           = -76150
      FRAME_-76150_CENTER             = -76
      TKFRAME_-76150_SPEC             = 'ANGLES'
      TKFRAME_-76150_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76150_ANGLES           = ( 180.0, 0.0, 0.0 )
      TKFRAME_-76150_AXES             = (   1,   2,   3   )
      TKFRAME_-76150_UNITS            = 'DEGREES'

   \begintext


MARDI
-------------------------------------------------

   The frame for MARDI -- MSL_MARDI -- is defined as follows:

      -- +Z axis is along the camera's central pixel view direction 
         ("into image");

      -- +Y axis is along the image central column and points from the
         image center toward the image top row ("up");

      -- +X completes the right hand frame and is along the image central
         row and points from the image center toward the image left 
         column ("left");

      -- the origin of the frame is located at the camera focal point.

   This diagram illustrates the MARDI frame:

      Rover -Y side view:
      -------------------
                               _
                              | | RSM
                              `-'
                               |              HGA
                               |          .```.         .
                               |         |  o  ---   .-' \ RTG
          RA                   |          `._.' | .-'     \
      -|-                  |-.-------------------'       .- 
       o---------o--------o| |                   |    .-' 
                             |.-------o----.     |-.-'
                             `x------->-----`-.--' 
                              |  Ymardi  .-----`o------.
                              |        .-|-.         .-|-.
                              |       |  o  |       |  o  |
                              v  <-------x.'         `._.'
                        Zmardi   Xr      |
                                         |
                                         |
                                      Zr v     Yr is into the page.
                                               Xmardi is into the page.

   Since MARDI are rigidly mounted on the rover body, its frame is
   defined as fixed offset frames with orientation given with respect
   to the rover frame.

   Nominally the MARDI frame is rotated by 90 degrees about Z relative 
   to the rover frame. This nominal orientation can be defined using the
   following angles:

      TKFRAME_-76160_ANGLES           = ( 0.0, 0.0, -90.0 )
      TKFRAME_-76160_AXES             = ( 1,   2,     3   )


   The actual MSL_MARDI frame orientation provided in the frame
   definition below was computed using the CAHVOR(E) camera model file,
   'MSL_CAL_002_SN_3001-MARDI-FLIGHT.cahvor'. According to this model
   the reference frame, MSL_ROVER, can be transformed into the camera
   frame, MSL_MARDI, by the following sequence of rotations: first by
   0.60374856 degrees about Y, then by 0.35729881 degrees about X, and
   finally by 89.88147298 degrees about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_MARDI                  = -76160
      FRAME_-76160_NAME               = 'MSL_MARDI'
      FRAME_-76160_CLASS              = 4
      FRAME_-76160_CLASS_ID           = -76160
      FRAME_-76160_CENTER             = -76
      TKFRAME_-76160_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76160_SPEC             = 'ANGLES'
      TKFRAME_-76160_UNITS            = 'DEGREES'
      TKFRAME_-76160_AXES             = (    2,        1,        3     )
      TKFRAME_-76160_ANGLES           = (   -0.604,   -0.357,  -89.881 )

   \begintext


REMS UVS
--------

   The REMS/UVS frame -- MSL_REMS_UVS -- is fixed with respect to the
   rover frame, MSL_ROVER, and defined as follows:

      -  +Z is the instrument boresight, nominally along the rover -Z
  
      -  +X is nominally co-aligned with the the rover +X

      -  +Y completes the right handed frame

      -  the origin is at the intersection of the instrument FOV center
         axis and the rover deck plane.

   The MSL_REMS_UVS frame is rotated by 180 degrees about +X from the
   MSL_ROVER frame.

   \begindata

      FRAME_MSL_REMS_UVS              = -76170
      FRAME_-76170_NAME               = 'MSL_REMS_UVS'
      FRAME_-76170_CLASS              = 4
      FRAME_-76170_CLASS_ID           = -76170
      FRAME_-76170_CENTER             = -76
      TKFRAME_-76170_SPEC             = 'ANGLES'
      TKFRAME_-76170_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76170_ANGLES           = ( 180.0, 0.0, 0.0 )
      TKFRAME_-76170_AXES             = (   1,   2,   3   )
      TKFRAME_-76170_UNITS            = 'DEGREES'

   \begintext


DAN
-------------------------------------------------

   The DAN frame -- MSL_DAN -- is fixed with respect to the rover
   frame, MSL_ROVER, and defined as follows:

      -  +Z is the instrument boresight, nominally along the rover +Z
  
      -  +X is nominally co-aligned with the the rover +X

      -  +Y completes the right handed frame

      -  the origin is at the intersection of the instrument FOV center
         axis and the rover bottom plane.

   The MSL_DAN frame is co-aligned wit the MSL_ROVER frame.

   \begindata

      FRAME_MSL_DAN                   = -76180
      FRAME_-76180_NAME               = 'MSL_DAN'
      FRAME_-76180_CLASS              = 4
      FRAME_-76180_CLASS_ID           = -76180
      FRAME_-76180_CENTER             = -76
      TKFRAME_-76180_SPEC             = 'ANGLES'
      TKFRAME_-76180_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76180_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76180_AXES             = ( 1,   2,   3   )
      TKFRAME_-76180_UNITS            = 'DEGREES'

   \begintext


SAM and CHEMIN
-------------------------------------------------

   The SAM and CHEMIN frames -- MSL_SAM, and MSL_CHEMIN -- are fixed
   with respect to the rover frame, MSL_ROVER, and defined to be
   co-aligned with it.

   \begindata

      FRAME_MSL_SAM                   = -76182
      FRAME_-76182_NAME               = 'MSL_SAM'
      FRAME_-76182_CLASS              = 4
      FRAME_-76182_CLASS_ID           = -76182
      FRAME_-76182_CENTER             = -76
      TKFRAME_-76182_SPEC             = 'ANGLES'
      TKFRAME_-76182_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76182_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76182_AXES             = ( 1,   2,   3   )
      TKFRAME_-76182_UNITS            = 'DEGREES'

      FRAME_MSL_CHEMIN                = -76183
      FRAME_-76183_NAME               = 'MSL_CHEMIN'
      FRAME_-76183_CLASS              = 4
      FRAME_-76183_CLASS_ID           = -76183
      FRAME_-76183_CENTER             = -76
      TKFRAME_-76183_SPEC             = 'ANGLES'
      TKFRAME_-76183_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76183_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76183_AXES             = ( 1,   2,   3   )
      TKFRAME_-76183_UNITS            = 'DEGREES'

   \begintext


Remote Sensing Mast (RSM) and RSM-mounted Instrument Frames
========================================================================

   This section defines frames for the RSM instruments and structures
   -- RSM joints and head, Mast Cameras (MASTCAM), Navigation Cameras
   (NAVCAMs), Chemistry & Camera experiment (CHEMCAM), and the
   booms and boom-mounted sensors of the Rover Environmental Monitoring
   Station (REMS).


RSM
-------------------------------------------------

   Five frames are defined for the RSM gimbals and head --
   MSL_RSM_ZERO_AZ, MSL_RSM_AZ, MSL_RSM_ZERO_EL, MSL_RSM_EL, and
   MSL_RSM_HEAD.
 
   The RSM "zero" AZ frame, MSL_RSM_ZERO_AZ, is defined to establish
   the AZ gimbal "zero" position with its +Z axis being the AZ rotation
   axis and its +X axis pointing toward the "zero" AZ hard stop (RSM
   looking backward). It is defined as a fixed-offset frame relative to
   the rover frame and is nominally rotated by -181 degrees about Z
   from it.
 
   The RSM AZ frame, MSL_RSM_AZ, is rotated from the MSL_RSM_ZERO_AZ
   frame by the AZ angle about Z. It is defined as a CK-based frame.
 
   This diagram shows the two AZ frames in the nominal forward-looking
   "home" (181 deg AZ, 91 deg EL) position:

      Rover -Z side ("top") view:
      ---------------------------

                        .-----.       .-----.       .-----.
                        |     | Yaz   |     |       |     |
                        |     |^      |             |     |
                        `--|--'|      `-- Yr        `--|--'
                           `---|------o- ^ -----o------'
                           RSM |    Xaz0 | ------.
                       <-------x-------> |       |--------.
                    Xaz      ||||        |       |-------.|
          RA                 |`| <-------x       |       || RTG 
       |                     | |  Xr         HGA |-------'|
      -o---------|--------|o-| v          =====-o---------'
       |                     `  Yaz0 ------------' 
                           .----------o---------o------.
                        .--|--.       .--|--.       .--|--.
                        |     |       |     |       |     |
                        |     |       |     |       |     |
                        `-----'       `-----'       `-----'

                                           Zr, Zaz0, Zaz are into the page.

   The RSM "zero" EL frame, MSL_RSM_ZERO_EL, is defined to establish
   the EL gimbal "zero" position with its +Y axis being the EL rotation
   axis and its its +X axis pointing toward the "zero" EL hard stop
   (RSM looking down). It is defined as a fixed-offset frame relative
   to the RSM AZ frame and is nominally rotated by -91 degrees about Y
   from it.
 
   The RSM EL frame, MSL_RSM_EL, is rotated from the MSL_RSM_ZERO_EL
   frame by the EL angle about Y. It is defined as a CK-based frame.
 
   The RSM head frame, MSL_RSM_HEAD, is defined as a fixed-offset frame
   relative to the MSL_RSM_EL frame by EL. It is defined to be
   co-aligned with the ROVER frame in the "home" position.
 
   This diagram shows the two EL and the HEAD frames in the nominal
   forward-looking "home" (181 deg AZ, 91 deg EL) position:

      Rover -Y side view:
      -------------------

                     Xel,h     _      Zel0
                       <-------x-------> 
                              `|'
                               |              HGA
                               | Zel,h    .```.         .
                               v Xel0    |  o  ---   .-' \ RTG
          RA                   |          `._.' | .-'     \
      -|-                  |-.-------------------'       .- 
       o---------o--------o| |                   |    .-' 
                             |.-------o----.     |-.-'
                           .-`--------------`-.--' 
                           |             .-----`o------.
                         .-|-.         .-|-.         .-|-.
                        |  o  |       |  o  |       |  o  |
                         `._.'   <-------x.'         `._.'
                                Xr       |
                                         |
                                         |
                                      Zr v
   
                                           Yr, Yel0, Yh are into the page.

   The AZ, EL, and head frames are co-aligned with the rover frame in 
   the nominal forward-looking "home" (181 deg AZ, 91 deg EL) position.

   The nominal nominal offset angles for the fixed offset frames in the
   RSM frame chain are:

      TKFRAME_-76201_ANGLES           = ( 0.0, 0.0, 181.0 )
      TKFRAME_-76201_AXES             = ( 1,   2,     3   )

      TKFRAME_-76203_ANGLES           = ( 0.0, 0.0,  91.0 )
      TKFRAME_-76203_AXES             = ( 1,   3,     2   )

      TKFRAME_-76205_ANGLES           = ( 0.0, 0.0,   0.0 )
      TKFRAME_-76205_AXES             = ( 3    1      2   )

   The actual offset angles were derived from the following AZ and EL
   "home" gimbal positions (in radians) and directions of the AZ and EL
   axes in the rover frame at "zero" AZ position (from [6]):

      home_az           3.167345
      home_el           1.588171
      az_axis_x        -0.001
      az_axis_y         0
      az_axis_z         1
      el_axis_x        -0.022
      el_axis_y        -1
      el_axis_z         0
      el_azimuth        0

   The frame definitions below incorporate the actual offsets.

   \begindata

      FRAME_MSL_RSM_ZERO_AZ           = -76201
      FRAME_-76201_NAME               = 'MSL_RSM_ZERO_AZ'
      FRAME_-76201_CLASS              = 4
      FRAME_-76201_CLASS_ID           = -76201
      FRAME_-76201_CENTER             = -76
      TKFRAME_-76201_SPEC             = 'ANGLES'
      TKFRAME_-76201_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76201_ANGLES           = ( -178.739697,   0.001260,  -0.057282 )
      TKFRAME_-76201_AXES             = (    3,          1,          2        )
      TKFRAME_-76201_UNITS            = 'DEGREES'

      FRAME_MSL_RSM_AZ                = -76202
      FRAME_-76202_NAME               = 'MSL_RSM_AZ'
      FRAME_-76202_CLASS              = 3
      FRAME_-76202_CLASS_ID           = -76202
      FRAME_-76202_CENTER             = -76
      CK_-76202_SCLK                  = -76
      CK_-76202_SPK                   = -76

      FRAME_MSL_RSM_ZERO_EL           = -76203
      FRAME_-76203_NAME               = 'MSL_RSM_ZERO_EL'
      FRAME_-76203_CLASS              = 4
      FRAME_-76203_CLASS_ID           = -76203
      FRAME_-76203_CENTER             = -76
      TKFRAME_-76203_SPEC             = 'ANGLES'
      TKFRAME_-76203_RELATIVE         = 'MSL_RSM_AZ'
      TKFRAME_-76203_ANGLES           = (   -0.000000,  -0.001260,  90.995495 )
      TKFRAME_-76203_AXES             = (    3,          1,          2        )
      TKFRAME_-76203_UNITS            = 'DEGREES'

      FRAME_MSL_RSM_EL                = -76204
      FRAME_-76204_NAME               = 'MSL_RSM_EL'
      FRAME_-76204_CLASS              = 3
      FRAME_-76204_CLASS_ID           = -76204
      FRAME_-76204_CENTER             = -76
      CK_-76204_SCLK                  = -76
      CK_-76204_SPK                   = -76

      FRAME_MSL_RSM_HEAD              = -76205
      FRAME_-76205_NAME               = 'MSL_RSM_HEAD'
      FRAME_-76205_CLASS              = 4
      FRAME_-76205_CLASS_ID           = -76205
      FRAME_-76205_CENTER             = -76
      TKFRAME_-76205_SPEC             = 'ANGLES'
      TKFRAME_-76205_RELATIVE         = 'MSL_RSM_EL'
      TKFRAME_-76205_ANGLES           = (    0.215198,   0.001260,  -0.057291 )
      TKFRAME_-76205_AXES             = (    3,          1,          2        )
      TKFRAME_-76205_UNITS            = 'DEGREES'

   \begintext


MASTCAM
-------------------------------------------------

   The MASTCAM frames are defined as follows:

      -- +Z axis is along the camera's central pixel view direction 
         ("into image");

      -- +Y axis is along the image central column and points from the
         image center toward the image top row ("up");

      -- +X completes the right hand frame and is along the image central
         row and points from the image center toward the image left 
         column ("left");

      -- the origin of the frame is located at the camera focal point.

   The MASTCAM frames for the RSM in nominal forward-looking (181 deg
   AZ, 91 deg EL) position are shown on this diagram:

      Rover -Y side view:
      -------------------
                               ^ Ymc
                               |
                               |
                      Zmc      | 
                       <-------*
                      Xh      `|'
                               |              HGA
                               |          .```.         .
                               v Zh      |  o  ---   .-' \ RTG
          RA                   |          `._.' | .-'     \
      -|-                  |-.-------------------'       .- 
       o---------o--------o| |                   |    .-' 
                             |.-------o----.     |-.-'
                           .-`--------------`-.--' 
                           |             .-----`o------.
                         .-|-.         .-|-.         .-|-.
                        |  o  |       |  o  |       |  o  |
                         `._.'   <-------x.'         `._.'
                                Xr       |
                                         |
                                         |
                                      Zr v
   
                                             Yr, Yh are into the page.
                                             Xmc is out of the page.

   Since MASTCAMs are rigidly mounted on RSM, their frames are defined
   as fixed offset frames with orientation given with respect to the
   RSM head frame.
 
   The nominal MASTCAM frame orientation are such that the MASTCAM left
   and right boresights are "toe"ed in by 1.25 degrees toward the RSM
   head +X axis. In order to align the RSM head frame with the camera
   frames in this nominal orientation, it has to be rotated by +90
   degrees about Y and then by "toe"-in angles (-1.25 degrees for the
   left camera and +1.25 degrees for the right camera) about X, and
   finally by -90 degrees about Z (to line up Y axis with the vertical
   direction.) The following sets of keywords can be incorporated into
   the frame definitions to provide this nominal orientation:

      TKFRAME_-76210_ANGLES           = ( -90.000,   1.250,  90.000 )
      TKFRAME_-76210_AXES             = (   2,       1,       3     )

      TKFRAME_-76220_ANGLES           = ( -90.000,  -1.250,  90.000 )
      TKFRAME_-76220_AXES             = (   2,       1,       3     )

   The actual MSL_MASTCAM_LEFT frame orientation provided in the frame
   definition below was computed using the CAHVOR(E) camera model file,
   'MSL_CAL_003_SN_3003_FILTER_0_FOCUS_02315-MCAML-FLIGHT.cahvor'.
   According to this model the reference frame, MSL_RSM_HEAD, can be
   transformed into the camera frame, MSL_MASTCAM_LEFT, by the
   following sequence of rotations: first by 90.00997282 degrees about
   Y, then by -1.48435653 degrees about X, and finally by -89.65469934
   degrees about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_MASTCAM_LEFT           = -76210
      FRAME_-76210_NAME               = 'MSL_MASTCAM_LEFT'
      FRAME_-76210_CLASS              = 4
      FRAME_-76210_CLASS_ID           = -76210
      FRAME_-76210_CENTER             = -76
      TKFRAME_-76210_RELATIVE         = 'MSL_RSM_HEAD'
      TKFRAME_-76210_SPEC             = 'ANGLES'
      TKFRAME_-76210_UNITS            = 'DEGREES'
      TKFRAME_-76210_AXES             = (    2,        1,        3     )
      TKFRAME_-76210_ANGLES           = (  -90.010,    1.484,   89.655 )

   \begintext


   The actual MSL_MASTCAM_RIGHT frame orientation provided in the frame
   definition below was computed using the CAHVOR(E) camera model file,
   'MSL_CAL_003_SN_3004_FILTER_0_FOCUS_02702-MCAMR-FLIGHT.cahvor'.
   According to this model the reference frame, MSL_RSM_HEAD, can be
   transformed into the camera frame, MSL_MASTCAM_RIGHT, by the
   following sequence of rotations: first by 89.99071558 degrees about
   Y, then by 1.33167368 degrees about X, and finally by -90.43555383
   degrees about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_MASTCAM_RIGHT          = -76220
      FRAME_-76220_NAME               = 'MSL_MASTCAM_RIGHT'
      FRAME_-76220_CLASS              = 4
      FRAME_-76220_CLASS_ID           = -76220
      FRAME_-76220_CENTER             = -76
      TKFRAME_-76220_RELATIVE         = 'MSL_RSM_HEAD'
      TKFRAME_-76220_SPEC             = 'ANGLES'
      TKFRAME_-76220_UNITS            = 'DEGREES'
      TKFRAME_-76220_AXES             = (    2,        1,        3     )
      TKFRAME_-76220_ANGLES           = (  -89.991,   -1.332,   90.436 )

   \begintext



NAVCAMs
-------------------------------------------------

   The NAVCAM frames are defined as follows:

      -- +Z axis is along the camera's central pixel view direction 
         ("into image");

      -- +Y axis is along the image central column and points from the
         image center toward the image top row ("up");

      -- +X completes the right hand frame and is along the image central
         row and points from the image center toward the image left 
         column ("left");

      -- the origin of the frame is located at the camera focal point.

   The NAVCAM frames for the RSM in nominal forward-looking (181 deg
   AZ, 91 deg EL) position are shown on this diagram:

      Rover -Y side view:
      -------------------
                               ^ Ync
                               |
                               |
                      Znc      | 
                       <-------*
                      Xh      `|'
                               |              HGA
                               |          .```.         .
                               v Zh      |  o  ---   .-' \ RTG
          RA                   |          `._.' | .-'     \
      -|-                  |-.-------------------'       .- 
       o---------o--------o| |                   |    .-' 
                             |.-------o----.     |-.-'
                           .-`--------------`-.--' 
                           |             .-----`o------.
                         .-|-.         .-|-.         .-|-.
                        |  o  |       |  o  |       |  o  |
                         `._.'   <-------x.'         `._.'
                                Xr       |
                                         |
                                         |
                                      Zr v
   
                                             Yr, Yh are into the page.
                                             Xnc is out of the page.

   Since NAVCAMs are rigidly mounted on RSM, their frames are defined
   as fixed offset frames with orientation given with respect to the
   RSM head frame.
 
   The nominal NAVCAM frame orientation are such that the NAVCAM left
   and right boresights are co-aligned with the RSM head +X axis. In
   order to align the RSM head frame with the camera frames in this
   nominal orientation, it has to be rotated by +90 degrees about Y,
   then by -90 degrees about Z (to line up Y axis with the vertical
   direction.) The following sets of keywords can be incorporated into
   the frame definitions to provide this nominal orientation:

      TKFRAME_-76231_ANGLES           = ( -90.000,   0.000,  90.000 )
      TKFRAME_-76231_AXES             = (   2,       1,       3     )

      TKFRAME_-76232_ANGLES           = ( -90.000,   0.000,  90.000 )
      TKFRAME_-76232_AXES             = (   2,       1,       3     )

      TKFRAME_-76233_ANGLES           = ( -90.000,   0.000,  90.000 )
      TKFRAME_-76233_AXES             = (   2,       1,       3     )

      TKFRAME_-76234_ANGLES           = ( -90.000,   0.000,  90.000 )
      TKFRAME_-76234_AXES             = (   2,       1,       3     )

   The actual MSL_NAVCAM_LEFT_A frame orientation provided in the frame
   definition below was computed using the CAHVOR(E) camera model file,
   'MSL_CAL_004_SN_0216-NAVL-FLIGHT-RCE-A.cahvor'. According to this
   model the reference frame, MSL_RSM_HEAD, can be transformed into the
   camera frame, MSL_NAVCAM_LEFT_A, by the following sequence of
   rotations: first by 89.51443329 degrees about Y, then by -1.12581998
   degrees about X, and finally by -90.38864561 degrees about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_NAVCAM_LEFT_A          = -76231
      FRAME_-76231_NAME               = 'MSL_NAVCAM_LEFT_A'
      FRAME_-76231_CLASS              = 4
      FRAME_-76231_CLASS_ID           = -76231
      FRAME_-76231_CENTER             = -76
      TKFRAME_-76231_RELATIVE         = 'MSL_RSM_HEAD'
      TKFRAME_-76231_SPEC             = 'ANGLES'
      TKFRAME_-76231_UNITS            = 'DEGREES'
      TKFRAME_-76231_AXES             = (    2,        1,        3     )
      TKFRAME_-76231_ANGLES           = (  -89.514,    1.126,   90.389 )

   \begintext


   The actual MSL_NAVCAM_LEFT_B frame orientation provided in the frame
   definition below was computed using the CAHVOR(E) camera model file,
   'MSL_CAL_005_SN_0215-NAVL-FLIGHT-RCE-B.cahvor'. According to this
   model the reference frame, MSL_RSM_HEAD, can be transformed into the
   camera frame, MSL_NAVCAM_LEFT_B, by the following sequence of
   rotations: first by 89.86145547 degrees about Y, then by -0.38921323
   degrees about X, and finally by -90.02355994 degrees about Z.

   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_NAVCAM_LEFT_B          = -76233
      FRAME_-76233_NAME               = 'MSL_NAVCAM_LEFT_B'
      FRAME_-76233_CLASS              = 4
      FRAME_-76233_CLASS_ID           = -76233
      FRAME_-76233_CENTER             = -76
      TKFRAME_-76233_RELATIVE         = 'MSL_RSM_HEAD'
      TKFRAME_-76233_SPEC             = 'ANGLES'
      TKFRAME_-76233_UNITS            = 'DEGREES'
      TKFRAME_-76233_AXES             = (    2,        1,        3     )
      TKFRAME_-76233_ANGLES           = (  -89.861,    0.389,   90.024 )

   \begintext


   The actual MSL_NAVCAM_RIGHT_A frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_004_SN_0206-NAVR-FLIGHT-RCE-A.cahvor'. According to
   this model the reference frame, MSL_RSM_HEAD, can be transformed
   into the camera frame, MSL_NAVCAM_RIGHT_A, by the following sequence
   of rotations: first by 89.24730809 degrees about Y, then by
   -0.53383267 degrees about X, and finally by -89.81089093 degrees
   about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.
 
   \begindata

      FRAME_MSL_NAVCAM_RIGHT_A         = -76232
      FRAME_-76232_NAME               = 'MSL_NAVCAM_RIGHT_A'
      FRAME_-76232_CLASS              = 4
      FRAME_-76232_CLASS_ID           = -76232
      FRAME_-76232_CENTER             = -76
      TKFRAME_-76232_RELATIVE         = 'MSL_RSM_HEAD'
      TKFRAME_-76232_SPEC             = 'ANGLES'
      TKFRAME_-76232_UNITS            = 'DEGREES'
      TKFRAME_-76232_AXES             = (    2,        1,        3     )
      TKFRAME_-76232_ANGLES           = (  -89.247,    0.534,   89.811 )

   \begintext


   The actual MSL_NAVCAM_RIGHT_B frame orientation provided in the
   frame definition below was computed using the CAHVOR(E) camera model
   file, 'MSL_CAL_005_SN_0218-NAVR-FLIGHT-RCE-B.cahvor'. According to
   this model the reference frame, MSL_RSM_HEAD, can be transformed
   into the camera frame, MSL_NAVCAM_RIGHT_B, by the following sequence
   of rotations: first by 89.91279432 degrees about Y, then by
   -0.33489566 degrees about X, and finally by -90.29024820 degrees
   about Z.

   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_NAVCAM_RIGHT_B         = -76234
      FRAME_-76234_NAME               = 'MSL_NAVCAM_RIGHT_B'
      FRAME_-76234_CLASS              = 4
      FRAME_-76234_CLASS_ID           = -76234
      FRAME_-76234_CENTER             = -76
      TKFRAME_-76234_RELATIVE         = 'MSL_RSM_HEAD'
      TKFRAME_-76234_SPEC             = 'ANGLES'
      TKFRAME_-76234_UNITS            = 'DEGREES'
      TKFRAME_-76234_AXES             = (    2,        1,        3     )
      TKFRAME_-76234_ANGLES           = (  -89.913,    0.335,   90.290 )

   \begintext


CHEMCAM
-------------------------------------------------

   The CHEMCAM RMI frame is defined as follows:

      -- +Z axis is along the camera's central pixel view direction 
         ("into image");

      -- +Y axis is along the image central column and points from the
         image center toward the image top row ("up");

      -- +X completes the right hand frame and is along the image central
         row and points from the image center toward the image left 
         column ("left");

      -- the origin of the frame is located at the camera focal point.

   The CHEMCAM LIBS laser and camera frames are defined to be nominally
   co-aligned with the CHEMCAM RMI frame.

   The CHEMCAM frames for the RSM in nominal forward-looking (181 deg
   AZ, 91 deg EL) position are shown on this diagram:

      Rover -Y side view:
      -------------------
                               ^ Ycc
                               |
                               |
                      Zcc      | 
                       <-------*
                      Xh      `|'
                               |              HGA
                               |          .```.         .
                               v Zh      |  o  ---   .-' \ RTG
          RA                   |          `._.' | .-'     \
      -|-                  |-.-------------------'       .- 
       o---------o--------o| |                   |    .-' 
                             |.-------o----.     |-.-'
                           .-`--------------`-.--' 
                           |             .-----`o------.
                         .-|-.         .-|-.         .-|-.
                        |  o  |       |  o  |       |  o  |
                         `._.'   <-------x.'         `._.'
                                Xr       |
                                         |
                                         |
                                      Zr v
   
                                             Yr, Yh are into the page.
                                             Xcc is out of the page.

   Since CHEMCAM is rigidly mounted on RSM, its frames are defined as
   fixed offset frames with orientation given with respect to the RSM
   head frame.

   The nominal CHEMCAM RMI orientation is such that the CHEMCAM RMI
   boresight is co-aligned with the RSM head +X axis. In order to align
   the RSM head frame with the RMI frame in this nominal orientation,
   it has to be rotated by +90 degrees about Y, then by -90 degrees
   about Z (to line up Y axis with the vertical direction.) The
   following sets of keywords should be included into the frame
   definitions to provide this nominal orientation:

      TKFRAME_-76243_ANGLES           = ( -90.000,   0.000,  90.000 )
      TKFRAME_-76243_AXES             = (   2,       1,       3     )

   The actual MSL_CHEMCAM_RMI frame orientation provided in the frame
   definition below was computed using the CAHVOR(E) camera model file,
   'MSL_CAL_003_SN_0001_FOCUS_12658-RMI-FLIGHT.cahv'. According to this
   model the reference frame, MSL_RSM_HEAD, can be transformed into the
   camera frame, MSL_CHEMCAM_RMI, by the following sequence of
   rotations: first by 89.81378695 degrees about Y, then by -0.21898509
   degrees about X, and finally by -90.08397425 degrees about Z.

   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define
   rotations from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_CHEMCAM_RMI            = -76243
      FRAME_-76243_NAME               = 'MSL_CHEMCAM_RMI'
      FRAME_-76243_CLASS              = 4
      FRAME_-76243_CLASS_ID           = -76243
      FRAME_-76243_CENTER             = -76
      TKFRAME_-76243_RELATIVE         = 'MSL_RSM_HEAD'
      TKFRAME_-76243_SPEC             = 'ANGLES'
      TKFRAME_-76243_UNITS            = 'DEGREES'
      TKFRAME_-76243_AXES             = (    2,        1,        3     )
      TKFRAME_-76243_ANGLES           = (  -89.814,    0.219,   90.084 )

   \begintext

   Based on the pre-launch calibrations the LIBS laser spot points along 
   the RMI pixel (496,531) view direction as shown on this diagram:

                                    ^ Yrmi
                    (0,0            |           (0,1023)
                      o-------------|-------------o
                      |             |             |
                      |             |             |
                      |             |             |
                      |             |             |
                      |             | (496,531)   |
                      |             | x           |
                   <----------------x             |
             Zrmi     |       (511.5,511.5)       |
                      |                           |
                      |                           |
                      |                           |
                      |                           |
                      |                           |
                      o---------------------------o
                 (1023,0)                    (1023,1023)

                                       Zrmi and Zlibs are into the page.


   Assuming the nominal RMI pixel IFOV of 20 microradians, two
   rotations needed to coalign the RMI frame with the LIBS laser frame
   are first by -0.00039 radians about Y, then by -0.00031 radians
   about X. This orientation is provided in the LIBS laser frame
   definition below, containing the opposite of this transformation
   because Euler angles specified in it define rotations from the
   "destination" frame to the "reference" frame.
   
   The CHEMCAM LIBS camera frame is defined to be co-aligned with the
   LIBS laser frame.

   \begindata

      FRAME_MSL_CHEMCAM_LIBS_LASER    = -76241
      FRAME_-76241_NAME               = 'MSL_CHEMCAM_LIBS_LASER'
      FRAME_-76241_CLASS              = 4
      FRAME_-76241_CLASS_ID           = -76241
      FRAME_-76241_CENTER             = -76
      TKFRAME_-76241_SPEC             = 'ANGLES'
      TKFRAME_-76241_RELATIVE         = 'MSL_CHEMCAM_RMI'
      TKFRAME_-76241_ANGLES           = ( 0.00039, 0.00031,   0.000 )
      TKFRAME_-76241_AXES             = ( 2,       1,         3     )
      TKFRAME_-76241_UNITS            = 'RADIANS'

      FRAME_MSL_CHEMCAM_LIBS_CAM      = -76242
      FRAME_-76242_NAME               = 'MSL_CHEMCAM_LIBS_CAM'
      FRAME_-76242_CLASS              = 4
      FRAME_-76242_CLASS_ID           = -76242
      FRAME_-76242_CENTER             = -76
      TKFRAME_-76242_SPEC             = 'ANGLES'
      TKFRAME_-76242_RELATIVE         = 'MSL_CHEMCAM_LIBS_LASER'
      TKFRAME_-76242_ANGLES           = (   0.000,   0.000,   0.000 )
      TKFRAME_-76242_AXES             = (   2,       1,       3     )
      TKFRAME_-76242_UNITS            = 'DEGREES'

   \begintext


REMS
-------------------------------------------------

   REMS boom frames -- MSL_REMS_BOOM1 and MSL_REMS_BOOM1 -- are defined 
   to have X axes along booms and Z axes nominally co-aligned with
   the rover Z axis.

   The REMS BOOM 1 frame is nominally rotated from the rover frame by
   122 degrees about Z while the REMS BOOM 2 frame is nominally rotated
   from the rover frame by 2 degrees about Z.

   This diagram illustrates the REMS boom frames:

      Rover -Z side ("top") view:
      ---------------------------

                        .-----.  Xb2  .-----.       .-----.
                        | Yb1       ^ |     |       |     |
                        |    ^     /  |             |     |
                        `--  |    /   `-- Yr        `--|--'
                             | --/----o- ^ -----o------'
                             |.-x------- | ------.
                     <-------x| |`-. Yb2 |       |--------.
                  Xb1        || |   `>   |       |-------.|
          RA             RSM |`- <-------x       |       || RTG 
       |                     |  Xr           HGA |-------'|
      -o---------|--------|o-|            =====-o---------'
       |                     `-------------------' 
                           .----------o---------o------.
                        .--|--.       .--|--.       .--|--.
                        |     |       |     |       |     |
                        |     |       |     |       |     |
                        `-----'       `-----'       `-----'

                                          Zr, Zb1, Zb2 is into the page.

   REMS sensor frames are defined to be co aligned with the their
   corresponding boom frames.

   \begindata

      FRAME_MSL_REMS_BOOM1            = -76250
      FRAME_-76250_NAME               = 'MSL_REMS_BOOM1'
      FRAME_-76250_CLASS              = 4
      FRAME_-76250_CLASS_ID           = -76250
      FRAME_-76250_CENTER             = -76
      TKFRAME_-76250_SPEC             = 'ANGLES'
      TKFRAME_-76250_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76250_ANGLES           = ( 0.0, 0.0, -122.0 )
      TKFRAME_-76250_AXES             = ( 1,   2,      3   )
      TKFRAME_-76250_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM1_WS1        = -76251
      FRAME_-76251_NAME               = 'MSL_REMS_BOOM1_WS1'
      FRAME_-76251_CLASS              = 4
      FRAME_-76251_CLASS_ID           = -76251
      FRAME_-76251_CENTER             = -76
      TKFRAME_-76251_SPEC             = 'ANGLES'
      TKFRAME_-76251_RELATIVE         = 'MSL_REMS_BOOM1'
      TKFRAME_-76251_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76251_AXES             = ( 1,   2,   3   )
      TKFRAME_-76251_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM1_WS2        = -76252
      FRAME_-76252_NAME               = 'MSL_REMS_BOOM1_WS2'
      FRAME_-76252_CLASS              = 4
      FRAME_-76252_CLASS_ID           = -76252
      FRAME_-76252_CENTER             = -76
      TKFRAME_-76252_SPEC             = 'ANGLES'
      TKFRAME_-76252_RELATIVE         = 'MSL_REMS_BOOM1'
      TKFRAME_-76252_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76252_AXES             = ( 1,   2,   3   )
      TKFRAME_-76252_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM1_WS3        = -76253
      FRAME_-76253_NAME               = 'MSL_REMS_BOOM1_WS3'
      FRAME_-76253_CLASS              = 4
      FRAME_-76253_CLASS_ID           = -76253
      FRAME_-76253_CENTER             = -76
      TKFRAME_-76253_SPEC             = 'ANGLES'
      TKFRAME_-76253_RELATIVE         = 'MSL_REMS_BOOM1'
      TKFRAME_-76253_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76253_AXES             = ( 1,   2,   3   )
      TKFRAME_-76253_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM1_ATS        = -76254
      FRAME_-76254_NAME               = 'MSL_REMS_BOOM1_ATS'
      FRAME_-76254_CLASS              = 4
      FRAME_-76254_CLASS_ID           = -76254
      FRAME_-76254_CENTER             = -76
      TKFRAME_-76254_SPEC             = 'ANGLES'
      TKFRAME_-76254_RELATIVE         = 'MSL_REMS_BOOM1'
      TKFRAME_-76254_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76254_AXES             = ( 1,   2,   3   )
      TKFRAME_-76254_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM1_GTS        = -76255
      FRAME_-76255_NAME               = 'MSL_REMS_BOOM1_GTS'
      FRAME_-76255_CLASS              = 4
      FRAME_-76255_CLASS_ID           = -76255
      FRAME_-76255_CENTER             = -76
      TKFRAME_-76255_SPEC             = 'ANGLES'
      TKFRAME_-76255_RELATIVE         = 'MSL_REMS_BOOM1'
      TKFRAME_-76255_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76255_AXES             = ( 1,   2,   3   )
      TKFRAME_-76255_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM2            = -76260
      FRAME_-76260_NAME               = 'MSL_REMS_BOOM2'
      FRAME_-76260_CLASS              = 4
      FRAME_-76260_CLASS_ID           = -76260
      FRAME_-76260_CENTER             = -76
      TKFRAME_-76260_SPEC             = 'ANGLES'
      TKFRAME_-76260_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76260_ANGLES           = ( 0.0, 0.0, -2.0 )
      TKFRAME_-76260_AXES             = ( 1,   2,    3   )
      TKFRAME_-76260_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM2_WS1        = -76261
      FRAME_-76261_NAME               = 'MSL_REMS_BOOM2_WS1'
      FRAME_-76261_CLASS              = 4
      FRAME_-76261_CLASS_ID           = -76261
      FRAME_-76261_CENTER             = -76
      TKFRAME_-76261_SPEC             = 'ANGLES'
      TKFRAME_-76261_RELATIVE         = 'MSL_REMS_BOOM2'
      TKFRAME_-76261_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76261_AXES             = ( 1,   2,   3   )
      TKFRAME_-76261_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM2_WS2        = -76262
      FRAME_-76262_NAME               = 'MSL_REMS_BOOM2_WS2'
      FRAME_-76262_CLASS              = 4
      FRAME_-76262_CLASS_ID           = -76262
      FRAME_-76262_CENTER             = -76
      TKFRAME_-76262_SPEC             = 'ANGLES'
      TKFRAME_-76262_RELATIVE         = 'MSL_REMS_BOOM2'
      TKFRAME_-76262_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76262_AXES             = ( 1,   2,   3   )
      TKFRAME_-76262_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM2_WS3        = -76263
      FRAME_-76263_NAME               = 'MSL_REMS_BOOM2_WS3'
      FRAME_-76263_CLASS              = 4
      FRAME_-76263_CLASS_ID           = -76263
      FRAME_-76263_CENTER             = -76
      TKFRAME_-76263_SPEC             = 'ANGLES'
      TKFRAME_-76263_RELATIVE         = 'MSL_REMS_BOOM2'
      TKFRAME_-76263_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76263_AXES             = ( 1,   2,   3   )
      TKFRAME_-76263_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM2_ATS        = -76264
      FRAME_-76264_NAME               = 'MSL_REMS_BOOM2_ATS'
      FRAME_-76264_CLASS              = 4
      FRAME_-76264_CLASS_ID           = -76264
      FRAME_-76264_CENTER             = -76
      TKFRAME_-76264_SPEC             = 'ANGLES'
      TKFRAME_-76264_RELATIVE         = 'MSL_REMS_BOOM2'
      TKFRAME_-76264_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76264_AXES             = ( 1,   2,   3   )
      TKFRAME_-76264_UNITS            = 'DEGREES'

      FRAME_MSL_REMS_BOOM2_HS         = -76265
      FRAME_-76265_NAME               = 'MSL_REMS_BOOM2_HS'
      FRAME_-76265_CLASS              = 4
      FRAME_-76265_CLASS_ID           = -76265
      FRAME_-76265_CENTER             = -76
      TKFRAME_-76265_SPEC             = 'ANGLES'
      TKFRAME_-76265_RELATIVE         = 'MSL_REMS_BOOM2'
      TKFRAME_-76265_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76265_AXES             = ( 1,   2,   3   )
      TKFRAME_-76265_UNITS            = 'DEGREES'

   \begintext


Robot Arm (RA) and RA-mounted Instrument Frames
========================================================================

   This section defines frames for the RA instruments and structures --
   RA joints and turret head, Mars Hand Lens Imager (MAHLI), Alpha
   Particle X-ray Spectrometer (APXS), Powder Acquisition Drill System
   (PADS), Dust Removal Tool (DRT), and Collection and Handling for
   Interior Martian Rock Analysis (CHIMRA).


RA Joints
-------------------------------------------------

   MSL RA base frame, MSL_RA_BASE, is defined to be coincident in
   orientation with the rover frame. As this frame is fixed with
   respect to the rover it's defined as zero offset frame with respect
   to the rover frame.
 
   All MSL RA joint frames -- SHOULDER_AZ, SHOULDER_EL, ELBOW, WRIST,
   and TURRET -- are defined in accordance with normal kinematics
   convention as follows:

      -- +Z axis is along the joint rotation axis, nominally pointing
         along rover +Z for SHOULDER_AZ, along rover -Y (for RA in 
         straight out position) for SHOULDER_EL, ELBOW, and WRIST, and
         along wrist +Y for TURRET;

      -- +X axis is along the link attached to the joint;

      -- +Y completes the right hand frame;

      -- the origin lies on the rotation axis at a point that provides for 
         the minimum magnitude of translations between the joints
         (see diagram above).

   This diagram illustrates RA base and joint frames in "zero joint
   rotation" position:


      RA side view:
      -------------

              Shoulder/Az                                Turret
                 axis                                     axis

                   |                                    |
                   .                                    .
                   |                                    |

                                                                     
              //////// Xbase                          .___.          
             //o------>                               | o |          
             //|//////                                | o------>     
              /|///////                               ._|_.    Xtr   
               | |   | .-.Xsh_az         .-.            |\   .       
         Ybase V | o----x->----> =======| x------> =====|=| x------> 
                 ._|_. `|   Xsh_el       `|'    Xelb    V  `|'     Xwr
                   |    |                 |            Ztr  |       
                   |    |                 |                 |
         o------>  V    V Ysh_el          V Yelb            V Ywr
         |     Xr   Zsh_az
         |
         |
         V                          Yr, Ybase, Ysh_az, Ytr are out of page;
          Zr                             Zsh_el, Zelb, Zwr are into page.


      RA top view:
      ------------

                    Shoulder/El         Elbow             Wrist
                       axis             axis              axis

                        |                 |                 |
                        .                 .                 .
                        |                 |                 |
                        
               //////    Zsh_el            Zelb              Zwr
             /Rover//   ^                 ^                 ^
             ////////  .|.               .|.                |
             ////Xbase |||               |||                |       
             //x----->| |    Xsh_el     | | |  Xelb       ._|_.  Xwr
             //| | x----x->---->        | x------> -------| x------>
             //|//`|'|__ Xsh_az        |__|__|------------|   |     
             //|///|/  \   `-----------'   /              '---'     
             //V///|/   `-----------------'       APXS .-'. /  Xtr  
            Ybase//V/ Ysh_az                          | x------>    
                /////                            MAHLI `|'          
                                                        |           
                                                        |
         x------> Xr                                    V Ytr
         |
         |
         |                                    Zr, Zbase, Zsh_az, Ysh_el, 
         V Yr                                 Yelb, Ywr, and Ztr are all
                                                      into the page
                                

   During normal surface operations the orientation of each of these
   frames with respect to each other varies and is controlled and
   telemetered using RA joint angles. Therefore, these frame are
   defined as a CK frames with the orientation for each frame provided
   with respect to its parent in the frame chain.

   \begindata

      FRAME_MSL_RA_BASE               = -76300
      FRAME_-76300_NAME               = 'MSL_RA_BASE'
      FRAME_-76300_CLASS              = 4
      FRAME_-76300_CLASS_ID           = -76300
      FRAME_-76300_CENTER             = -76
      TKFRAME_-76300_SPEC             = 'ANGLES'
      TKFRAME_-76300_RELATIVE         = 'MSL_ROVER'
      TKFRAME_-76300_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-76300_AXES             = ( 1,   2,   3   )
      TKFRAME_-76300_UNITS            = 'DEGREES'

      FRAME_MSL_RA_SHOULDER_AZ        = -76301
      FRAME_-76301_NAME               = 'MSL_RA_SHOULDER_AZ'
      FRAME_-76301_CLASS              = 3
      FRAME_-76301_CLASS_ID           = -76301
      FRAME_-76301_CENTER             = -76
      CK_-76301_SCLK                  = -76
      CK_-76301_SPK                   = -76

      FRAME_MSL_RA_SHOULDER_EL        = -76302
      FRAME_-76302_NAME               = 'MSL_RA_SHOULDER_EL'
      FRAME_-76302_CLASS              = 3
      FRAME_-76302_CLASS_ID           = -76302
      FRAME_-76302_CENTER             = -76
      CK_-76302_SCLK                  = -76
      CK_-76302_SPK                   = -76

      FRAME_MSL_RA_ELBOW              = -76303
      FRAME_-76303_NAME               = 'MSL_RA_ELBOW'
      FRAME_-76303_CLASS              = 3
      FRAME_-76303_CLASS_ID           = -76303
      FRAME_-76303_CENTER             = -76
      CK_-76303_SCLK                  = -76
      CK_-76303_SPK                   = -76

      FRAME_MSL_RA_WRIST              = -76304
      FRAME_-76304_NAME               = 'MSL_RA_WRIST'
      FRAME_-76304_CLASS              = 3
      FRAME_-76304_CLASS_ID           = -76304
      FRAME_-76304_CENTER             = -76
      CK_-76304_SCLK                  = -76
      CK_-76304_SPK                   = -76

      FRAME_MSL_RA_TURRET             = -76305
      FRAME_-76305_NAME               = 'MSL_RA_TURRET'
      FRAME_-76305_CLASS              = 3
      FRAME_-76305_CLASS_ID           = -76305
      FRAME_-76305_CENTER             = -76
      CK_-76305_SCLK                  = -76
      CK_-76305_SPK                   = -76

   \begintext


RA Instruments
-------------------------------------------------

   Two frames are defined for each of the MSL RA instruments -- MAHLI,
   APXS, PADS, DRT, and CHIMRA.
 
   The first frame for each instrument, named ``MSL_<INST>_REF'', is
   the reference frame consistent with the FSW frame for which
   orientation is provided in the RA telemetry. The X axis of this
   frame is along the instrument's principal direction (boresight,
   central axis, etc) and the Z axis is nominally co-aligned with the Z
   axis of the TURRET frame.
 
   Although the ``_REF'' frames are rotated from the TURRET frame about
   Z axis by the following fixed angles:
 
      Instrument     Angle, deg [4]     Angle, rad (deg           ) [5]
      -------------  --------------     --------------------------------
      MAHLI_REF       125.00             2.1380283 ( 122.50       )
      APXS_REF       -133.00             3.9618974 (-133.00       )
      PADS_REF          0.00             0.0       (   0.00       )
      DRT_REF         +69.67             1.22612   ( +70.25150118 )
      CHIMRA_REF      -60.00 (TBD)       5.20867   ( -61.56519212 )

   They are defined as CK-based frames to allow storing their
   orientation either relative to the TURRET frame (based on the angles
   above) or relative to the ROVER frame (based on the orientation
   quaternions provided in telemetry).
 
   The second frame for each instrument, named ``MSL_<INST>'', is the
   reference frame that has the Z axis along the instrument's principal
   direction (boresight, central axis, etc) and Y axis nominally
   co-aligned with the Y axis of the ``_REF'' frame.
 
   These frames are defined as fixed-offset frames relative to the
   corresponding ``_REF'' frames and are nominally rotated from them by
   +90 degrees about Y.

   This diagram illustrates RA turret head and instrument frames (looking 
   in the direction of Ztr axis):
                                                   
                  Zapxs        Yapxs              ^ Zchimra
                  Xapxs_ref    Yapxs_ref         /  Xchimra_ref
                       ^.     ^                 /
                         `. .'                /*.
                           * `.    .-------. / / `-> 
                          `.   `.  |        `./     Ychimra
                            `.   `.---.       `.    Ychimra_ref
                        .------..'     `.      |====
                        |      /         \----.        
                        |      |    x------>  |--*----> 
                        |      \    |    Xtr  '  |     Zpads
                        `------ `.  |  . \     |=|==   Xpads_ref
                               /  `-|-'   \----' v
               Ymahli         /   / V  \   \      Ypads
               Ymahli_ref <-./   / Ytr  \   \     Ypads_ref
                             `*./        \.*'
                             /         <-'  \
                            /     Ydrt       \
                Zmahli     V      Ydrt_ref    V Zdrt
                Xmahli_ref                      Xdrt_ref


                                     Ztr is into the page.
                                     Z*ref are into the page.
                                     X* are out of the page

   The RA instrument frames except for MAHLI are defined below. The
   MAHLI frame definitions with additional comments are provided after
   the block below.

   \begindata

      FRAME_MSL_APXS_REF              = -76320
      FRAME_-76320_NAME               = 'MSL_APXS_REF'
      FRAME_-76320_CLASS              = 3
      FRAME_-76320_CLASS_ID           = -76320
      FRAME_-76320_CENTER             = -76
      CK_-76320_SCLK                  = -76
      CK_-76320_SPK                   = -76

      FRAME_MSL_APXS                  = -76321
      FRAME_-76321_NAME               = 'MSL_APXS'
      FRAME_-76321_CLASS              = 4
      FRAME_-76321_CLASS_ID           = -76321
      FRAME_-76321_CENTER             = -76
      TKFRAME_-76321_SPEC             = 'ANGLES'
      TKFRAME_-76321_RELATIVE         = 'MSL_APXS_REF'
      TKFRAME_-76321_ANGLES           = ( -90.0, 0.0, 0.0 )
      TKFRAME_-76321_AXES             = (   2,   1,   3   )
      TKFRAME_-76321_UNITS            = 'DEGREES'

      FRAME_MSL_PADS_REF              = -76330
      FRAME_-76330_NAME               = 'MSL_PADS_REF'
      FRAME_-76330_CLASS              = 3
      FRAME_-76330_CLASS_ID           = -76330
      FRAME_-76330_CENTER             = -76
      CK_-76330_SCLK                  = -76
      CK_-76330_SPK                   = -76

      FRAME_MSL_PADS                  = -76331
      FRAME_-76331_NAME               = 'MSL_PADS'
      FRAME_-76331_CLASS              = 4
      FRAME_-76331_CLASS_ID           = -76331
      FRAME_-76331_CENTER             = -76
      TKFRAME_-76331_SPEC             = 'ANGLES'
      TKFRAME_-76331_RELATIVE         = 'MSL_PADS_REF'
      TKFRAME_-76331_ANGLES           = ( -90.0, 0.0, 0.0 )
      TKFRAME_-76331_AXES             = (   2,   1,   3   )
      TKFRAME_-76331_UNITS            = 'DEGREES'

      FRAME_MSL_DRT_REF               = -76340
      FRAME_-76340_NAME               = 'MSL_DRT_REF'
      FRAME_-76340_CLASS              = 3
      FRAME_-76340_CLASS_ID           = -76340
      FRAME_-76340_CENTER             = -76
      CK_-76340_SCLK                  = -76
      CK_-76340_SPK                   = -76

      FRAME_MSL_DRT                   = -76341
      FRAME_-76341_NAME               = 'MSL_DRT'
      FRAME_-76341_CLASS              = 4
      FRAME_-76341_CLASS_ID           = -76341
      FRAME_-76341_CENTER             = -76
      TKFRAME_-76341_SPEC             = 'ANGLES'
      TKFRAME_-76341_RELATIVE         = 'MSL_DRT_REF'
      TKFRAME_-76341_ANGLES           = ( -90.0, 0.0, 0.0 )
      TKFRAME_-76341_AXES             = (   2,   1,   3   )
      TKFRAME_-76341_UNITS            = 'DEGREES'

      FRAME_MSL_CHIMRA_REF            = -76350
      FRAME_-76350_NAME               = 'MSL_CHIMRA_REF'
      FRAME_-76350_CLASS              = 3
      FRAME_-76350_CLASS_ID           = -76350
      FRAME_-76350_CENTER             = -76
      CK_-76350_SCLK                  = -76
      CK_-76350_SPK                   = -76

      FRAME_MSL_CHIMRA                = -76351
      FRAME_-76351_NAME               = 'MSL_CHIMRA'
      FRAME_-76351_CLASS              = 4
      FRAME_-76351_CLASS_ID           = -76351
      FRAME_-76351_CENTER             = -76
      TKFRAME_-76351_SPEC             = 'ANGLES'
      TKFRAME_-76351_RELATIVE         = 'MSL_CHIMRA_REF'
      TKFRAME_-76351_ANGLES           = ( -90.0, 0.0, 0.0 )
      TKFRAME_-76351_AXES             = (   2,   1,   3   )
      TKFRAME_-76351_UNITS            = 'DEGREES'

   \begintext

   This is the definition of the MAHLI_REF frame.

   \begindata

      FRAME_MSL_MAHLI_REF             = -76310
      FRAME_-76310_NAME               = 'MSL_MAHLI_REF'
      FRAME_-76310_CLASS              = 3
      FRAME_-76310_CLASS_ID           = -76310
      FRAME_-76310_CENTER             = -76
      CK_-76310_SCLK                  = -76
      CK_-76310_SPK                   = -76

   \begintext

   The actual MSL_MAHLI frame orientation provided in the frame
   definition below was computed using the CAHVOR(E) camera model file,
   'MSL_CAL_002_SN_3002_FOCUS_12594-MAHLI-FLIGHT.cahvor'. According to
   this model the reference frame, MSL_MAHLI_REF, can be transformed
   into the camera frame, MSL_MAHLI, by the following sequence of
   rotations: first by 90.53516876 degrees about Y, then by 0.47298218
   degrees about X, and finally by 0.50359886 degrees about Z.
 
   The frame definition below contains the opposite of this
   transformation because Euler angles specified in it define rotations
   from the "destination" frame to the "reference" frame.

   \begindata

      FRAME_MSL_MAHLI                  = -76311
      FRAME_-76311_NAME               = 'MSL_MAHLI'
      FRAME_-76311_CLASS              = 4
      FRAME_-76311_CLASS_ID           = -76311
      FRAME_-76311_CENTER             = -76
      TKFRAME_-76311_RELATIVE         = 'MSL_MAHLI_REF'
      TKFRAME_-76311_SPEC             = 'ANGLES'
      TKFRAME_-76311_UNITS            = 'DEGREES'
      TKFRAME_-76311_AXES             = (    2,        1,        3     )
      TKFRAME_-76311_ANGLES           = (  -90.535,   -0.473,   -0.504 )

   \begintext


MSL NAIF ID Codes -- Definition Section
========================================================================

   This section contains name to NAIF ID mappings for the MSL.

   \begindata

      NAIF_BODY_NAME += ( 'MSL_LANDING_SITE'               )
      NAIF_BODY_CODE += ( -76900                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL'                            )
      NAIF_BODY_CODE += ( -76                              )
                                                            
      NAIF_BODY_NAME += ( 'MSL_ROVER'                      )
      NAIF_BODY_CODE += ( -76000                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_SPACECRAFT'                 )
      NAIF_BODY_CODE += ( -76010                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CRUISE_STAGE'               )
      NAIF_BODY_CODE += ( -76020                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DESCENT_STAGE'              )
      NAIF_BODY_CODE += ( -76030                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_ROVER_MECH'                 )
      NAIF_BODY_CODE += ( -76040                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CACS'                       )
      NAIF_BODY_CODE += ( -76050                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DIMU_A'                     )
      NAIF_BODY_CODE += ( -76031                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_PLGA'                       )
      NAIF_BODY_CODE += ( -76060                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_TLGA'                       )
      NAIF_BODY_CODE += ( -76061                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_PUHF'                       )
      NAIF_BODY_CODE += ( -76062                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MGA'                        )
      NAIF_BODY_CODE += ( -76063                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DLGA'                       )
      NAIF_BODY_CODE += ( -76064                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DUHF'                       )
      NAIF_BODY_CODE += ( -76065                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MISP_T1'              )
      NAIF_BODY_CODE += ( -76071                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MISP_T2'              )
      NAIF_BODY_CODE += ( -76072                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MISP_T3'              )
      NAIF_BODY_CODE += ( -76073                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MISP_T4'              )
      NAIF_BODY_CODE += ( -76074                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MISP_T5'              )
      NAIF_BODY_CODE += ( -76075                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MISP_T6'              )
      NAIF_BODY_CODE += ( -76076                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MISP_T7'              )
      NAIF_BODY_CODE += ( -76077                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MEADS_P1'             )
      NAIF_BODY_CODE += ( -76081                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MEADS_P2'             )
      NAIF_BODY_CODE += ( -76082                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MEADS_P3'             )
      NAIF_BODY_CODE += ( -76083                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MEADS_P4'             )
      NAIF_BODY_CODE += ( -76084                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MEADS_P5'             )
      NAIF_BODY_CODE += ( -76085                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MEADS_P6'             )
      NAIF_BODY_CODE += ( -76086                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MEDLI_MEADS_P7'             )
      NAIF_BODY_CODE += ( -76087                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RLGA'                       )
      NAIF_BODY_CODE += ( -76110                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RUHF'                       )
      NAIF_BODY_CODE += ( -76111                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HGA_ZERO_AZ'                )
      NAIF_BODY_CODE += ( -76121                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HGA_AZ'                     )
      NAIF_BODY_CODE += ( -76122                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HGA_ZERO_EL'                )
      NAIF_BODY_CODE += ( -76123                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HGA_EL'                     )
      NAIF_BODY_CODE += ( -76124                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HGA'                        )
      NAIF_BODY_CODE += ( -76125                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HGA_EB'                     )
      NAIF_BODY_CODE += ( -76126                           )

      NAIF_BODY_NAME += ( 'MSL_HAZCAM_FRONT_LEFT_A'        )
      NAIF_BODY_CODE += ( -76131                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HAZCAM_FRONT_RIGHT_A'       )
      NAIF_BODY_CODE += ( -76132                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HAZCAM_FRONT_LEFT_B'        )
      NAIF_BODY_CODE += ( -76133                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HAZCAM_FRONT_RIGHT_B'       )
      NAIF_BODY_CODE += ( -76134                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HAZCAM_BACK_LEFT_A'         )
      NAIF_BODY_CODE += ( -76141                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HAZCAM_BACK_RIGHT_A'        )
      NAIF_BODY_CODE += ( -76142                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HAZCAM_BACK_LEFT_B'         )
      NAIF_BODY_CODE += ( -76143                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_HAZCAM_BACK_RIGHT_B'        )
      NAIF_BODY_CODE += ( -76144                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RAD'                        )
      NAIF_BODY_CODE += ( -76150                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MARDI'                      )
      NAIF_BODY_CODE += ( -76160                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_UVS'                   )
      NAIF_BODY_CODE += ( -76170                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DAN'                        )
      NAIF_BODY_CODE += ( -76180                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DAN_PNG'                    )
      NAIF_BODY_CODE += ( -76191                           )

      NAIF_BODY_NAME += ( 'MSL_DAN_DE'                     )
      NAIF_BODY_CODE += ( -76192                           )

      NAIF_BODY_NAME += ( 'MSL_DAN_DE_CTN'                 )
      NAIF_BODY_CODE += ( -76193                           )

      NAIF_BODY_NAME += ( 'MSL_DAN_DE_CETN'                )
      NAIF_BODY_CODE += ( -76194                           )

      NAIF_BODY_NAME += ( 'MSL_REMS_PS'                    )
      NAIF_BODY_CODE += ( -76181                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_SAM'                        )
      NAIF_BODY_CODE += ( -76182                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_SAM_1_INLET'                )
      NAIF_BODY_CODE += ( -76184                           )

      NAIF_BODY_NAME += ( 'MSL_SAM_2_INLET'                )
      NAIF_BODY_CODE += ( -76185                           )

      NAIF_BODY_NAME += ( 'MSL_CHEMIN'                     )
      NAIF_BODY_CODE += ( -76183                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CHEMIN_INLET'               )
      NAIF_BODY_CODE += ( -76186                           )

      NAIF_BODY_NAME += ( 'MSL_RSM_ZERO_AZ'                )
      NAIF_BODY_CODE += ( -76201                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RSM_AZ'                     )
      NAIF_BODY_CODE += ( -76202                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RSM_ZERO_EL'                )
      NAIF_BODY_CODE += ( -76203                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RSM_EL'                     )
      NAIF_BODY_CODE += ( -76204                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RSM_HEAD'                   )
      NAIF_BODY_CODE += ( -76205                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT'               )
      NAIF_BODY_CODE += ( -76210                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F1'            )
      NAIF_BODY_CODE += ( -76211                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F2'            )
      NAIF_BODY_CODE += ( -76212                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F3'            )
      NAIF_BODY_CODE += ( -76213                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F4'            )
      NAIF_BODY_CODE += ( -76214                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F5'            )
      NAIF_BODY_CODE += ( -76215                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F6'            )
      NAIF_BODY_CODE += ( -76216                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F7'            )
      NAIF_BODY_CODE += ( -76217                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_LEFT_F8'            )
      NAIF_BODY_CODE += ( -76218                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT'              )
      NAIF_BODY_CODE += ( -76220                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F1'           )
      NAIF_BODY_CODE += ( -76221                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F2'           )
      NAIF_BODY_CODE += ( -76222                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F3'           )
      NAIF_BODY_CODE += ( -76223                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F4'           )
      NAIF_BODY_CODE += ( -76224                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F5'           )
      NAIF_BODY_CODE += ( -76225                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F6'           )
      NAIF_BODY_CODE += ( -76226                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F7'           )
      NAIF_BODY_CODE += ( -76227                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MASTCAM_RIGHT_F8'           )
      NAIF_BODY_CODE += ( -76228                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_NAVCAM_LEFT_A'              )
      NAIF_BODY_CODE += ( -76231                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_NAVCAM_RIGHT_A'             )
      NAIF_BODY_CODE += ( -76232                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_NAVCAM_LEFT_B'              )
      NAIF_BODY_CODE += ( -76233                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_NAVCAM_RIGHT_B'             )
      NAIF_BODY_CODE += ( -76234                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CHEMCAM'                    )
      NAIF_BODY_CODE += ( -76240                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CHEMCAM_LIBS_LASER'         )
      NAIF_BODY_CODE += ( -76241                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CHEMCAM_LIBS_CAM'           )
      NAIF_BODY_CODE += ( -76242                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CHEMCAM_RMI'                )
      NAIF_BODY_CODE += ( -76243                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM1'                 )
      NAIF_BODY_CODE += ( -76250                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM1_WS1'             )
      NAIF_BODY_CODE += ( -76251                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM1_WS2'             )
      NAIF_BODY_CODE += ( -76252                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM1_WS3'             )
      NAIF_BODY_CODE += ( -76253                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM1_ATS'             )
      NAIF_BODY_CODE += ( -76254                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM1_GTS'             )
      NAIF_BODY_CODE += ( -76255                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM1_TIP'             )
      NAIF_BODY_CODE += ( -76256                           )

      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM2'                 )
      NAIF_BODY_CODE += ( -76260                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM2_WS1'             )
      NAIF_BODY_CODE += ( -76261                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM2_WS2'             )
      NAIF_BODY_CODE += ( -76262                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM2_WS3'             )
      NAIF_BODY_CODE += ( -76263                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM2_ATS'             )
      NAIF_BODY_CODE += ( -76264                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM2_HS'              )
      NAIF_BODY_CODE += ( -76265                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_REMS_BOOM2_TIP'             )
      NAIF_BODY_CODE += ( -76266                           )

      NAIF_BODY_NAME += ( 'MSL_RA_BASE'                    )
      NAIF_BODY_CODE += ( -76300                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RA_SHOULDER_AZ'             )
      NAIF_BODY_CODE += ( -76301                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RA_SHOULDER_EL'             )
      NAIF_BODY_CODE += ( -76302                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RA_ELBOW'                   )
      NAIF_BODY_CODE += ( -76303                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RA_WRIST'                   )
      NAIF_BODY_CODE += ( -76304                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RA_TURRET'                  )
      NAIF_BODY_CODE += ( -76305                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_RA_TURRET_HEAD'             )
      NAIF_BODY_CODE += ( -76306                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MAHLI_REF'                  )
      NAIF_BODY_CODE += ( -76310                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_MAHLI'                      )
      NAIF_BODY_CODE += ( -76311                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_APXS_REF'                   )
      NAIF_BODY_CODE += ( -76320                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_APXS'                       )
      NAIF_BODY_CODE += ( -76321                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_PADS_REF'                   )
      NAIF_BODY_CODE += ( -76330                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_PADS'                       )
      NAIF_BODY_CODE += ( -76331                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DRT_REF'                    )
      NAIF_BODY_CODE += ( -76340                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_DRT'                        )
      NAIF_BODY_CODE += ( -76341                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CHIMRA_REF'                 )
      NAIF_BODY_CODE += ( -76350                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_CHIMRA'                     )
      NAIF_BODY_CODE += ( -76351                           )
                                                            
      NAIF_BODY_NAME += ( 'MSL_FD_RESTRAINT_PORT'          )
      NAIF_BODY_CODE += ( -76401                           )

      NAIF_BODY_NAME += ( 'MSL_FD_RESTRAINT_SBRD'          )
      NAIF_BODY_CODE += ( -76402                           )

      NAIF_BODY_NAME += ( 'MSL_FD_DECK_SIDE_PORT'          )
      NAIF_BODY_CODE += ( -76403                           )

      NAIF_BODY_NAME += ( 'MSL_FD_DECK_FRONT_PORT'         )
      NAIF_BODY_CODE += ( -76404                           )

      NAIF_BODY_NAME += ( 'MSL_FD_DECK_CENTER_PORT'        )
      NAIF_BODY_CODE += ( -76405                           )

      NAIF_BODY_NAME += ( 'MSL_FD_DECK_CENTER_SBRD'        )
      NAIF_BODY_CODE += ( -76406                           )

      NAIF_BODY_NAME += ( 'MSL_FD_DECK_FRONT_SBRD'         )
      NAIF_BODY_CODE += ( -76407                           )

      NAIF_BODY_NAME += ( 'MSL_FD_OCM_PORT'                )
      NAIF_BODY_CODE += ( -76408                           )

      NAIF_BODY_NAME += ( 'MSL_FD_OCM_CENTER'              )
      NAIF_BODY_CODE += ( -76409                           )

      NAIF_BODY_NAME += ( 'MSL_FD_OCM_STARBOARD'           )
      NAIF_BODY_CODE += ( -76410                           )

      NAIF_BODY_NAME += ( 'MSL_SCI_OBS_TRAY'               )
      NAIF_BODY_CODE += ( -76411                           )

      NAIF_BODY_NAME += ( 'MSL_ENG_OBS_TRAY'               )
      NAIF_BODY_CODE += ( -76412                           )

      NAIF_BODY_NAME += ( 'MSL_BIT_BOX_1_TARGET'           )
      NAIF_BODY_CODE += ( -76413                           )

      NAIF_BODY_NAME += ( 'MSL_BIT_BOX_2_TARGET'           )
      NAIF_BODY_CODE += ( -76414                           )

      NAIF_BODY_NAME += ( 'MSL_OCM_LOCATION_1'             )
      NAIF_BODY_CODE += ( -76415                           )

      NAIF_BODY_NAME += ( 'MSL_OCM_LOCATION_2'             )
      NAIF_BODY_CODE += ( -76416                           )

      NAIF_BODY_NAME += ( 'MSL_OCM_LOCATION_3'             )
      NAIF_BODY_CODE += ( -76417                           )

      NAIF_BODY_NAME += ( 'MSL_OCM_LOCATION_4'             )
      NAIF_BODY_CODE += ( -76418                           )

      NAIF_BODY_NAME += ( 'MSL_OCM_LOCATION_5'             )
      NAIF_BODY_CODE += ( -76419                           )

      NAIF_BODY_NAME += ( 'MSL_OCM_LOCATION_6'             )
      NAIF_BODY_CODE += ( -76420                           )

      NAIF_BODY_NAME += ( 'MSL_APXS_CALTARGET'             )
      NAIF_BODY_CODE += ( -76421                           )

      NAIF_BODY_NAME += ( 'MSL_MAHLI_CALTARGET'            )
      NAIF_BODY_CODE += ( -76422                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL1'                  )
      NAIF_BODY_CODE += ( -76423                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL2'                  )
      NAIF_BODY_CODE += ( -76424                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL3'                  )
      NAIF_BODY_CODE += ( -76425                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL4'                  )
      NAIF_BODY_CODE += ( -76426                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL5'                  )
      NAIF_BODY_CODE += ( -76427                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL6'                  )
      NAIF_BODY_CODE += ( -76428                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL7'                  )
      NAIF_BODY_CODE += ( -76429                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL8'                  )
      NAIF_BODY_CODE += ( -76430                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL9'                  )
      NAIF_BODY_CODE += ( -76431                           )

      NAIF_BODY_NAME += ( 'MSL_CCAM_CAL10'                 )
      NAIF_BODY_CODE += ( -76432                           )

      NAIF_BODY_NAME += ( 'MSL_MCAM_CAL'                   )
      NAIF_BODY_CODE += ( -76433                           )

   \begintext

   MSL site name to NAIF ID mappings.

   \begindata
                                                            
      NAIF_BODY_NAME += ( 'MSL_SITE_1'                     )
      NAIF_BODY_CODE += ( -76501                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_2'                     )
      NAIF_BODY_CODE += ( -76502                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_3'                     )
      NAIF_BODY_CODE += ( -76503                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_4'                     )
      NAIF_BODY_CODE += ( -76504                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_5'                     )
      NAIF_BODY_CODE += ( -76505                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_6'                     )
      NAIF_BODY_CODE += ( -76506                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_7'                     )
      NAIF_BODY_CODE += ( -76507                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_8'                     )
      NAIF_BODY_CODE += ( -76508                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_9'                     )
      NAIF_BODY_CODE += ( -76509                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_10'                    )
      NAIF_BODY_CODE += ( -76510                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_11'                    )
      NAIF_BODY_CODE += ( -76511                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_12'                    )
      NAIF_BODY_CODE += ( -76512                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_13'                    )
      NAIF_BODY_CODE += ( -76513                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_14'                    )
      NAIF_BODY_CODE += ( -76514                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_15'                    )
      NAIF_BODY_CODE += ( -76515                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_16'                    )
      NAIF_BODY_CODE += ( -76516                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_17'                    )
      NAIF_BODY_CODE += ( -76517                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_18'                    )
      NAIF_BODY_CODE += ( -76518                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_19'                    )
      NAIF_BODY_CODE += ( -76519                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_20'                    )
      NAIF_BODY_CODE += ( -76520                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_21'                    )
      NAIF_BODY_CODE += ( -76521                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_22'                    )
      NAIF_BODY_CODE += ( -76522                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_23'                    )
      NAIF_BODY_CODE += ( -76523                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_24'                    )
      NAIF_BODY_CODE += ( -76524                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_25'                    )
      NAIF_BODY_CODE += ( -76525                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_26'                    )
      NAIF_BODY_CODE += ( -76526                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_27'                    )
      NAIF_BODY_CODE += ( -76527                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_28'                    )
      NAIF_BODY_CODE += ( -76528                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_29'                    )
      NAIF_BODY_CODE += ( -76529                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_30'                    )
      NAIF_BODY_CODE += ( -76530                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_31'                    )
      NAIF_BODY_CODE += ( -76531                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_32'                    )
      NAIF_BODY_CODE += ( -76532                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_33'                    )
      NAIF_BODY_CODE += ( -76533                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_34'                    )
      NAIF_BODY_CODE += ( -76534                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_35'                    )
      NAIF_BODY_CODE += ( -76535                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_36'                    )
      NAIF_BODY_CODE += ( -76536                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_37'                    )
      NAIF_BODY_CODE += ( -76537                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_38'                    )
      NAIF_BODY_CODE += ( -76538                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_39'                    )
      NAIF_BODY_CODE += ( -76539                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_40'                    )
      NAIF_BODY_CODE += ( -76540                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_41'                    )
      NAIF_BODY_CODE += ( -76541                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_42'                    )
      NAIF_BODY_CODE += ( -76542                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_43'                    )
      NAIF_BODY_CODE += ( -76543                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_44'                    )
      NAIF_BODY_CODE += ( -76544                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_45'                    )
      NAIF_BODY_CODE += ( -76545                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_46'                    )
      NAIF_BODY_CODE += ( -76546                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_47'                    )
      NAIF_BODY_CODE += ( -76547                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_48'                    )
      NAIF_BODY_CODE += ( -76548                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_49'                    )
      NAIF_BODY_CODE += ( -76549                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_50'                    )
      NAIF_BODY_CODE += ( -76550                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_51'                    )
      NAIF_BODY_CODE += ( -76551                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_52'                    )
      NAIF_BODY_CODE += ( -76552                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_53'                    )
      NAIF_BODY_CODE += ( -76553                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_54'                    )
      NAIF_BODY_CODE += ( -76554                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_55'                    )
      NAIF_BODY_CODE += ( -76555                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_56'                    )
      NAIF_BODY_CODE += ( -76556                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_57'                    )
      NAIF_BODY_CODE += ( -76557                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_58'                    )
      NAIF_BODY_CODE += ( -76558                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_59'                    )
      NAIF_BODY_CODE += ( -76559                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_60'                    )
      NAIF_BODY_CODE += ( -76560                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_61'                    )
      NAIF_BODY_CODE += ( -76561                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_62'                    )
      NAIF_BODY_CODE += ( -76562                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_63'                    )
      NAIF_BODY_CODE += ( -76563                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_64'                    )
      NAIF_BODY_CODE += ( -76564                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_65'                    )
      NAIF_BODY_CODE += ( -76565                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_66'                    )
      NAIF_BODY_CODE += ( -76566                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_67'                    )
      NAIF_BODY_CODE += ( -76567                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_68'                    )
      NAIF_BODY_CODE += ( -76568                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_69'                    )
      NAIF_BODY_CODE += ( -76569                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_70'                    )
      NAIF_BODY_CODE += ( -76570                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_71'                    )
      NAIF_BODY_CODE += ( -76571                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_72'                    )
      NAIF_BODY_CODE += ( -76572                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_73'                    )
      NAIF_BODY_CODE += ( -76573                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_74'                    )
      NAIF_BODY_CODE += ( -76574                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_75'                    )
      NAIF_BODY_CODE += ( -76575                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_76'                    )
      NAIF_BODY_CODE += ( -76576                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_77'                    )
      NAIF_BODY_CODE += ( -76577                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_78'                    )
      NAIF_BODY_CODE += ( -76578                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_79'                    )
      NAIF_BODY_CODE += ( -76579                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_80'                    )
      NAIF_BODY_CODE += ( -76580                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_81'                    )
      NAIF_BODY_CODE += ( -76581                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_82'                    )
      NAIF_BODY_CODE += ( -76582                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_83'                    )
      NAIF_BODY_CODE += ( -76583                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_84'                    )
      NAIF_BODY_CODE += ( -76584                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_85'                    )
      NAIF_BODY_CODE += ( -76585                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_86'                    )
      NAIF_BODY_CODE += ( -76586                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_87'                    )
      NAIF_BODY_CODE += ( -76587                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_88'                    )
      NAIF_BODY_CODE += ( -76588                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_89'                    )
      NAIF_BODY_CODE += ( -76589                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_90'                    )
      NAIF_BODY_CODE += ( -76590                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_91'                    )
      NAIF_BODY_CODE += ( -76591                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_92'                    )
      NAIF_BODY_CODE += ( -76592                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_93'                    )
      NAIF_BODY_CODE += ( -76593                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_94'                    )
      NAIF_BODY_CODE += ( -76594                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_95'                    )
      NAIF_BODY_CODE += ( -76595                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_96'                    )
      NAIF_BODY_CODE += ( -76596                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_97'                    )
      NAIF_BODY_CODE += ( -76597                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_98'                    )
      NAIF_BODY_CODE += ( -76598                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_99'                    )
      NAIF_BODY_CODE += ( -76599                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_100'                   )
      NAIF_BODY_CODE += ( -76600                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_101'                   )
      NAIF_BODY_CODE += ( -76601                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_102'                   )
      NAIF_BODY_CODE += ( -76602                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_103'                   )
      NAIF_BODY_CODE += ( -76603                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_104'                   )
      NAIF_BODY_CODE += ( -76604                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_105'                   )
      NAIF_BODY_CODE += ( -76605                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_106'                   )
      NAIF_BODY_CODE += ( -76606                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_107'                   )
      NAIF_BODY_CODE += ( -76607                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_108'                   )
      NAIF_BODY_CODE += ( -76608                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_109'                   )
      NAIF_BODY_CODE += ( -76609                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_110'                   )
      NAIF_BODY_CODE += ( -76610                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_111'                   )
      NAIF_BODY_CODE += ( -76611                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_112'                   )
      NAIF_BODY_CODE += ( -76612                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_113'                   )
      NAIF_BODY_CODE += ( -76613                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_114'                   )
      NAIF_BODY_CODE += ( -76614                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_115'                   )
      NAIF_BODY_CODE += ( -76615                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_116'                   )
      NAIF_BODY_CODE += ( -76616                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_117'                   )
      NAIF_BODY_CODE += ( -76617                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_118'                   )
      NAIF_BODY_CODE += ( -76618                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_119'                   )
      NAIF_BODY_CODE += ( -76619                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_120'                   )
      NAIF_BODY_CODE += ( -76620                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_121'                   )
      NAIF_BODY_CODE += ( -76621                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_122'                   )
      NAIF_BODY_CODE += ( -76622                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_123'                   )
      NAIF_BODY_CODE += ( -76623                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_124'                   )
      NAIF_BODY_CODE += ( -76624                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_125'                   )
      NAIF_BODY_CODE += ( -76625                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_126'                   )
      NAIF_BODY_CODE += ( -76626                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_127'                   )
      NAIF_BODY_CODE += ( -76627                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_128'                   )
      NAIF_BODY_CODE += ( -76628                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_129'                   )
      NAIF_BODY_CODE += ( -76629                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_130'                   )
      NAIF_BODY_CODE += ( -76630                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_131'                   )
      NAIF_BODY_CODE += ( -76631                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_132'                   )
      NAIF_BODY_CODE += ( -76632                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_133'                   )
      NAIF_BODY_CODE += ( -76633                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_134'                   )
      NAIF_BODY_CODE += ( -76634                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_135'                   )
      NAIF_BODY_CODE += ( -76635                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_136'                   )
      NAIF_BODY_CODE += ( -76636                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_137'                   )
      NAIF_BODY_CODE += ( -76637                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_138'                   )
      NAIF_BODY_CODE += ( -76638                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_139'                   )
      NAIF_BODY_CODE += ( -76639                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_140'                   )
      NAIF_BODY_CODE += ( -76640                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_141'                   )
      NAIF_BODY_CODE += ( -76641                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_142'                   )
      NAIF_BODY_CODE += ( -76642                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_143'                   )
      NAIF_BODY_CODE += ( -76643                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_144'                   )
      NAIF_BODY_CODE += ( -76644                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_145'                   )
      NAIF_BODY_CODE += ( -76645                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_146'                   )
      NAIF_BODY_CODE += ( -76646                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_147'                   )
      NAIF_BODY_CODE += ( -76647                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_148'                   )
      NAIF_BODY_CODE += ( -76648                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_149'                   )
      NAIF_BODY_CODE += ( -76649                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_150'                   )
      NAIF_BODY_CODE += ( -76650                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_151'                   )
      NAIF_BODY_CODE += ( -76651                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_152'                   )
      NAIF_BODY_CODE += ( -76652                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_153'                   )
      NAIF_BODY_CODE += ( -76653                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_154'                   )
      NAIF_BODY_CODE += ( -76654                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_155'                   )
      NAIF_BODY_CODE += ( -76655                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_156'                   )
      NAIF_BODY_CODE += ( -76656                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_157'                   )
      NAIF_BODY_CODE += ( -76657                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_158'                   )
      NAIF_BODY_CODE += ( -76658                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_159'                   )
      NAIF_BODY_CODE += ( -76659                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_160'                   )
      NAIF_BODY_CODE += ( -76660                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_161'                   )
      NAIF_BODY_CODE += ( -76661                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_162'                   )
      NAIF_BODY_CODE += ( -76662                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_163'                   )
      NAIF_BODY_CODE += ( -76663                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_164'                   )
      NAIF_BODY_CODE += ( -76664                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_165'                   )
      NAIF_BODY_CODE += ( -76665                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_166'                   )
      NAIF_BODY_CODE += ( -76666                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_167'                   )
      NAIF_BODY_CODE += ( -76667                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_168'                   )
      NAIF_BODY_CODE += ( -76668                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_169'                   )
      NAIF_BODY_CODE += ( -76669                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_170'                   )
      NAIF_BODY_CODE += ( -76670                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_171'                   )
      NAIF_BODY_CODE += ( -76671                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_172'                   )
      NAIF_BODY_CODE += ( -76672                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_173'                   )
      NAIF_BODY_CODE += ( -76673                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_174'                   )
      NAIF_BODY_CODE += ( -76674                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_175'                   )
      NAIF_BODY_CODE += ( -76675                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_176'                   )
      NAIF_BODY_CODE += ( -76676                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_177'                   )
      NAIF_BODY_CODE += ( -76677                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_178'                   )
      NAIF_BODY_CODE += ( -76678                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_179'                   )
      NAIF_BODY_CODE += ( -76679                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_180'                   )
      NAIF_BODY_CODE += ( -76680                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_181'                   )
      NAIF_BODY_CODE += ( -76681                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_182'                   )
      NAIF_BODY_CODE += ( -76682                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_183'                   )
      NAIF_BODY_CODE += ( -76683                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_184'                   )
      NAIF_BODY_CODE += ( -76684                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_185'                   )
      NAIF_BODY_CODE += ( -76685                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_186'                   )
      NAIF_BODY_CODE += ( -76686                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_187'                   )
      NAIF_BODY_CODE += ( -76687                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_188'                   )
      NAIF_BODY_CODE += ( -76688                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_189'                   )
      NAIF_BODY_CODE += ( -76689                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_190'                   )
      NAIF_BODY_CODE += ( -76690                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_191'                   )
      NAIF_BODY_CODE += ( -76691                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_192'                   )
      NAIF_BODY_CODE += ( -76692                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_193'                   )
      NAIF_BODY_CODE += ( -76693                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_194'                   )
      NAIF_BODY_CODE += ( -76694                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_195'                   )
      NAIF_BODY_CODE += ( -76695                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_196'                   )
      NAIF_BODY_CODE += ( -76696                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_197'                   )
      NAIF_BODY_CODE += ( -76697                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_198'                   )
      NAIF_BODY_CODE += ( -76698                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_199'                   )
      NAIF_BODY_CODE += ( -76699                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_200'                   )
      NAIF_BODY_CODE += ( -76700                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_201'                   )
      NAIF_BODY_CODE += ( -76701                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_202'                   )
      NAIF_BODY_CODE += ( -76702                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_203'                   )
      NAIF_BODY_CODE += ( -76703                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_204'                   )
      NAIF_BODY_CODE += ( -76704                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_205'                   )
      NAIF_BODY_CODE += ( -76705                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_206'                   )
      NAIF_BODY_CODE += ( -76706                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_207'                   )
      NAIF_BODY_CODE += ( -76707                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_208'                   )
      NAIF_BODY_CODE += ( -76708                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_209'                   )
      NAIF_BODY_CODE += ( -76709                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_210'                   )
      NAIF_BODY_CODE += ( -76710                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_211'                   )
      NAIF_BODY_CODE += ( -76711                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_212'                   )
      NAIF_BODY_CODE += ( -76712                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_213'                   )
      NAIF_BODY_CODE += ( -76713                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_214'                   )
      NAIF_BODY_CODE += ( -76714                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_215'                   )
      NAIF_BODY_CODE += ( -76715                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_216'                   )
      NAIF_BODY_CODE += ( -76716                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_217'                   )
      NAIF_BODY_CODE += ( -76717                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_218'                   )
      NAIF_BODY_CODE += ( -76718                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_219'                   )
      NAIF_BODY_CODE += ( -76719                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_220'                   )
      NAIF_BODY_CODE += ( -76720                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_221'                   )
      NAIF_BODY_CODE += ( -76721                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_222'                   )
      NAIF_BODY_CODE += ( -76722                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_223'                   )
      NAIF_BODY_CODE += ( -76723                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_224'                   )
      NAIF_BODY_CODE += ( -76724                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_225'                   )
      NAIF_BODY_CODE += ( -76725                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_226'                   )
      NAIF_BODY_CODE += ( -76726                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_227'                   )
      NAIF_BODY_CODE += ( -76727                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_228'                   )
      NAIF_BODY_CODE += ( -76728                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_229'                   )
      NAIF_BODY_CODE += ( -76729                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_230'                   )
      NAIF_BODY_CODE += ( -76730                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_231'                   )
      NAIF_BODY_CODE += ( -76731                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_232'                   )
      NAIF_BODY_CODE += ( -76732                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_233'                   )
      NAIF_BODY_CODE += ( -76733                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_234'                   )
      NAIF_BODY_CODE += ( -76734                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_235'                   )
      NAIF_BODY_CODE += ( -76735                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_236'                   )
      NAIF_BODY_CODE += ( -76736                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_237'                   )
      NAIF_BODY_CODE += ( -76737                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_238'                   )
      NAIF_BODY_CODE += ( -76738                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_239'                   )
      NAIF_BODY_CODE += ( -76739                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_240'                   )
      NAIF_BODY_CODE += ( -76740                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_241'                   )
      NAIF_BODY_CODE += ( -76741                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_242'                   )
      NAIF_BODY_CODE += ( -76742                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_243'                   )
      NAIF_BODY_CODE += ( -76743                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_244'                   )
      NAIF_BODY_CODE += ( -76744                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_245'                   )
      NAIF_BODY_CODE += ( -76745                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_246'                   )
      NAIF_BODY_CODE += ( -76746                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_247'                   )
      NAIF_BODY_CODE += ( -76747                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_248'                   )
      NAIF_BODY_CODE += ( -76748                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_249'                   )
      NAIF_BODY_CODE += ( -76749                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_250'                   )
      NAIF_BODY_CODE += ( -76750                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_251'                   )
      NAIF_BODY_CODE += ( -76751                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_252'                   )
      NAIF_BODY_CODE += ( -76752                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_253'                   )
      NAIF_BODY_CODE += ( -76753                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_254'                   )
      NAIF_BODY_CODE += ( -76754                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_255'                   )
      NAIF_BODY_CODE += ( -76755                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_256'                   )
      NAIF_BODY_CODE += ( -76756                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_257'                   )
      NAIF_BODY_CODE += ( -76757                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_258'                   )
      NAIF_BODY_CODE += ( -76758                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_259'                   )
      NAIF_BODY_CODE += ( -76759                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_260'                   )
      NAIF_BODY_CODE += ( -76760                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_261'                   )
      NAIF_BODY_CODE += ( -76761                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_262'                   )
      NAIF_BODY_CODE += ( -76762                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_263'                   )
      NAIF_BODY_CODE += ( -76763                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_264'                   )
      NAIF_BODY_CODE += ( -76764                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_265'                   )
      NAIF_BODY_CODE += ( -76765                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_266'                   )
      NAIF_BODY_CODE += ( -76766                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_267'                   )
      NAIF_BODY_CODE += ( -76767                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_268'                   )
      NAIF_BODY_CODE += ( -76768                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_269'                   )
      NAIF_BODY_CODE += ( -76769                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_270'                   )
      NAIF_BODY_CODE += ( -76770                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_271'                   )
      NAIF_BODY_CODE += ( -76771                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_272'                   )
      NAIF_BODY_CODE += ( -76772                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_273'                   )
      NAIF_BODY_CODE += ( -76773                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_274'                   )
      NAIF_BODY_CODE += ( -76774                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_275'                   )
      NAIF_BODY_CODE += ( -76775                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_276'                   )
      NAIF_BODY_CODE += ( -76776                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_277'                   )
      NAIF_BODY_CODE += ( -76777                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_278'                   )
      NAIF_BODY_CODE += ( -76778                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_279'                   )
      NAIF_BODY_CODE += ( -76779                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_280'                   )
      NAIF_BODY_CODE += ( -76780                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_281'                   )
      NAIF_BODY_CODE += ( -76781                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_282'                   )
      NAIF_BODY_CODE += ( -76782                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_283'                   )
      NAIF_BODY_CODE += ( -76783                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_284'                   )
      NAIF_BODY_CODE += ( -76784                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_285'                   )
      NAIF_BODY_CODE += ( -76785                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_286'                   )
      NAIF_BODY_CODE += ( -76786                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_287'                   )
      NAIF_BODY_CODE += ( -76787                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_288'                   )
      NAIF_BODY_CODE += ( -76788                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_289'                   )
      NAIF_BODY_CODE += ( -76789                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_290'                   )
      NAIF_BODY_CODE += ( -76790                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_291'                   )
      NAIF_BODY_CODE += ( -76791                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_292'                   )
      NAIF_BODY_CODE += ( -76792                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_293'                   )
      NAIF_BODY_CODE += ( -76793                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_294'                   )
      NAIF_BODY_CODE += ( -76794                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_295'                   )
      NAIF_BODY_CODE += ( -76795                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_296'                   )
      NAIF_BODY_CODE += ( -76796                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_297'                   )
      NAIF_BODY_CODE += ( -76797                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_298'                   )
      NAIF_BODY_CODE += ( -76798                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_299'                   )
      NAIF_BODY_CODE += ( -76799                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_300'                   )
      NAIF_BODY_CODE += ( -76800                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_301'                   )
      NAIF_BODY_CODE += ( -76801                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_302'                   )
      NAIF_BODY_CODE += ( -76802                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_303'                   )
      NAIF_BODY_CODE += ( -76803                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_304'                   )
      NAIF_BODY_CODE += ( -76804                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_305'                   )
      NAIF_BODY_CODE += ( -76805                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_306'                   )
      NAIF_BODY_CODE += ( -76806                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_307'                   )
      NAIF_BODY_CODE += ( -76807                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_308'                   )
      NAIF_BODY_CODE += ( -76808                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_309'                   )
      NAIF_BODY_CODE += ( -76809                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_310'                   )
      NAIF_BODY_CODE += ( -76810                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_311'                   )
      NAIF_BODY_CODE += ( -76811                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_312'                   )
      NAIF_BODY_CODE += ( -76812                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_313'                   )
      NAIF_BODY_CODE += ( -76813                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_314'                   )
      NAIF_BODY_CODE += ( -76814                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_315'                   )
      NAIF_BODY_CODE += ( -76815                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_316'                   )
      NAIF_BODY_CODE += ( -76816                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_317'                   )
      NAIF_BODY_CODE += ( -76817                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_318'                   )
      NAIF_BODY_CODE += ( -76818                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_319'                   )
      NAIF_BODY_CODE += ( -76819                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_320'                   )
      NAIF_BODY_CODE += ( -76820                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_321'                   )
      NAIF_BODY_CODE += ( -76821                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_322'                   )
      NAIF_BODY_CODE += ( -76822                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_323'                   )
      NAIF_BODY_CODE += ( -76823                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_324'                   )
      NAIF_BODY_CODE += ( -76824                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_325'                   )
      NAIF_BODY_CODE += ( -76825                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_326'                   )
      NAIF_BODY_CODE += ( -76826                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_327'                   )
      NAIF_BODY_CODE += ( -76827                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_328'                   )
      NAIF_BODY_CODE += ( -76828                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_329'                   )
      NAIF_BODY_CODE += ( -76829                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_330'                   )
      NAIF_BODY_CODE += ( -76830                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_331'                   )
      NAIF_BODY_CODE += ( -76831                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_332'                   )
      NAIF_BODY_CODE += ( -76832                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_333'                   )
      NAIF_BODY_CODE += ( -76833                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_334'                   )
      NAIF_BODY_CODE += ( -76834                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_335'                   )
      NAIF_BODY_CODE += ( -76835                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_336'                   )
      NAIF_BODY_CODE += ( -76836                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_337'                   )
      NAIF_BODY_CODE += ( -76837                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_338'                   )
      NAIF_BODY_CODE += ( -76838                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_339'                   )
      NAIF_BODY_CODE += ( -76839                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_340'                   )
      NAIF_BODY_CODE += ( -76840                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_341'                   )
      NAIF_BODY_CODE += ( -76841                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_342'                   )
      NAIF_BODY_CODE += ( -76842                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_343'                   )
      NAIF_BODY_CODE += ( -76843                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_344'                   )
      NAIF_BODY_CODE += ( -76844                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_345'                   )
      NAIF_BODY_CODE += ( -76845                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_346'                   )
      NAIF_BODY_CODE += ( -76846                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_347'                   )
      NAIF_BODY_CODE += ( -76847                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_348'                   )
      NAIF_BODY_CODE += ( -76848                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_349'                   )
      NAIF_BODY_CODE += ( -76849                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_350'                   )
      NAIF_BODY_CODE += ( -76850                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_351'                   )
      NAIF_BODY_CODE += ( -76851                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_352'                   )
      NAIF_BODY_CODE += ( -76852                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_353'                   )
      NAIF_BODY_CODE += ( -76853                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_354'                   )
      NAIF_BODY_CODE += ( -76854                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_355'                   )
      NAIF_BODY_CODE += ( -76855                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_356'                   )
      NAIF_BODY_CODE += ( -76856                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_357'                   )
      NAIF_BODY_CODE += ( -76857                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_358'                   )
      NAIF_BODY_CODE += ( -76858                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_359'                   )
      NAIF_BODY_CODE += ( -76859                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_360'                   )
      NAIF_BODY_CODE += ( -76860                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_361'                   )
      NAIF_BODY_CODE += ( -76861                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_362'                   )
      NAIF_BODY_CODE += ( -76862                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_363'                   )
      NAIF_BODY_CODE += ( -76863                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_364'                   )
      NAIF_BODY_CODE += ( -76864                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_365'                   )
      NAIF_BODY_CODE += ( -76865                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_366'                   )
      NAIF_BODY_CODE += ( -76866                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_367'                   )
      NAIF_BODY_CODE += ( -76867                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_368'                   )
      NAIF_BODY_CODE += ( -76868                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_369'                   )
      NAIF_BODY_CODE += ( -76869                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_370'                   )
      NAIF_BODY_CODE += ( -76870                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_371'                   )
      NAIF_BODY_CODE += ( -76871                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_372'                   )
      NAIF_BODY_CODE += ( -76872                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_373'                   )
      NAIF_BODY_CODE += ( -76873                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_374'                   )
      NAIF_BODY_CODE += ( -76874                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_375'                   )
      NAIF_BODY_CODE += ( -76875                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_376'                   )
      NAIF_BODY_CODE += ( -76876                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_377'                   )
      NAIF_BODY_CODE += ( -76877                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_378'                   )
      NAIF_BODY_CODE += ( -76878                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_379'                   )
      NAIF_BODY_CODE += ( -76879                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_380'                   )
      NAIF_BODY_CODE += ( -76880                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_381'                   )
      NAIF_BODY_CODE += ( -76881                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_382'                   )
      NAIF_BODY_CODE += ( -76882                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_383'                   )
      NAIF_BODY_CODE += ( -76883                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_384'                   )
      NAIF_BODY_CODE += ( -76884                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_385'                   )
      NAIF_BODY_CODE += ( -76885                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_386'                   )
      NAIF_BODY_CODE += ( -76886                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_387'                   )
      NAIF_BODY_CODE += ( -76887                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_388'                   )
      NAIF_BODY_CODE += ( -76888                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_389'                   )
      NAIF_BODY_CODE += ( -76889                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_390'                   )
      NAIF_BODY_CODE += ( -76890                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_391'                   )
      NAIF_BODY_CODE += ( -76891                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_392'                   )
      NAIF_BODY_CODE += ( -76892                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_393'                   )
      NAIF_BODY_CODE += ( -76893                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_394'                   )
      NAIF_BODY_CODE += ( -76894                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_395'                   )
      NAIF_BODY_CODE += ( -76895                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_396'                   )
      NAIF_BODY_CODE += ( -76896                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_397'                   )
      NAIF_BODY_CODE += ( -76897                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_398'                   )
      NAIF_BODY_CODE += ( -76898                           )

      NAIF_BODY_NAME += ( 'MSL_SITE_399'                   )
      NAIF_BODY_CODE += ( -76899                           )

   \begintext

End of FK file.
