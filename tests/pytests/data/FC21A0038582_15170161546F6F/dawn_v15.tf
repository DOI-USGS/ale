KPL/FK

Dawn Frames Kernel
========================================================================

   This frame kernel contains complete set of frame definitions for the
   Dawn Spacecraft (DAWN) including definitions for the Dawn spacecraft
   and Dawn science instrument and engineering structure frames. This
   kernel also contains NAIF ID/name mapping for the Dawn instruments.


Version and Date
========================================================================


   Version 1.5 -- October 19, 2018 -- Boris Semenov, NAIF

      Added more DSK surface name-IDs.

   Version 1.4 -- March 29, 2018 -- Boris Semenov, NAIF

      Added DSK surface name-ID mapping section.

   Version 1.3 -- April 6, 2016 -- Boris Semenov, NAIF

      Redefined the DAWN_FC2 frame as a CK-based frame to facilitate
      providing different FC2 alignments relative to the s/c frame for
      Vesta and Ceres missions.

   Version 1.2 -- October 29, 2012 -- Boris Semenov, NAIF

      Added Euler dynamic frames for Vesta coordinate systems
      ``IAU-2000'' and ``DAWN-Claudia''.

   Version 1.1 -- July 5, 2011 -- Boris Semenov, NAIF

      Corrected description of the FC1 and FC2 frames. All frame 
      definitions are identical to Version 1.0.

   Version 1.0 -- March 21, 2011 -- Boris Semenov, NAIF

      Incorporated updated definitions of the DAWN_VIR_VIS, DAWN_VIR_IR, 
      DAWN_VIR_VIS_ZERO, and DAWN_VIR_IR_ZERO frames provided by 
      by Dr. Federico Tosi, INAF in "dawn_v10_draft3.tf"

      Changed IDs of DAWN_VIR_VIS_ZERO and DAWN_VIR_IR_ZERO frames from
      -203212 and -203214 to -203221 and -203223 to eliminate confusion
      that may result from the IDs -203212 and -203214 mapping to
      different names in the frames and physical objects domains. Since
      these frames are fixed-offset, class 4 frames that are referred to
      only by name in SPICE API calls and other kernels (e.g. IKs) this
      change should be transparent to the users of this FK.

      Updated star tracker alignments (frames DAWN_ST1 and DAWN_ST2)
      and FC alignments (frames DAWN_FC1 and DAWN_FC2) per MCR-111570.

   Version 0.9 -- July 1, 2010 -- Boris Semenov, NAIF

      Incorporated updated definitions of the DAWN_VIR_VIS, DAWN_VIR_IR, 
      DAWN_VIR_VIS_ZERO, and DAWN_VIR_IR_ZERO frames provided by 
      Eleonora Ammannito, INAF in "VIR_kernel_update_r2.doc".

   Version 0.8 -- October 8, 2009 -- Dr. Federico Tosi, INAF - IFSI
                                     Boris Semenov, NAIF

      Redefined VIR frames to better encapsulate the instrument scan
      mirror motion and optical path misalignments, specifically:

         -  added DAWN_VIR frame (-203200), fixed to the spacecraft, to
            capture the instrument base alignment relative to the spacecraft

         -  added DAWN_VIR_SCAN frame (-203201), CK-based, to capture
            the motion of the mirror

         -  changed the reference frame for DAWN_VIR_VIS and
            DAWN_VIR_IR to DAWN_VIR_SCAN

         -  renamed DAWN_VIR_VIS_SCAN and DAWN_VIR_IR_SCAN to
            DAWN_VIR_VIS_ZERO and DAWN_VIR_IR_ZERO while keeping the
            same IDs (-203212 and -203214) and redefined them to be
            fixed offset frames relative to DAWN_VIR

         -  added DAWN_VIR/-203200 and DAWN_VIR_SCAN/-203201 name/ID
            mappings

   Version 0.7 -- July 3, 2009 -- Boris Semenov, NAIF

      Incorporated Thomas Roatsch's name/ID mappings and individual
      frames for FC filters. Included FC alignments used by DLR to
      process MGA data in the comments.

      Corrected and completed comments in the "Dawn Mission
      NAIF ID Codes -- Summary Section" section.

      Re-worded some of the version section entries for clarity.

   Version 0.6t2 -- June 23, 2009 -- Steven Joy, DSC

      Propagated test version of the DAWN_VIR_VIS_96/-203218 and
      DAWN_VIR_VIS_32/-203219 name/ID mapping. Neither this nor the
      previous updates have still been reviewed by the VIR team.

   Version 0.6t -- April 09, 2009 -- Steven Joy, DSC

      Propagated test version of the DAWN_VIR_VIS_16/-203217 name/ID
      mapping. Neither this nor the previous updates have still been
      reviewed by the VIR team.

   Version 0.6t -- Jan 12, 2009 -- Steven Joy, DSC

      Propagated test version of the DAWN_VIR_VIS_124/-203215 and
      DAWN_VIR_VIS_64/-203216 name/ID mappings from version 0.5t into
      the version 0.6 released by Boris. These updates have still not
      been reviewed by the VIR team.

   Version 0.6 -- January 5, 2009 -- Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.

      Incorporated FC alignments provided by Nick Mastrodemos.

   Version 0.5t -- June 23, 2008 -- Steven Joy, DSC with Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.

      Added definitions for name/ID mappings DAWN_VIR_VIS_124/-203215
      and DAWN_VIR_VIS_64/-203216. These IDs are needed to define
      special FOV for the VIR scan mirror of 128 (64) positions, half
      on either side of the mirror center position.

      Updated HGA frame orientation based on [7].


   Version 0.5 -- January 10, 2008 -- Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.

      Updated HGA frame orientation based on [7].

   Version 0.4 -- May 8, 2007 -- Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.

      Added ID/name mappings for the solar array corners.

   Version 0.3 -- August 7, 2006 -- Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.

      Added star tracker (ST1 and ST2) and flight thruster (FT1, FT2,
      and FT3) frames and name/ID mappings. In this version all STn and
      FTn frames provide on nominal alignment.

   Version 0.2 -- July 27, 2006 -- Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.

      Renamed DAWN_LGA frame and structure name to be DAWN_LGA+Z. Added
      DAWN_LGA+X and DAWN_LGA-Z frames and structure name/IDs.

   Version 0.1 -- May 9, 2005 -- Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.

      Added FC and VIR radiator name-ID mapping keywords. Removed
      definitions of VESTA_FIXED and CERES_FIXED frames; these frames
      are now defined in separate FKs (per request from SOA).

   Version 0.0 -- October 11, 2004 -- Boris Semenov, NAIF

      Preliminary Version. Pending review and approval by Dawn
      instrument teams.


References
========================================================================

   1. ``Frames Required Reading''

   2. ``Kernel Pool Required Reading''

   3. ``C-Kernel Required Reading''

   4. Dawn CDR Presentations.

   5. Dawn Instrument ICD/MICD documents.

   6. DAWN FSDD, CDRL TD-006, 9/27/04

   7. HGA Boresight Direction, e-mail from Ed Swenka, 01/10/08

   8. "Post-Processing Of Framing Camera Alignments"
      (FCalignment_110118.docx), Ian Roundhill, January 18, 2011

   9. "Star Tracker and IRU In-Flight Calibration Test Report.
      DN-DAWN-ACS-026" (DN-DAWN-ACS-026_01.pdf), A. Cross, 01/15/08


Contact Information
========================================================================

   Boris V. Semenov, NAIF/JPL, (818)-354-8136, boris.semenov@jpl.nasa.gov


Implementation Notes
========================================================================

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must ``load'' the kernel, normally during
   program initialization. The SPICE routine FURNSH loads a kernel file
   into the pool as shown below.

      CALL FURNSH ( 'frame_kernel_name; )    -- FORTRAN
      furnsh_c ( "frame_kernel_name" );      -- C
      cspice_furnsh, frame_kernel_name       -- IDL
      cspice_furnsh( 'frame_kernel_name' )   -- MATLAB

   This file was created and may be updated with a text editor or word
   processor.


Dawn Mission NAIF ID Codes -- Summary Section
========================================================================

   The following names and NAIF ID codes are assigned to the Dawn
   spacecraft, its structures and science instruments (the keywords
   implementing these definitions are located in the section "Dawn
   Mission NAIF ID Codes -- Definition Section" at the end of this
   file):

   Dawn Targets:

            VESTA                   2000004 (synonyms: '4 VESTA')
            CERES                   2000001 (synonyms: '1 CERES')

   Dawn Spacecraft and Spacecraft Structures names/IDs:

            DAWN                   -203
            DAWN_SPACECRAFT        -203000
            DAWN_SA+Y              -203010
            DAWN_SA+Y_GIMBAL       -203011
            DAWN_SA+Y_C1           -203012
            DAWN_SA+Y_C2           -203013
            DAWN_SA+Y_C3           -203014
            DAWN_SA+Y_C4           -203015
            DAWN_SA-Y              -203020
            DAWN_SA-Y_GIMBAL       -203021
            DAWN_SA-Y_C1           -203022
            DAWN_SA-Y_C2           -203023
            DAWN_SA-Y_C3           -203024
            DAWN_SA-Y_C4           -203025
            DAWN_HGA               -203040
            DAWN_LGA+Z             -203030
            DAWN_LGA+X             -203050
            DAWN_LGA-Z             -203060
            DAWN_ST1               -203071
            DAWN_ST2               -203072
            DAWN_FT1               -203081
            DAWN_FT2               -203082
            DAWN_FT3               -203083

   FC1 and FC2 names/IDs:

            DAWN_FC1               -203110
            DAWN_FC1_FILTER_1      -203111
            DAWN_FC1_FILTER_2      -203112
            DAWN_FC1_FILTER_3      -203113
            DAWN_FC1_FILTER_4      -203114
            DAWN_FC1_FILTER_5      -203115
            DAWN_FC1_FILTER_6      -203116
            DAWN_FC1_FILTER_7      -203117
            DAWN_FC1_FILTER_8      -203118
            DAWN_FC1_RAD           -203119

            DAWN_FC2               -203120
            DAWN_FC2_FILTER_1      -203121
            DAWN_FC2_FILTER_2      -203122
            DAWN_FC2_FILTER_3      -203123
            DAWN_FC2_FILTER_4      -203124
            DAWN_FC2_FILTER_5      -203125
            DAWN_FC2_FILTER_6      -203126
            DAWN_FC2_FILTER_7      -203127
            DAWN_FC2_FILTER_8      -203128
            DAWN_FC2_RAD           -203129

   VIR names/IDs:

            DAWN_VIR               -203200
            DAWN_VIR_SCAN          -203201
            DAWN_VIR_VIS           -203211
            DAWN_VIR_IR            -203213
            DAWN_VIR_VIS_SCAN      -203212
            DAWN_VIR_IR_SCAN       -203214
            DAWN_VIR_VIS_128       -203215
            DAWN_VIR_VIS_64        -203216
            DAWN_VIR_VIS_16        -203217
            DAWN_VIR_VIS_96        -203218
            DAWN_VIR_VIS_32        -203219
            DAWN_VIR_RAD           -203209

   GRAND names/IDs:

            DAWN_GRAND             -203300


   Vesta and Ceres DSK Surface names/IDs:

      See "Dawn DSK Surface NAIF Names/ID Codes -- Definition Section"
      section at the end the file.


Dawn Frames
========================================================================

   The following Dawn frames are defined in this kernel file:

           Name                  Relative to           Type       NAIF ID
      ======================  ===================  ============   =======

   Dawn Target frames:
   ----------------------------------------------------
      IAU_VESTA               J2000                PCK            10099   (1)
      VESTA_FIXED             J2000                PCK            2000004 (2)
      VESTA_IAU_2000          J2000                DYNAMIC        -203911
      DAWN_CLAUDIA            J2000                DYNAMIC        -203912

      CERES_FIXED             J2000                PCK            2000001

   Dawn Spacecraft and Spacecraft Structures frames:
   ----------------------------------------------------
      DAWN_SPACECRAFT         J2000                CK             -203000
      DAWN_SA+Y               DAWN_SPACECRAFT      CK             -203010
      DAWN_SA-Y               DAWN_SPACECRAFT      CK             -203020
      DAWN_HGA                DAWN_SPACECRAFT      FIXED          -203040
      DAWN_LGA+Z              DAWN_SPACECRAFT      FIXED          -203030
      DAWN_LGA+X              DAWN_SPACECRAFT      FIXED          -203050
      DAWN_LGA-Z              DAWN_SPACECRAFT      FIXED          -203060
      DAWN_ST1                DAWN_SPACECRAFT      FIXED          -203071
      DAWN_ST2                DAWN_SPACECRAFT      FIXED          -203072
      DAWN_FT1                DAWN_SPACECRAFT      FIXED (3)      -203081
      DAWN_FT2                DAWN_SPACECRAFT      FIXED (3)      -203082
      DAWN_FT3                DAWN_SPACECRAFT      FIXED (3)      -203083

   FC1 and FC2 frames:
   -------------------
      DAWN_FC1                DAWN_SPACECRAFT      FIXED          -203110
      DAWN_FC1_FILTER_1       DAWN_FC1             FIXED          -203111
      DAWN_FC1_FILTER_2       DAWN_FC1             FIXED          -203112
      DAWN_FC1_FILTER_3       DAWN_FC1             FIXED          -203113
      DAWN_FC1_FILTER_4       DAWN_FC1             FIXED          -203114
      DAWN_FC1_FILTER_5       DAWN_FC1             FIXED          -203115
      DAWN_FC1_FILTER_6       DAWN_FC1             FIXED          -203116
      DAWN_FC1_FILTER_7       DAWN_FC1             FIXED          -203117
      DAWN_FC1_FILTER_8       DAWN_FC1             FIXED          -203118

      DAWN_FC2                DAWN_SPACECRAFT      CK             -203120
      DAWN_FC2_FILTER_1       DAWN_FC2             FIXED          -203121
      DAWN_FC2_FILTER_2       DAWN_FC2             FIXED          -203122
      DAWN_FC2_FILTER_3       DAWN_FC2             FIXED          -203123
      DAWN_FC2_FILTER_4       DAWN_FC2             FIXED          -203124
      DAWN_FC2_FILTER_5       DAWN_FC2             FIXED          -203125
      DAWN_FC2_FILTER_6       DAWN_FC2             FIXED          -203126
      DAWN_FC2_FILTER_7       DAWN_FC2             FIXED          -203127
      DAWN_FC2_FILTER_8       DAWN_FC2             FIXED          -203128

   VIR frames:
   -----------
      DAWN_VIR                DAWN_SPACECRAFT      FIXED          -203200
      DAWN_VIR_SCAN           DAWN_VIR             CK             -203201
      DAWN_VIR_VIS            DAWN_VIR_SCAN        FIXED          -203211
      DAWN_VIR_IR             DAWN_VIR_SCAN        FIXED          -203213

      DAWN_VIR_VIS_ZERO       DAWN_VIR             FIXED          -203221
      DAWN_VIR_IR_ZERO        DAWN_VIR             FIXED          -203223

   GRAND frames:
   -------------
      DAWN_GRAND              DAWN_SPACECRAFT      FIXED          -203300

   (1) The IAU_VESTA frame is built into the Toolkit starting with the
       Toolkit version N0054 (June 2010).

   (2) The VESTA_FIXED frame is defined in a separate FK. It is essentially
       a synonym of the IAU_VESTA frame.

   (3) Although all flight thrusters are mounted on gimbals and can be
       articulated within a small angular range (+/- 5 degrees during
       flight), in this version of the FK their frames are defined as
       fixed offset frames with respect to the spacecraft frame.


Spacecraft and Spacecraft Structures Frame Tree
========================================================================

   The diagram below shows the frame hierarchy for the Dawn spacecraft
   and its structure frame (not including science instrument frames.)

     "VESTA_IAU_2000"
     ----------------
           ^     
           |<-dynamic
           |
           |  "DAWN_CLAUDIA"
           |  --------------
           |     ^
           |     |<-dynamic
           |     |
           |     |      
           |     |             "J2000" INERTIAL
           +-----------------------------------------------------+
           |                |         |                          |
           |<-pck           |<-pck    |                          |<-pck
           |                |         |                          |
           V                V         |                          V
      "VESTA_FIXED"    "IAU_VESTA"    |                    "CERES_FIXED"
      -------------   -------------   |                    -------------
                                      |
                                      |
                                      |
       "DAWN_ST1" "DAWN_ST2"          |         "DAWN_LGA+X" "DAWN_LGA-Z"
       ---------  ----------          |         ------------ ------------
           ^           ^              |              ^           ^
           |           |              |              |           |
           |<-fixed    |<-fixed       |<-ck          |<-fixed    |<-fixed
           |           |              |              |           |
           |           |              V              |           |
           |           |       "DAWN_SPACECRAFT"     |           |
           +-----------------------------------------------------+
           |           |              .  | | |       |           |
           |<-ck       |<-ck          .  | | |       |<-fixed    |<-fixed
           |           |              .  | | |       |           |
           V           V              .  | | |       V           V
       "DAWN_SA+Y" "DAWN_SA-Y"        .  | | |  "DAWN_LGA+Z" "DAWN_HGA"
       ----------- -----------        .  | | |  ------------ ----------
                                      .  | | |
                                      .  | | |<-fixed
                                      .  | | |
                                      .  | | V
                                      .  | | "DAWN_FT1"
                                      .  | | ----------
                                      .  | |
                                      .  | |<-fixed
                                      .  | |
                                      .  | V
                                      .  | "DAWN_FT2"
                                      .  | ----------
                                      .  |
                                      .  |<-fixed
                                      .  |
                                      .  V
                                      .  "DAWN_FT3"
                                      .  ----------
                                      .
                                      .
                                      V
                Individual instrument frame trees are provided
                      in the other sections of this file


Dawn Target Frames
========================================================================

   This section of the file contains the body-fixed frame definitions
   for the Dawn mission targets -- asteroids 4 Vesta and 1 Ceres.

   All target body-fixed frames are defined according to the standard
   body-fixed frame formation rules:

      -  +Z axis is toward the North pole;

      -  +X axis is toward the prime meridian;

      -  +Y axis completes the right handed frame;

      -  the origin of this frame is at the center of the body.

   Some of these frames are defined as PCK-based frames with the
   orientation computed by evaluating corresponding rotation constants
   provided in the PCK file(s). 

   Other frames are defined as Euler frames with the constants
   equivalent to the PCK constants included directly into the frame
   definition.


Vesta Frames
--------------------------------------

``IAU_VESTA'' Frame

   The IAU_VESTA frame is a PCK-based Vesta body-fixed frame built into
   the SPICE Toolkit starting with the Toolkit version N0064 (June
   2010). Its orientation is computed using Vesta rotation constants
   from the last loaded PCK file containing such constants.
   

``VESTA_FIXED'' Frame

   The VESTA_FIXED frame is a PCK-based Vesta body-fixed frame defined
   in a separate FK (per request from SOA team on May 9, 2005). This
   frame is a synonym of the IAU_VESTA frame. Its orientation is
   computed using Vesta rotation constants from the last loaded PCK
   file containing such constants.


``IAU-2000'' Frame

   The VESTA_IAU_2000 frame represents the ``IAU-2000'' coordinate
   system used by the project prior to its arrival to Vesta as well as
   in numerous pre-DAWN ground observations of Vesta. It is defined as
   an Euler frame mathematically identical to the PCK frame IAU_VESTA
   when the IAU_VESTA frame is used with the rotation constants from
   the DAWN PCK file ``pck00008.tpc''.

   The ``pck00008.tpc'' PCK data defining the ``IAU-2000'' coordinate
   system orientation are:

      BODY2000004_POLE_RA       = (   301.      0.         0.  )
      BODY2000004_POLE_DEC      = (    41.      0.         0.  )
      BODY2000004_PM            = (   292.   1617.332776   0.  )

   According to the section ``Example of an Euler Frame'' of [1] the
   angles for the Euler frame relative to the J2000 frame with the
   reference epoch of J2000 TDB are derived from the standard set of
   text PCK body orientation terms as follows:

      ANGLE_1 = -pi/2 - RA
      ANGLE_2 = -pi/2 + Dec
      ANGLE_3 =       - PM

   The definition below incorporates the angles derived in this way
   from the ``IAU-2000'' coordinate system orientation constants
   above, in degrees with with the PM rate term converted to
   degrees/sec.

   \begindata

      FRAME_VESTA_IAU_2000             =  -203911
      FRAME_-203911_NAME               = 'VESTA_IAU_2000'
      FRAME_-203911_CLASS              =  5
      FRAME_-203911_CLASS_ID           =  -203911
      FRAME_-203911_CENTER             =  2000004
      FRAME_-203911_RELATIVE           = 'J2000'
      FRAME_-203911_DEF_STYLE          = 'PARAMETERIZED'
      FRAME_-203911_FAMILY             = 'EULER'
      FRAME_-203911_EPOCH              =  @2000-JAN-01/12:00:00
      FRAME_-203911_AXES               =  ( 3  1  3 )
      FRAME_-203911_UNITS              = 'DEGREES'
      FRAME_-203911_ANGLE_1_COEFFS     = (  -31.00                           )
      FRAME_-203911_ANGLE_2_COEFFS     = (  -49.00                           )
      FRAME_-203911_ANGLE_3_COEFFS     = ( -292.00, -0.18719129351851851E-01 )

   \begintext


``DAWN-Claudia'' Frame

   The DAWN_CLAUDIA frame represents the ``DAWN-Claudia'' coordinate
   system used by the project during operations at Vesta. It is defined
   as an Euler frame mathematically identical to the PCK frame
   IAU_VESTA when the IAU_VESTA frame is used with the rotation
   constants from the DAWN PCK file ``dawn_vesta_v04.tpc''.

   The ``dawn_vesta_v04.tpc'' PCK data defining the ``DAWN-Claudia''
   coordinate system orientation are:

      BODY2000004_POLE_RA    = ( 309.031    0.0              0.0 )
      BODY2000004_POLE_DEC   = (  42.235    0.0              0.0 )
      BODY2000004_PM         = (  75.39  1617.3329428        0.0 )

   According to the section ``Example of an Euler Frame'' of [1] the
   angles for the Euler frame relative to the J2000 frame with the
   reference epoch of J2000 TDB are derived from the standard set of
   text PCK body orientation terms as follows:

      ANGLE_1 = -pi/2 - RA
      ANGLE_2 = -pi/2 + Dec
      ANGLE_3 =       - PM

   The definition below incorporates the angles derived in this way
   from the ``DAWN-Claudia'' coordinate system orientation constants
   above, in degrees with with the PM rate term converted to
   degrees/sec.

   \begindata

      FRAME_DAWN_CLAUDIA               =  -203912
      FRAME_-203912_NAME               = 'DAWN_CLAUDIA'
      FRAME_-203912_CLASS              =  5
      FRAME_-203912_CLASS_ID           =  -203912
      FRAME_-203912_CENTER             =  2000004
      FRAME_-203912_RELATIVE           = 'J2000'
      FRAME_-203912_DEF_STYLE          = 'PARAMETERIZED'
      FRAME_-203912_FAMILY             = 'EULER'
      FRAME_-203912_EPOCH              =  @2000-JAN-01/12:00:00
      FRAME_-203912_AXES               =  ( 3  1  3 )
      FRAME_-203912_UNITS              = 'DEGREES'
      FRAME_-203912_ANGLE_1_COEFFS     = (  -39.031                          )
      FRAME_-203912_ANGLE_2_COEFFS     = (  -47.765                          )
      FRAME_-203912_ANGLE_3_COEFFS     = (  -75.39, -0.18719131282407406E-01 )

   \begintext


Dawn Spacecraft and Spacecraft Structures Frames
========================================================================

   This section of the file contains the definitions of the spacecraft
   and spacecraft structures frames.


Dawn Spacecraft Frame
--------------------------------------

   The Dawn spacecraft frame is defined as follows:

      -  +Z axis is along the nominal boresight direction of the framing
         cameras;

      -  +X axis is along the nominal boresight direction of the HGA;

      -  +Y axis completes the right-hand frame;

      -  the origin of this frame is the launch vehicle interface point.

   These diagrams illustrate the DAWN_SPACECRAFT frame:


   +X s/c side (HGA side) view:
   ----------------------------
                                    ^
                                    | toward asteroid
                                    |

                               Science Deck
                             ._____________.
   .__  _______________.     |   ._____.   |     .______________  ___.
   |  \ \               \    | .'       `. |    /               \ \  |
   |  / /                \   |/     |     \|   /                / /  |
   |  \ \                 `. .      |      . .'                 \ \  |
   |  / /                 | o|      o      |o |                 / /  |
   |  \ \                 .' .    .' `.    . `.                 \ \  |
   |  / /                /   |\ +Zsc^  `  /|   \                / /  |
   .__\ \_______________/    | `.   |   .' |    \_______________\ \__.
     -Y Solar Array          .___` -|- '___.       +Y Solar Array
                                  / | \
                                  `-o-----> +Ysc
                                  +Xsc
                                                     +Xsc is out of
                                                        the page

   +Z s/c side (science deck side) view:
   -------------------------------------

                             ._____________.
                             |             |
                             |             |
                             |  +Zsc    +Ysc
   o==/ /==================o |      o----->|o==================/ /==o
     -Y Solar Array          |      |      |        +Y Solar Array
                             |      |      |
                             .______|______.
                                 .--V +Xsc
                          HGA  .'       `.
                              /___________\
                                  `.|.'                 +Zsc is out
                                                       of the page


   Since the orientation of the DAWN_SPACECRAFT frame is computed
   on-board, sent down in telemetry, and stored in the s/c CK files, it
   is defined as a CK-based frame.

   \begindata

      FRAME_DAWN_SPACECRAFT            = -203000
      FRAME_-203000_NAME               = 'DAWN_SPACECRAFT'
      FRAME_-203000_CLASS              =  3
      FRAME_-203000_CLASS_ID           = -203000
      FRAME_-203000_CENTER             = -203
      CK_-203000_SCLK                  = -203
      CK_-203000_SPK                   = -203

   \begintext


Dawn Solar Array Frames
--------------------------------------

   Since the Dawn solar arrays can be articulated (each having one
   degree of freedom), the solar Array frames, DAWN_SA+Y and DAWN_SA-Y,
   are defined as CK frames with their orientation given relative to
   the DAWN_SPACECRAFT frame.

   Both array frames are defined as follows (from [5]):

      -  +Y axis is parallel to the longest side of the array,
         positively oriented from the yoke to the end of the wing;

      -  +Z axis is normal to the solar array plane, the solar cells
         facing +Z;

      -  +X axis is defined such that (X,Y,Z) is right handed;

      -  the origin of the frame is located at the yoke geometric
         center.

   The axis of rotation is parallel to the Y axis of the spacecraft and
   solar array frames.

   This diagram illustrates the DAWN_SA+Y and DAWN_SA-Y frames:


   +X s/c side (HGA side) view:
   ----------------------------
                                    ^
                                    | toward asteroid
                                    |

                               Science Deck
                             ._____________. +Xsa+y
   .__  _______________.     |   ._____.   |^    .______________  ___.
   |  \ \               \    | .'       `. ||   /               \ \  |
   |  / /                \   |/     |     \||  /                / /  |
   |  \ \                 `. .      |      .|.'                 \ \  |
   |  / /      +Ysa-y <-----x|      o      |x-----> +Ysa+y      / /  |
   |  \ \                 .'|.    .' `.    . `.                 \ \  |
   |  / /                /  ||\ +Zsc^  `  /|   \                / /  |
   .__\ \_______________/   || `.   |   .' |    \_______________\ \__.
     -Y Solar Array         V.___` -|- '___.       +Y Solar Array
                      +Xsa-y      / | \
                                  `-o-----> +Ysc
                                  +Xsc
                                               +Xsc is out of the page

                                                  +Zsa+y and +Zsa-y
                                                  are into the page

                                                 Active solar cell is
                                              facing away from the viewer


   These sets of keywords define the solar array frames as CK frames:

   \begindata

      FRAME_DAWN_SA+Y                  = -203010
      FRAME_-203010_NAME               = 'DAWN_SA+Y'
      FRAME_-203010_CLASS              =  3
      FRAME_-203010_CLASS_ID           = -203010
      FRAME_-203010_CENTER             = -203
      CK_-203010_SCLK                  = -203
      CK_-203010_SPK                   = -203

      FRAME_DAWN_SA-Y                  = -203020
      FRAME_-203020_NAME               = 'DAWN_SA-Y'
      FRAME_-203020_CLASS              =  3
      FRAME_-203020_CLASS_ID           = -203020
      FRAME_-203020_CENTER             = -203
      CK_-203020_SCLK                  = -203
      CK_-203020_SPK                   = -203

   \begintext


Dawn Low Gain Antenna Frames
--------------------------------------

   DAWN has three Low Gain Antennas (LGAs) [see 6]: LGA+Z pointing along
   the s/c +Z axis, LGA+X pointing along the s/c +X axis, and LGA-Z
   pointing along the s/c -Z axis. All three LGAs are rigidly attached
   to the s/c bus. Therefore, their frames -- DAWN_LGA+Z, DAWN_LGA+X,
   and DAWN_LGA-Z -- are defined as a fixed offset frames with
   orientation given relative to the DAWN_SPACECRAFT frame.

   Each of the DAWN LGA frames - DAWN_LGA+Z, DAWN_LGA+X, and DAWN_LGA-Z
   -- is defined as follows:

      -  +Z axis is in the antenna boresight direction;

      -  +Y axis is nominally parallel to the s/c +Y axis;

      -  +X axis completes the right handed frame;

      -  the origin of the frame is located at the geometric center of
         the LGA outer patch.

   Neither +X nor +Y axis of in each frame is lined up with the antenna
   clock angle reference direction, which was not known at the time
   when this version of the FK was created.

   This diagram illustrates the DAWN LGA frames:

   +Z s/c side view:
   -----------------

                             ._____________.
                             |             |
                             |             |
                             |  +Zsc    +Ysc
   o==/ /==================o |      o---->  o==================/ /==o
   -Y Solar Array          +Zlga+z  |      ^+Xlga-z     +Y Solar Array
                             | o----->     |
                             ._|____|______|
                               | .--V--.   x-----> +Ylga-z
                               |  +Xsc  `. x-----> +Ylga+x
                               V _________\|
                          +Xlga+z `.|.'    |
                                           |
                                   HGA     V
                                         +Zlga+x

                                                   +Zsc and +Zlga+z are out
                                                         of the page

                                                   +Xlga+x and +Zlga-z are
                                                        into the page

   As seen on the diagram the DAWN_LGA+Z frame and the s/c frame are
   nominally co-aligned, the DAWN_LGA+X frame is rotated from the s/c
   frame by +90 degrees about Y, and the DAWN_LGA-Z frame is rotated
   from the s/c frame by 180 degree about Y.

   These sets of keywords define the LGA frames:

   \begindata

      FRAME_DAWN_LGA+Z                 =  -203030
      FRAME_-203030_NAME               = 'DAWN_LGA+Z'
      FRAME_-203030_CLASS              =  4
      FRAME_-203030_CLASS_ID           =  -203030
      FRAME_-203030_CENTER             =  -203
      TKFRAME_-203030_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203030_SPEC             = 'ANGLES'
      TKFRAME_-203030_UNITS            = 'DEGREES'
      TKFRAME_-203030_ANGLES           = (   0.000,  0.000,   0.000 )
      TKFRAME_-203030_AXES             = (   1,      2,       3     )

      FRAME_DAWN_LGA+X                 =  -203050
      FRAME_-203050_NAME               = 'DAWN_LGA+X'
      FRAME_-203050_CLASS              =  4
      FRAME_-203050_CLASS_ID           =  -203050
      FRAME_-203050_CENTER             =  -203
      TKFRAME_-203050_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203050_SPEC             = 'ANGLES'
      TKFRAME_-203050_UNITS            = 'DEGREES'
      TKFRAME_-203050_ANGLES           = (   0.000, -90.00,   0.000 )
      TKFRAME_-203050_AXES             = (   1,       2,      3     )

      FRAME_DAWN_LGA-Z                 =  -203060
      FRAME_-203060_NAME               = 'DAWN_LGA-Z'
      FRAME_-203060_CLASS              =  4
      FRAME_-203060_CLASS_ID           =  -203060
      FRAME_-203060_CENTER             =  -203
      TKFRAME_-203060_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203060_SPEC             = 'ANGLES'
      TKFRAME_-203060_UNITS            = 'DEGREES'
      TKFRAME_-203060_ANGLES           = (   0.000, 180.00,  0.000 )
      TKFRAME_-203060_AXES             = (   1,       2,     3     )

   \begintext


Dawn High Gain Antenna Frame
--------------------------------------

   The Dawn High Gain Antenna is rigidly attached to the +X side of the
   s/c bus. Therefore, the Dawn HGA frame, DAWN_HGA, is defined as a
   fixed offset frame with its orientation given relative to the
   DAWN_SPACECRAFT frame.

   The DAWN_HGA frame is defined as follows:

      -  +Z axis is in the antenna boresight direction;

      -  +Y axis is nominally parallel to the s/c +Y axis;

      -  +X axis completes the right handed frame;

      -  the origin of the frame is located at the geometric center of
         the HGA dish outer rim circle.

   This diagram illustrates the DAWN_HGA frame:

   +Z s/c side (science deck side) view:
   -------------------------------------

                             ._____________.
                             |             |
                             |             |
                             |  +Zsc    +Ysc
   o==/ /==================o |      o----->|o==================/ /==o
     -Y Solar Array          |      |      |        +Y Solar Array
                             |      |      |
                             .______|______.
                                 .--V +Xsc
                          HGA  .'       `.
                              /_____x----->
                                  `.|.'  +Yhga         +Zsc is out
                                    |                  of the page
                                    |
                                    V +Zhga            +Xhga is into
                                                         the page



   As seen on the diagram a single rotation by +90 degrees about +Y is
   needed to align the s/c frame with the HGA frame.

   According to [7] the actual HGA boresight direction in the spacecraft
   frame used in the on-board FSW code and ACS ground tools is:

      AcTable.Boresight.HighGainAntenna.v =
        {0.999999899738598, -0.000219079458599998, -0.000390547031294996}

   The following two rotations are needed to align the spacecraft frame
   with the HGA frame having +Z along this direction: first by
   +90.02237670 degrees about Y, then by +0.01255233 degrees about X.

   This set of keywords defines the HGA frame as a fixed offset frame:

   \begindata

      FRAME_DAWN_HGA                   =  -203040
      FRAME_-203040_NAME               = 'DAWN_HGA'
      FRAME_-203040_CLASS              =  4
      FRAME_-203040_CLASS_ID           =  -203040
      FRAME_-203040_CENTER             =  -203
      TKFRAME_-203040_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203040_SPEC             = 'ANGLES'
      TKFRAME_-203040_UNITS            = 'DEGREES'
      TKFRAME_-203040_ANGLES           = ( -90.02237670, -0.01255233,   0.0 )
      TKFRAME_-203040_AXES             = (   2,           1,            3   )

   \begintext


Dawn Star Tracker Frames
--------------------------------------

   The star trackers (ST1 and ST2) are rigidly attached to the +Z side
   of the s/c bus. Therefore, the star tracker frames, DAWN_ST1 and
   DAWN_ST2, are defined as fixed offset frames with their orientation
   given relative to the DAWN_SPACECRAFT frame.

   The star tracker frames are defined as follows:

      -  +Z axis is in the star tracker boresight direction;

      -  +X axis nominally points along the s/c +Z axis;

      -  +Y axis completes the right handed frame;

      -  the origin of the frame is located at the star tracker focal
         point.

   This diagram illustrates the star tracker frames:

   +Z s/c side (science deck side) view:
   -------------------------------------


                       \   30 deg   |   30 deg   /
                        \<--------->|<--------->/
                         \          |          /

                          +Zst2            +Zst1
                            ^      +Yst1    ^
                             \        ^    /
                              \     .'    /
                             ._\__.'_____/_.
                          +Xst2 o'      o +Xst1
                             |           `.|
                             |  +Zsc       `.  +Yst2
   o==/ /==================o |      o----->|o`>================/ /==o
     -Y Solar Array          |      |   +Ysc        +Y Solar Array
                             |      |      |
                             .______|______.
                                 .--V +Xsc
                          HGA  .'       `.
                              /___________\
                                  `.|.'           +Zsc, +Xst1 and +Xst2 are
                                                      out of the page


   As seen on the diagram two rotations -- first by -90 degrees about Y
   and second by -30 degree (for ST1) or +30 degrees (for ST2) about X
   -- are needed to align the s/c frame with the star tracker frames.

   This nominal alignment was provided in the officially released 
   FK versions 0.1 to 0.8.

   Per MCR-111570, 03/18/11, the nominal alignments were replaced with
   the following actual alignments provided in [8] and [9].

      Star Tracker 1 wrt Spacecraft Frame: R_ST1_2,SF in the "Inputs"
      section of [8]:

         -0.00167591111459615  -0.00865834105070781  -0.999961111469934
          0.497384915940654     0.867489831468208    -0.00834492024941222
          0.867528349229375    -0.497379558716992     0.00285269238533509
 
      Star Tracker 2 wrt Spacecraft Frame: QBdyToSta[1] and
      DcmBdyToSta[1] on page 24 in the "Appendix A" of [9]:

          0.003428012232148947 -0.002975996167250916 -0.9999896960363887
         -0.5035582110269716    0.8639506124714249   -0.004297361698267110
          0.8639544992876952    0.5035677537899567    0.001463045961520582

   Because these alignments are for the ACS star tracker frames with
   the boresight along -Z they had to be rotated by 180 degrees about Y
   to agree with the frame definition described above (boresight along
   +Z). Such rotation is equivalent to flipping the sign for each
   component in the top and bottom rows of the matrices. These
   alignments were incorporated into the FK starting with version 1.0
   (March 2011).

   The sets of keywords below defines the star tracker frames.

   \begindata

      FRAME_DAWN_ST1                   =  -203071
      FRAME_-203071_NAME               = 'DAWN_ST1'
      FRAME_-203071_CLASS              =  4
      FRAME_-203071_CLASS_ID           =  -203071
      FRAME_-203071_CENTER             =  -203
      TKFRAME_-203071_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203071_SPEC             = 'MATRIX'
      TKFRAME_-203071_MATRIX           = (

          0.00167591111459615   0.00865834105070781   0.999961111469934
          0.497384915940654     0.867489831468208    -0.00834492024941222
         -0.867528349229375     0.497379558716992    -0.00285269238533509

                                         )


      FRAME_DAWN_ST2                   =  -203072
      FRAME_-203072_NAME               = 'DAWN_ST2'
      FRAME_-203072_CLASS              =  4
      FRAME_-203072_CLASS_ID           =  -203072
      FRAME_-203072_CENTER             =  -203
      TKFRAME_-203072_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203072_SPEC             = 'MATRIX'
      TKFRAME_-203072_MATRIX           = (

         -0.003428012232148947  0.002975996167250916  0.9999896960363887
         -0.5035582110269716    0.8639506124714249   -0.004297361698267110
         -0.8639544992876952   -0.5035677537899567   -0.001463045961520582

                                         )

   \begintext


Dawn Flight Thruster Frames
--------------------------------------

   Although the flight thrusters (FT1, FT2, and FT3) are mounted on
   gimbals and can be articulated within a small angular range (+/- 5
   degrees during flight), in this version of the FK their frames --
   DAWN_FT1, DAWN_FT2, and DAWN_FT3 -- are defined as fixed offset
   frames with respect to the spacecraft frame.

   Each of the flight thruster frames is defined in the same way as
   follows:

      -  +Z axis is in the thrust direction;

      -  +Y axis nominally points along the s/c +Y axis;

      -  +X axis completes the right handed frame;

      -  the origin of the frame is located at the geometric center of
         the thruster nozzle outer rim circle.

   This diagram illustrates the flight thruster frames:

   +Y s/c side view:
   -----------------
                               Science Deck
                             ._____________.
                        .    |      .      |
                        |`.  |      |+Y solar array
                        |  \ |      |      |
                      .'|   .|      |      |
                      --|    |      o      |
                      `.|   '|      |      |     .> +Xft1
                        |  / |  +Zsc^      |   .'
                        |.'  |      |      | .'
                             o______|______o'
                            / `.  / | \     \
                           /  <-----o----->  \
                          / +Xsc  `>|  +Xft3  \
                         V     +Xft1|          V +Zft1
                     +Zft2          |
                                    V +Zft3

                      /    48 deg   |   48 deg    \     +Ysc, +Yft1, +Yft2,
                     /<------------>|<------------>\   and +Yft3 are out of
                    /               |               \       the page


   As seen on the diagram one rotation -- about +Y by -132 degrees for
   FT1, +132 degrees for FT2, or 180 degrees for FT3 -- is needed to
   align the s/c frame with the individual thruster frame.

   The sets of keywords below defines the flight thruster frames:

   \begindata

      FRAME_DAWN_FT1                   =  -203081
      FRAME_-203081_NAME               = 'DAWN_FT1'
      FRAME_-203081_CLASS              =  4
      FRAME_-203081_CLASS_ID           =  -203081
      FRAME_-203081_CENTER             =  -203
      TKFRAME_-203081_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203081_SPEC             = 'ANGLES'
      TKFRAME_-203081_UNITS            = 'DEGREES'
      TKFRAME_-203081_ANGLES           = (   0.000,  132.000,   0.000 )
      TKFRAME_-203081_AXES             = (   3,        2,       1     )

      FRAME_DAWN_FT2                   =  -203082
      FRAME_-203082_NAME               = 'DAWN_FT2'
      FRAME_-203082_CLASS              =  4
      FRAME_-203082_CLASS_ID           =  -203082
      FRAME_-203082_CENTER             =  -203
      TKFRAME_-203082_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203082_SPEC             = 'ANGLES'
      TKFRAME_-203082_UNITS            = 'DEGREES'
      TKFRAME_-203082_ANGLES           = (   0.000, -132.000,   0.000 )
      TKFRAME_-203082_AXES             = (   3,        2,       1     )

      FRAME_DAWN_FT3                   =  -203083
      FRAME_-203083_NAME               = 'DAWN_FT3'
      FRAME_-203083_CLASS              =  4
      FRAME_-203083_CLASS_ID           =  -203083
      FRAME_-203083_CENTER             =  -203
      TKFRAME_-203083_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203083_SPEC             = 'ANGLES'
      TKFRAME_-203083_UNITS            = 'DEGREES'
      TKFRAME_-203083_ANGLES           = (   0.000,  180.000,   0.000 )
      TKFRAME_-203083_AXES             = (   3,        2,       1     )

   \begintext


FC1 and FC2 Frames
========================================================================

   This section of the file contains the definitions of the Framing
   Camera 1 (FC1) and Framing Camera 2 (FC2) frames.


FC1 and FC2 Frame Tree
--------------------------------------

   The diagram below shows the FC1 and FC2 frame hierarchy.

     "VESTA_IAU_2000"
     ----------------
           ^     
           |<-dynamic
           |
           |  "DAWN_CLAUDIA"
           |  --------------
           |     ^
           |     |<-dynamic
           |     |
           |     |      
           |     |             "J2000" INERTIAL
           +-----------------------------------------------------+
           |                |         |                          |
           |<-pck           |<-pck    |                          |<-pck
           |                |         |                          |
           V                V         |                          V
      "VESTA_FIXED"    "IAU_VESTA"    |                    "CERES_FIXED"
      -------------   -------------   |                    -------------
                                      |
                                      |<-ck
                                      |
                                      V
                               "DAWN_SPACECRAFT"
           +-----------------------------------------------------+
           |                                                     |
           |<-fixed                                              |<-ck
           |                                                     |
           V                                                     V
       "DAWN_FC1"                                            "DAWN_FC2"
       ----------                                            ----------
           |                                                     |
           |<-fixed                                              |<-fixed
           |                                                     |
           V                                                     V
       "DAWN_FC1_FILTER_[1..8]"                "DAWN_FC2_FILTER_[1..8]"
       ------------------------                ------------------------


FC1 and FC2 Frames
--------------------------------------

   The Framing Cameras 1 and 2 camera frames -- DAWN_FC1 and DAWN_FC2
   -- are defined as follows:

      -  +Z axis points along the camera boresight;

      -  +X axis is parallel to the apparent image lines; it is
         nominally co-aligned with the s/c +X axis;

      -  +Y axis completes the right handed frame; it is nominally
         parallel the to the apparent image columns and co-aligned with
         the s/c +Y axis;

      -  the origin of the frame is located at the camera focal point.

   The Framing Cameras 1 and 2 filter frames -- DAWN_FC1_FILTER_[1..8]
   and DAWN_FC2_FILTER_[1..8] -- are defined to be nominally co-aligned
   with the corresponding camera frames.

   This diagram illustrates the FC1 and FC2 camera frames:


   +Z s/c side (science deck side) view:
   -------------------------------------

                                       +Yfc1
                             ._____________.
                             |    o---o->---> +Yfc2
                             |    |   |    |
                             |    |   |    |+Ysc
   o==/ /==================o |    | o-|--->|o==================/ /==o
     -Y Solar Array          |    V | V    |        +Y Solar Array
                             |+Xfc1 | +Xfc2
                             .______|______.
                                 .--V +Xsc
                          HGA  .'       `.           +Zsc, +Zfc1, and
                              /___________\            +Zfc2 are out
                                  `.|.'                 of the page


   Nominally, the FC1 and FC2 frames are co-aligned with the s/c frame.
   The following nominal FC1 and FC2 frame definitions were provided in
   the FK from October 11, 2004 to January 5, 2009 (ver. 0.0 to 0.5):

      FRAME_DAWN_FC1                   =  -203110
      FRAME_-203110_NAME               = 'DAWN_FC1'
      FRAME_-203110_CLASS              =  4
      FRAME_-203110_CLASS_ID           =  -203110
      FRAME_-203110_CENTER             =  -203
      TKFRAME_-203110_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203110_SPEC             = 'ANGLES'
      TKFRAME_-203110_UNITS            = 'DEGREES'
      TKFRAME_-203110_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203110_AXES             = ( 1,   2,   3   )

      FRAME_DAWN_FC2                   =  -203120
      FRAME_-203120_NAME               = 'DAWN_FC2'
      FRAME_-203120_CLASS              =  4
      FRAME_-203120_CLASS_ID           =  -203120
      FRAME_-203120_CENTER             =  -203
      TKFRAME_-203120_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203120_SPEC             = 'ANGLES'
      TKFRAME_-203120_UNITS            = 'DEGREES'
      TKFRAME_-203120_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203120_AXES             = ( 1,   2,   3   )

   These frame alignments were used by FC team at DLR, Germany to
   process February 2009 Mars Gravity Assist data;

      FRAME_DAWN_FC1                   =  -203110
      FRAME_-203110_NAME               = 'DAWN_FC1'
      FRAME_-203110_CLASS              =  4
      FRAME_-203110_CLASS_ID           =  -203110
      FRAME_-203110_CENTER             =  -203
      TKFRAME_-203110_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203110_SPEC             = 'ANGLES'
      TKFRAME_-203110_UNITS            = 'DEGREES'
      TKFRAME_-203110_ANGLES          = ( -0.00302298, 0.07040644, 0.06472022 )
      TKFRAME_-203110_AXES             = ( 1,   2,   3   )

      FRAME_DAWN_FC2                   =  -203120
      FRAME_-203120_NAME               = 'DAWN_FC2'
      FRAME_-203120_CLASS              =  4
      FRAME_-203120_CLASS_ID           =  -203120
      FRAME_-203120_CENTER             =  -203
      TKFRAME_-203120_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203120_SPEC             = 'ANGLES'
      TKFRAME_-203120_UNITS            = 'DEGREES'
      TKFRAME_-203120_ANGLES          = ( -0.07360254, 0.19136667, 0.08369507 )
      TKFRAME_-203120_AXES             = ( 1,   2,   3   )

   The two matrices below represent the actual orientation of the FC1
   and FC2 frames. These matrices rotate vectors from the s/c frame to
   the corresponding camera frames. These matrices were determined by
   Nick Mastrodemos, DAWN Optical Navigation Team, using the Functional,
   Performance and Calibration test images taken in 2007-2008. They
   were provided to NAIF on January 5, 2009 and were incorporated into
   the FK starting with version 0.6 (January 2009):

      FC1:

          0.999998669076022  -0.001146506533836   0.001160762229150
          0.001146467511555   0.999999342218256   0.000034282638084
         -0.001160800770890  -0.000032951816272   0.999999325727647


      FC2:

          0.999993562497650  -0.001478972971767   0.003269189839768
          0.001475079784942   0.999998200488307   0.001192962604301
         -0.003270948316271  -0.001188132608755   0.999993944600674

   Per MCR-111570, 03/18/11, the actual alignments above were replaced
   with the following updated values provided in [8] and incorporated
   in the FK with version 1.0 (March 2011):

      FC1:

          0.99999903640145     -0.00088624511873564   0.001068487703551
          0.000886539816130183  0.999999569071098    -0.000275356733179061
         -0.0010682432039777    0.000276303753185878  0.999999391254863

      FC2:

          0.999994211374513    -0.00121806017718856   0.00317701858659441
          0.00121525960845842   0.999998871426382     0.000883293086781172
         -0.00317809089965586  -0.000879427042947794  0.999994563157032

   The FC1 matrix is provided in the frame definition below.

   Per MCR-117124, 04/01/15, the FC2 alignment was adjusted based on an
   analysis of 196 pictures from Ceres Opnavs 1-5, taken 01/13/15 to
   03/02/15. The new alignment is:

      FC2:

         0.999994729254062      -0.00121791297955331    0.00300965983081698
         0.00121554146625864     0.999998949394655      0.000789674082497601
        -0.00301061841759422    -0.000786011525583021   0.999995159168256

   Since two different FC2 alignments -- one applicable to the Vesta
   mission time period and the other to the Ceres mission time period
   -- cannot be incorporated into a single fixed-offset frame, in the
   FK version 1.3 (April 2015) the DAWN_FC2 frame class was changed
   from a fixed-offset (class 4) to CK based (class 3) and two
   different FC2 alignments were stored in a CK file for ID -203120.

   \begindata

      FRAME_DAWN_FC1                   =  -203110
      FRAME_-203110_NAME               = 'DAWN_FC1'
      FRAME_-203110_CLASS              =  4
      FRAME_-203110_CLASS_ID           =  -203110
      FRAME_-203110_CENTER             =  -203
      TKFRAME_-203110_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203110_SPEC             = 'MATRIX'
      TKFRAME_-203110_MATRIX           = (

          0.99999903640145     -0.00088624511873564   0.001068487703551
          0.000886539816130183  0.999999569071098    -0.000275356733179061
         -0.0010682432039777    0.000276303753185878  0.999999391254863

                                         )

      FRAME_DAWN_FC2                   =  -203120
      FRAME_-203120_NAME               = 'DAWN_FC2'
      FRAME_-203120_CLASS              =  3
      FRAME_-203120_CLASS_ID           = -203120
      FRAME_-203120_CENTER             = -203
      CK_-203120_SCLK                  = -203
      CK_-203120_SPK                   = -203

   \begintext

   The keywords below define the FC filter frames to be co-aligned with
   the corresponding camera frames.

   \begindata

      FRAME_DAWN_FC1_FILTER_1       = -203111
      FRAME_-203111_NAME            = 'DAWN_FC1_FILTER_1'
      FRAME_-203111_CLASS           = 4
      FRAME_-203111_CLASS_ID        = -203111
      FRAME_-203111_CENTER          = -203
      TKFRAME_-203111_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203111_SPEC          = 'ANGLES'
      TKFRAME_-203111_UNITS         = 'DEGREES'
      TKFRAME_-203111_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203111_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC1_FILTER_2       = -203112
      FRAME_-203112_NAME            = 'DAWN_FC1_FILTER_2'
      FRAME_-203112_CLASS           = 4
      FRAME_-203112_CLASS_ID        = -203112
      FRAME_-203112_CENTER          = -203
      TKFRAME_-203112_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203112_SPEC          = 'ANGLES'
      TKFRAME_-203112_UNITS         = 'DEGREES'
      TKFRAME_-203112_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203112_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC1_FILTER_3       = -203113
      FRAME_-203113_NAME            = 'DAWN_FC1_FILTER_3'
      FRAME_-203113_CLASS           = 4
      FRAME_-203113_CLASS_ID        = -203113
      FRAME_-203113_CENTER          = -203
      TKFRAME_-203113_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203113_SPEC          = 'ANGLES'
      TKFRAME_-203113_UNITS         = 'DEGREES'
      TKFRAME_-203113_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203113_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC1_FILTER_4       = -203114
      FRAME_-203114_NAME            = 'DAWN_FC1_FILTER_4'
      FRAME_-203114_CLASS           = 4
      FRAME_-203114_CLASS_ID        = -203114
      FRAME_-203114_CENTER          = -203
      TKFRAME_-203114_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203114_SPEC          = 'ANGLES'
      TKFRAME_-203114_UNITS         = 'DEGREES'
      TKFRAME_-203114_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203114_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC1_FILTER_5       = -203115
      FRAME_-203115_NAME            = 'DAWN_FC1_FILTER_5'
      FRAME_-203115_CLASS           = 4
      FRAME_-203115_CLASS_ID        = -203115
      FRAME_-203115_CENTER          = -203
      TKFRAME_-203115_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203115_SPEC          = 'ANGLES'
      TKFRAME_-203115_UNITS         = 'DEGREES'
      TKFRAME_-203115_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203115_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC1_FILTER_6       = -203116
      FRAME_-203116_NAME            = 'DAWN_FC1_FILTER_6'
      FRAME_-203116_CLASS           = 4
      FRAME_-203116_CLASS_ID        = -203116
      FRAME_-203116_CENTER          = -203
      TKFRAME_-203116_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203116_SPEC          = 'ANGLES'
      TKFRAME_-203116_UNITS         = 'DEGREES'
      TKFRAME_-203116_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203116_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC1_FILTER_7       = -203117
      FRAME_-203117_NAME            = 'DAWN_FC1_FILTER_7'
      FRAME_-203117_CLASS           = 4
      FRAME_-203117_CLASS_ID        = -203117
      FRAME_-203117_CENTER          = -203
      TKFRAME_-203117_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203117_SPEC          = 'ANGLES'
      TKFRAME_-203117_UNITS         = 'DEGREES'
      TKFRAME_-203117_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203117_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC1_FILTER_8       = -203118
      FRAME_-203118_NAME            = 'DAWN_FC1_FILTER_8'
      FRAME_-203118_CLASS           = 4
      FRAME_-203118_CLASS_ID        = -203118
      FRAME_-203118_CENTER          = -203
      TKFRAME_-203118_RELATIVE      = 'DAWN_FC1'
      TKFRAME_-203118_SPEC          = 'ANGLES'
      TKFRAME_-203118_UNITS         = 'DEGREES'
      TKFRAME_-203118_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203118_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_1       = -203121
      FRAME_-203121_NAME            = 'DAWN_FC2_FILTER_1'
      FRAME_-203121_CLASS           = 4
      FRAME_-203121_CLASS_ID        = -203121
      FRAME_-203121_CENTER          = -203
      TKFRAME_-203121_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203121_SPEC          = 'ANGLES'
      TKFRAME_-203121_UNITS         = 'DEGREES'
      TKFRAME_-203121_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203121_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_2       = -203122
      FRAME_-203122_NAME            = 'DAWN_FC2_FILTER_2'
      FRAME_-203122_CLASS           = 4
      FRAME_-203122_CLASS_ID        = -203122
      FRAME_-203122_CENTER          = -203
      TKFRAME_-203122_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203122_SPEC          = 'ANGLES'
      TKFRAME_-203122_UNITS         = 'DEGREES'
      TKFRAME_-203122_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203122_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_3       = -203123
      FRAME_-203123_NAME            = 'DAWN_FC2_FILTER_3'
      FRAME_-203123_CLASS           = 4
      FRAME_-203123_CLASS_ID        = -203123
      FRAME_-203123_CENTER          = -203
      TKFRAME_-203123_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203123_SPEC          = 'ANGLES'
      TKFRAME_-203123_UNITS         = 'DEGREES'
      TKFRAME_-203123_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203123_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_4       = -203124
      FRAME_-203124_NAME            = 'DAWN_FC2_FILTER_4'
      FRAME_-203124_CLASS           = 4
      FRAME_-203124_CLASS_ID        = -203124
      FRAME_-203124_CENTER          = -203
      TKFRAME_-203124_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203124_SPEC          = 'ANGLES'
      TKFRAME_-203124_UNITS         = 'DEGREES'
      TKFRAME_-203124_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203124_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_5       = -203125
      FRAME_-203125_NAME            = 'DAWN_FC2_FILTER_5'
      FRAME_-203125_CLASS           = 4
      FRAME_-203125_CLASS_ID        = -203125
      FRAME_-203125_CENTER          = -203
      TKFRAME_-203125_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203125_SPEC          = 'ANGLES'
      TKFRAME_-203125_UNITS         = 'DEGREES'
      TKFRAME_-203125_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203125_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_6       = -203126
      FRAME_-203126_NAME            = 'DAWN_FC2_FILTER_6'
      FRAME_-203126_CLASS           = 4
      FRAME_-203126_CLASS_ID        = -203126
      FRAME_-203126_CENTER          = -203
      TKFRAME_-203126_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203126_SPEC          = 'ANGLES'
      TKFRAME_-203126_UNITS         = 'DEGREES'
      TKFRAME_-203126_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203126_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_7       = -203127
      FRAME_-203127_NAME            = 'DAWN_FC2_FILTER_7'
      FRAME_-203127_CLASS           = 4
      FRAME_-203127_CLASS_ID        = -203127
      FRAME_-203127_CENTER          = -203
      TKFRAME_-203127_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203127_SPEC          = 'ANGLES'
      TKFRAME_-203127_UNITS         = 'DEGREES'
      TKFRAME_-203127_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203127_AXES          = ( 1,   2,   3   )

      FRAME_DAWN_FC2_FILTER_8       = -203128
      FRAME_-203128_NAME            = 'DAWN_FC2_FILTER_8'
      FRAME_-203128_CLASS           = 4
      FRAME_-203128_CLASS_ID        = -203128
      FRAME_-203128_CENTER          = -203
      TKFRAME_-203128_RELATIVE      = 'DAWN_FC2'
      TKFRAME_-203128_SPEC          = 'ANGLES'
      TKFRAME_-203128_UNITS         = 'DEGREES'
      TKFRAME_-203128_ANGLES        = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203128_AXES          = ( 1,   2,   3   )

   \begintext


VIR frames:
========================================================================

   This section of the file contains the definitions of the VIR frames.


VIR Frame Tree
--------------------------------------

   The diagram below shows the VIR frame hierarchy.

     "VESTA_IAU_2000"
     ----------------
           ^     
           |<-dynamic
           |
           |  "DAWN_CLAUDIA"
           |  --------------
           |     ^
           |     |<-dynamic
           |     |
           |     |      
           |     |             "J2000" INERTIAL
           +-----------------------------------------------------+
           |                |         |                          |
           |<-pck           |<-pck    |                          |<-pck
           |                |         |                          |
           V                V         |                          V
      "VESTA_FIXED"    "IAU_VESTA"    |                    "CERES_FIXED"
      -------------   -------------   |                    -------------
                                      |
                                      |<-ck
                                      |
                                      V
                               "DAWN_SPACECRAFT"
                               -----------------
                                      |
                                      |<-fixed
                                      |
                                      V
                                  DAWN_VIR
           +-----------------------------------------------------+
           |                          |                          |
           |<-fixed                   |<-ck                      |<-fixed
           |                          |                          |
           V                          V                          V
      "DAWN_VIR_VIS_ZERO"       DAWN_VIR_SCAN          "DAWN_VIR_IR_ZERO"
      -------------------    +-----------------+       ------------------
                             |                 |
                             |<-fixed          |<-fixed
                             |                 |
                             V                 V
                      "DAWN_VIR_VIS"     "DAWN_VIR_IR"
                      --------------     -------------


VIR Frames
--------------------------------------

   The following frames are defined for VIR instrument:

      -  the instrument base frame -- DAWN_VIR (ID -203200) -- is
         defined as a fixed offset frame relative to the
         DAWN_SPACECRAFT frame. It is intended to capture the
         instrument base and scan mirror rotation axis alignment
         relative to the spacecraft.
 
      -  the reflected scan mirror view direction frame --
         DAWN_VIR_SCAN (ID -203201) -- is defined as a CK-based frame
         with its orientation provided in CK files relative to the
         DAWN_VIR frame. It is intended to capture change in the
         instrument view direction due to the scan mirror motion.
 
      -  the reflected instrument detector view frames -- DAWN_VIR_VIS
         (ID -203211) and DAWN_VIR_IR (ID -203213) -- are defined as
         fixed offset frames relative to the DAWN_VIR_SCAN frame. They
         are intended to capture reflected instrument boresights and
         detector misalignments at any scan mirror position.
 
      -  the non-moving instrument detector view frames -- DAWN_VIR_VIS_ZERO
         (ID -203221) and DAWN_VIR_IR_ZERO (ID -203223) -- are defined as
         fixed offset frames relative to the DAWN_VIR frame. They are
         intended to capture instrument boresights and detector
         misalignments at the "0" scan mirror position and are used for
         defining FOVs corresponding the full (256-step) and partial
         (128-, 64-, 16-, 96-, and 32-step) scan mirror sweeps.
 
   The +Y axes of the base and scan mirror frames (DAWN_VIR and
   DAWN_VIR_SCAN) are co-aligned with the scan mirror rotation axis.

   The +Z axes of the reflected and non-moving instrument detector view
   frames are in the direction of the instrument boresight at any (for
   DAWN_VIR_VIS and DAWN_VIR_IR) or the "0" (for DAWN_VIR_VIS_ZERO and
   DAWN_VIR_IR_ZERO) scan mirror position. The +Y axes of these frames
   are in the direction of spatial resolution and the +X axes complete
   the right-handed frames.
   
   Nominally, at the "0" scan mirror position all VIR frames are co-aligned 
   with each other and with the s/c frame as shown on this diagram:


   +Z s/c side (science deck side) view:
   -------------------------------------

                                   o-----> +Yvir*
                             ._____|_______.
                             |     |       |
                             +Xvir*|       |
                             |     V      +Ysc
   o==/ /==================o |      o----->|o==================/ /==o
     -Y Solar Array          |      |      |        +Y Solar Array
                             |      |      |
                             .______|______.
                                 .--V +Xsc
                          HGA  .'       `.
                              /___________\
                                  `.|.'                 +Zsc and +Zvir*
                                                      are out of the page

   The keywords below define the VIR frames. In this version of the FK 
   all misalignments are set be zero.

   Preliminary alignment data for VIR-VIS indicate that the instrument 
   boresight (center pixel view direction) is tilted with respect to the 
   spacecraft +Z axis by 1.50 milliradians toward -Y axis and 1.25 
   milliradians toward +X. This misalignment is incorporated as a two 
   rotations, first by +1.50 milliradians about +X axis and then by 
   +1.25 milliradians about +Y axis, into the definitions of the 
   DAWN_VIR_VIS and DAWN_VIR_VIS_ZERO frames.

   Preliminary alignment data for VIR-IR indicate that the instrument 
   boresight (center pixel view direction) is tilted with respect to the 
   spacecraft +Z axis by 0.75 milliradians toward -Y axis and 1.50 
   milliradians toward +X. This misalignment is incorporated as a two 
   rotations, first by +0.75 milliradians about +X axis and then by 
   +1.50 milliradians about +Y axis, into the definitions of the
   DAWN_VIR_IR and DAWN_VIR_IR_ZERO frames.

   \begindata

      FRAME_DAWN_VIR                   =  -203200
      FRAME_-203200_NAME               = 'DAWN_VIR'
      FRAME_-203200_CLASS              =  4
      FRAME_-203200_CLASS_ID           =  -203200
      FRAME_-203200_CENTER             =  -203
      TKFRAME_-203200_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203200_SPEC             = 'ANGLES'
      TKFRAME_-203200_UNITS            = 'DEGREES'
      TKFRAME_-203200_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203200_AXES             = ( 1,   2,   3   )

      FRAME_DAWN_VIR_SCAN              = -203201
      FRAME_-203201_NAME               = 'DAWN_VIR_SCAN'
      FRAME_-203201_CLASS              =  3
      FRAME_-203201_CLASS_ID           = -203201
      FRAME_-203201_CENTER             = -203
      CK_-203201_SCLK                  = -203
      CK_-203201_SPK                   = -203

      FRAME_DAWN_VIR_VIS               =  -203211
      FRAME_-203211_NAME               = 'DAWN_VIR_VIS'
      FRAME_-203211_CLASS              =  4
      FRAME_-203211_CLASS_ID           =  -203211
      FRAME_-203211_CENTER             =  -203
      TKFRAME_-203211_RELATIVE         = 'DAWN_VIR_SCAN'
      TKFRAME_-203211_SPEC             = 'ANGLES'
      TKFRAME_-203211_UNITS            = 'DEGREES'
      TKFRAME_-203211_ANGLES           = (-0.085943669, -0.071619724, 0.0)
      TKFRAME_-203211_AXES             = ( 1,   2,   3   )

      FRAME_DAWN_VIR_IR                =  -203213
      FRAME_-203213_NAME               = 'DAWN_VIR_IR'
      FRAME_-203213_CLASS              =  4
      FRAME_-203213_CLASS_ID           =  -203213
      FRAME_-203213_CENTER             =  -203
      TKFRAME_-203213_RELATIVE         = 'DAWN_VIR_SCAN'
      TKFRAME_-203213_SPEC             = 'ANGLES'
      TKFRAME_-203213_UNITS            = 'DEGREES'
      TKFRAME_-203213_ANGLES           = (-0.042971835, -0.085943669, 0.0)
      TKFRAME_-203213_AXES             = ( 1,   2,   3   )

      FRAME_DAWN_VIR_VIS_ZERO          = -203221
      FRAME_-203221_NAME               = 'DAWN_VIR_VIS_ZERO'
      FRAME_-203221_CLASS              =  4
      FRAME_-203221_CLASS_ID           =  -203221
      FRAME_-203221_CENTER             =  -203
      TKFRAME_-203221_RELATIVE         = 'DAWN_VIR'
      TKFRAME_-203221_SPEC             = 'ANGLES'
      TKFRAME_-203221_UNITS            = 'DEGREES'
      TKFRAME_-203221_ANGLES           = (-0.085943669, -0.071619724, 0.0)
      TKFRAME_-203221_AXES             = ( 1,   2,   3   )

      FRAME_DAWN_VIR_IR_ZERO           = -203223
      FRAME_-203223_NAME               = 'DAWN_VIR_IR_ZERO'
      FRAME_-203223_CLASS              =  4
      FRAME_-203223_CLASS_ID           =  -203223
      FRAME_-203223_CENTER             =  -203
      TKFRAME_-203223_RELATIVE         = 'DAWN_VIR'
      TKFRAME_-203223_SPEC             = 'ANGLES'
      TKFRAME_-203223_UNITS            = 'DEGREES'
      TKFRAME_-203223_ANGLES           = (-0.042971835, -0.085943669, 0.0)
      TKFRAME_-203223_AXES             = ( 1,   2,   3   )

   \begintext


GRAND frames
========================================================================

   This section of the file contains the definitions of the GRAND
   frames.


GRAND Frame Tree
--------------------------------------

   The diagram below shows the GRAND frame hierarchy.

     "VESTA_IAU_2000"
     ----------------
           ^     
           |<-dynamic
           |
           |  "DAWN_CLAUDIA"
           |  --------------
           |     ^
           |     |<-dynamic
           |     |
           |     |      
           |     |             "J2000" INERTIAL
           +-----------------------------------------------------+
           |                |         |                          |
           |<-pck           |<-pck    |                          |<-pck
           |                |         |                          |
           V                V         |                          V
      "VESTA_FIXED"    "IAU_VESTA"    |                    "CERES_FIXED"
      -------------   -------------   |                    -------------
                                      |
                                      |<-ck
                                      |
                                      V
                               "DAWN_SPACECRAFT"
           +-----------------------------------------------------+
           |
           |<-fixed
           |
           V
     "DAWN_GRAND"
     ------------


GRAND Frame
--------------------------------------

   The GRAND frame is defined as follows:

      -  +Z axis is nominally co-aligned with the s/c +Z axis;

      -  +Y axis is nominally co-aligned with the s/c +Y axis;

      -  +X axis completes the right handed frame; it is nominally
         co-aligned with the s/c +X axis;

      -  the origin of the frame is located at the instrument "FOV"
         vertex.

   This diagram illustrates the GRAND frame:


   +Z s/c side (science deck side) view:
   -------------------------------------

                             ._____________.
                             |             |
                             |             |
                             |  +Zsc    +Ysc
   o==/ /==================o |      o----->|o==================/ /==o
     -Y Solar Array     +Xgrand     |      |        +Y Solar Array
                             |o-----> +Ygrand
                             .|_____|______.
                              |  .--V +Xsc
                              | '       `.
                              V __________\
                        +Xgrand   `.|.'                 +Zsc and +Zgrand
                                      HGA                   are out
                                                          of the page

   Nominally, the GRAND frame is co-aligned with the s/c frame.

   \begindata

      FRAME_DAWN_GRAND                 =  -203300
      FRAME_-203300_NAME               = 'DAWN_GRAND'
      FRAME_-203300_CLASS              =  4
      FRAME_-203300_CLASS_ID           =  -203300
      FRAME_-203300_CENTER             =  -203
      TKFRAME_-203300_RELATIVE         = 'DAWN_SPACECRAFT'
      TKFRAME_-203300_SPEC             = 'ANGLES'
      TKFRAME_-203300_UNITS            = 'DEGREES'
      TKFRAME_-203300_ANGLES           = ( 0.0, 0.0, 0.0 )
      TKFRAME_-203300_AXES             = ( 1,   2,   3   )

   \begintext


Dawn Mission NAIF ID Codes -- Definition Section
========================================================================

   This section contains name to NAIF ID mappings for the Dawn mission.


Dawn Target IDs:
-------------------------------------------------------------

   This table summarizes Dawn Target IDs:

            Name                   ID       Synonyms
            ---------------------  -------  ---------------------------
            VESTA                  2000004  '4 VESTA'
            CERES                  2000001  '1 CERES'

   Name-ID Mapping keywords:

         \begindata

            NAIF_BODY_NAME += ( '4 VESTA' )
            NAIF_BODY_CODE += ( 2000004 )

            NAIF_BODY_NAME += ( 'VESTA' )
            NAIF_BODY_CODE += ( 2000004 )

            NAIF_BODY_NAME += ( '1 CERES' )
            NAIF_BODY_CODE += ( 2000001 )

            NAIF_BODY_NAME += ( 'CERES' )
            NAIF_BODY_CODE += ( 2000001 )

         \begintext


Dawn Spacecraft ID
-------------------------------------------------------------

   This table summarizes Dawn Spacecraft IDs:

            Name                   ID
            ---------------------  -------
            DAWN                   -203

   Name-ID Mapping keywords:

         \begindata

            NAIF_BODY_NAME += ( 'DAWN' )
            NAIF_BODY_CODE += ( -203 )

         \begintext


Dawn Spacecraft Structures IDs
--------------------------------------

   This table summarizes Dawn Spacecraft Structure IDs:

            Name                   ID
            ---------------------  -------
            DAWN                   -203
            DAWN_SPACECRAFT        -203000
            DAWN_SA+Y              -203010
            DAWN_SA+Y_GIMBAL       -203011
            DAWN_SA+Y_C1           -203012
            DAWN_SA+Y_C2           -203013
            DAWN_SA+Y_C3           -203014
            DAWN_SA+Y_C4           -203015
            DAWN_SA-Y              -203020
            DAWN_SA-Y_GIMBAL       -203021
            DAWN_SA-Y_C1           -203022
            DAWN_SA-Y_C2           -203023
            DAWN_SA-Y_C3           -203024
            DAWN_SA-Y_C4           -203025
            DAWN_HGA               -203040
            DAWN_LGA+Z             -203030
            DAWN_LGA+X             -203050
            DAWN_LGA-Z             -203060
            DAWN_ST1               -203071
            DAWN_ST2               -203072
            DAWN_FT1               -203081
            DAWN_FT2               -203082
            DAWN_FT3               -203083

   Name-ID Mapping keywords:

         \begindata

            NAIF_BODY_NAME += ( 'DAWN_SPACECRAFT' )
            NAIF_BODY_CODE += ( -203000 )

            NAIF_BODY_NAME += ( 'DAWN_SA+Y' )
            NAIF_BODY_CODE += ( -203010 )

            NAIF_BODY_NAME += ( 'DAWN_SA+Y_GIMBAL' )
            NAIF_BODY_CODE += ( -203011 )

            NAIF_BODY_NAME += ( 'DAWN_SA+Y_C1' )
            NAIF_BODY_CODE += ( -203012 )

            NAIF_BODY_NAME += ( 'DAWN_SA+Y_C2' )
            NAIF_BODY_CODE += ( -203013 )

            NAIF_BODY_NAME += ( 'DAWN_SA+Y_C3' )
            NAIF_BODY_CODE += ( -203014 )

            NAIF_BODY_NAME += ( 'DAWN_SA+Y_C4' )
            NAIF_BODY_CODE += ( -203015 )

            NAIF_BODY_NAME += ( 'DAWN_SA-Y' )
            NAIF_BODY_CODE += ( -203020 )

            NAIF_BODY_NAME += ( 'DAWN_SA-Y_GIMBAL' )
            NAIF_BODY_CODE += ( -203021 )

            NAIF_BODY_NAME += ( 'DAWN_SA-Y_C1' )
            NAIF_BODY_CODE += ( -203022 )

            NAIF_BODY_NAME += ( 'DAWN_SA-Y_C2' )
            NAIF_BODY_CODE += ( -203023 )

            NAIF_BODY_NAME += ( 'DAWN_SA-Y_C3' )
            NAIF_BODY_CODE += ( -203024 )

            NAIF_BODY_NAME += ( 'DAWN_SA-Y_C4' )
            NAIF_BODY_CODE += ( -203025 )

            NAIF_BODY_NAME += ( 'DAWN_HGA' )
            NAIF_BODY_CODE += ( -203040 )

            NAIF_BODY_NAME += ( 'DAWN_LGA+Z' )
            NAIF_BODY_CODE += ( -203030 )

            NAIF_BODY_NAME += ( 'DAWN_LGA+X' )
            NAIF_BODY_CODE += ( -203050 )

            NAIF_BODY_NAME += ( 'DAWN_LGA-Z' )
            NAIF_BODY_CODE += ( -203060 )

            NAIF_BODY_NAME += ( 'DAWN_ST1' )
            NAIF_BODY_CODE += ( -203071 )

            NAIF_BODY_NAME += ( 'DAWN_ST2' )
            NAIF_BODY_CODE += ( -203072 )

            NAIF_BODY_NAME += ( 'DAWN_FT1' )
            NAIF_BODY_CODE += ( -203081 )

            NAIF_BODY_NAME += ( 'DAWN_FT2' )
            NAIF_BODY_CODE += ( -203082 )

            NAIF_BODY_NAME += ( 'DAWN_FT3' )
            NAIF_BODY_CODE += ( -203083 )

         \begintext


FC1 and FC2 IDs
--------------------------------------

   This table summarizes FC1 and FC2 IDs:

            Name                   ID
            ---------------------  -------
            DAWN_FC1               -203110
            DAWN_FC1_FILTER_1      -203111
            DAWN_FC1_FILTER_2      -203112
            DAWN_FC1_FILTER_3      -203113
            DAWN_FC1_FILTER_4      -203114
            DAWN_FC1_FILTER_5      -203115
            DAWN_FC1_FILTER_6      -203116
            DAWN_FC1_FILTER_7      -203117
            DAWN_FC1_FILTER_8      -203118
            DAWN_FC1_RAD           -203119

            DAWN_FC2               -203120
            DAWN_FC2_FILTER_1      -203121
            DAWN_FC2_FILTER_2      -203122
            DAWN_FC2_FILTER_3      -203123
            DAWN_FC2_FILTER_4      -203124
            DAWN_FC2_FILTER_5      -203125
            DAWN_FC2_FILTER_6      -203126
            DAWN_FC2_FILTER_7      -203127
            DAWN_FC2_FILTER_8      -203128
            DAWN_FC2_RAD           -203129

   Name-ID Mapping keywords:

         \begindata

            NAIF_BODY_NAME += ( 'DAWN_FC1' )
            NAIF_BODY_CODE += ( -203110 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_1' )
            NAIF_BODY_CODE += ( -203111 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_2' )
            NAIF_BODY_CODE += ( -203112 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_3' )
            NAIF_BODY_CODE += ( -203113 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_4' )
            NAIF_BODY_CODE += ( -203114 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_5' )
            NAIF_BODY_CODE += ( -203115 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_6' )
            NAIF_BODY_CODE += ( -203116 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_7' )
            NAIF_BODY_CODE += ( -203117 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_FILTER_8' )
            NAIF_BODY_CODE += ( -203118 )

            NAIF_BODY_NAME += ( 'DAWN_FC1_RAD' )
            NAIF_BODY_CODE += ( -203119 )

            NAIF_BODY_NAME += ( 'DAWN_FC2' )
            NAIF_BODY_CODE += ( -203120 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_1' )
            NAIF_BODY_CODE += ( -203121 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_2' )
            NAIF_BODY_CODE += ( -203122 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_3' )
            NAIF_BODY_CODE += ( -203123 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_4' )
            NAIF_BODY_CODE += ( -203124 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_5' )
            NAIF_BODY_CODE += ( -203125 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_6' )
            NAIF_BODY_CODE += ( -203126 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_7' )
            NAIF_BODY_CODE += ( -203127 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_FILTER_8' )
            NAIF_BODY_CODE += ( -203128 )

            NAIF_BODY_NAME += ( 'DAWN_FC2_RAD' )
            NAIF_BODY_CODE += ( -203129 )

         \begintext


VIR IDs
--------------------------------------

   This table summarizes VIR IDs:

            Name                   ID
            ---------------------  -------
            DAWN_VIR               -203200
            DAWN_VIR_SCAN          -203201
            DAWN_VIR_VIS           -203211
            DAWN_VIR_IR            -203213
            DAWN_VIR_VIS_SCAN      -203212
            DAWN_VIR_IR_SCAN       -203214
            DAWN_VIR_VIS_128       -203215
            DAWN_VIR_VIS_64        -203216
            DAWN_VIR_VIS_16        -203217
            DAWN_VIR_VIS_96        -203218
            DAWN_VIR_VIS_32        -203219
            DAWN_VIR_RAD           -203209

   Name-ID Mapping keywords:

         \begindata

            NAIF_BODY_NAME += ( 'DAWN_VIR' )
            NAIF_BODY_CODE += ( -203200 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_SCAN' )
            NAIF_BODY_CODE += ( -203201 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_VIS' )
            NAIF_BODY_CODE += ( -203211 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_IR' )
            NAIF_BODY_CODE += ( -203213 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_VIS_SCAN' )
            NAIF_BODY_CODE += ( -203212 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_IR_SCAN' )
            NAIF_BODY_CODE += ( -203214 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_VIS_128' )
            NAIF_BODY_CODE += ( -203215 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_VIS_64' )
            NAIF_BODY_CODE += ( -203216 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_VIS_16' )
            NAIF_BODY_CODE += ( -203217 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_VIS_96' )
            NAIF_BODY_CODE += ( -203218 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_VIS_32' )
            NAIF_BODY_CODE += ( -203219 )

            NAIF_BODY_NAME += ( 'DAWN_VIR_RAD' )
            NAIF_BODY_CODE += ( -203209 )

         \begintext


GRAND IDs
--------------------------------------

   This table summarizes GRAND IDs:

            Name                   ID
            ---------------------  -------
            DAWN_GRAND             -203300

   Name-ID Mapping keywords:

         \begindata

            NAIF_BODY_NAME += ( 'DAWN_GRAND' )
            NAIF_BODY_CODE += ( -203300 )

         \begintext


Dawn DSK Surface NAIF Names/ID Codes -- Definition Section
========================================================================

   This section contains name to NAIF ID mappings for the Vesta and 
   Ceres DSK surfaces.

   The following surface name-ID mappings are defined for Ceres:

             Surface Name          Surface ID     Body ID
      ---------------------------  ----------  -------------
      DAWN_DLR_SPG_135METER_V1       10101        2000001

      DAWN_GRAV_SPC_0128ICQ_V1       20101        2000001
      DAWN_GRAV_SPC_0256ICQ_V1       20102        2000001
      DAWN_GRAV_SPC_0512ICQ_V1       20103        2000001
      DAWN_GRAV_SPC_1024ICQ_V1       20104        2000001
      DAWN_GRAV_SPC_300METER_V1      20105        2000001
      DAWN_GRAV_SPC_100METER_V1      20106        2000001
      DAWN_GRAV_SPC_035METER_V1      20107        2000001
      DAWN_GRAV_SPC_0128ICQ_V2       20108        2000001
      DAWN_GRAV_SPC_0256ICQ_V2       20109        2000001
      DAWN_GRAV_SPC_0512ICQ_V2       20110        2000001
      DAWN_GRAV_SPC_1024ICQ_V2       20111        2000001


   \begindata

      NAIF_SURFACE_NAME += 'DAWN_DLR_SPG_135METER_V1'
      NAIF_SURFACE_CODE += 10101
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_0128ICQ_V1'
      NAIF_SURFACE_CODE += 20101
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_0256ICQ_V1'
      NAIF_SURFACE_CODE += 20102
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_0512ICQ_V1'
      NAIF_SURFACE_CODE += 20103
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_1024ICQ_V1'
      NAIF_SURFACE_CODE += 20104
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_300METER_V1'
      NAIF_SURFACE_CODE += 20105
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_100METER_V1'
      NAIF_SURFACE_CODE += 20106
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_035METER_V1'
      NAIF_SURFACE_CODE += 20107
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_0128ICQ_V2'
      NAIF_SURFACE_CODE += 20108
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_0256ICQ_V2'
      NAIF_SURFACE_CODE += 20109
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_0512ICQ_V2'
      NAIF_SURFACE_CODE += 20110
      NAIF_SURFACE_BODY += 2000001

      NAIF_SURFACE_NAME += 'DAWN_GRAV_SPC_1024ICQ_V2'
      NAIF_SURFACE_CODE += 20111
      NAIF_SURFACE_BODY += 2000001

   \begintext


End of FK.
