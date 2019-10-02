KPL/FK

\beginlabel
PDS_VERSION_ID               = PDS3
RECORD_TYPE                  = STREAM
RECORD_BYTES                 = "N/A"
^SPICE_KERNEL                = "nh_v220.tf"
MISSION_NAME                 = "NEW HORIZONS"
SPACECRAFT_NAME              = "NEW HORIZONS"
DATA_SET_ID                  = "NH-J/P/SS-SPICE-6-V1.0"
KERNEL_TYPE_ID               = FK
PRODUCT_ID                   = "nh_v220.tf"
PRODUCT_CREATION_TIME        = 2014-07-01T00:00:00
PRODUCER_ID                  = "APL"
MISSION_PHASE_NAME           = "N/A"
PRODUCT_VERSION_TYPE         = ACTUAL
PLATFORM_OR_MOUNTING_NAME    = "N/A"
START_TIME                   = "N/A"
STOP_TIME                    = "N/A"
SPACECRAFT_CLOCK_START_COUNT = "N/A"
SPACECRAFT_CLOCK_STOP_COUNT  = "N/A"
TARGET_NAME                  = {
                               JUPITER,
                               PLUTO,
                               "SOLAR SYSTEM"
                               }
INSTRUMENT_NAME              = "N/A"
NAIF_INSTRUMENT_ID           = "N/A"
SOURCE_PRODUCT_ID            = "N/A"
NOTE                         = "See comments in the file for details"
OBJECT                       = SPICE_KERNEL
  INTERCHANGE_FORMAT         = ASCII
  KERNEL_TYPE                = FRAMES
  DESCRIPTION                = "NH frames kernel"
END_OBJECT                   = SPICE_KERNEL
\endlabel

KPL/FK

New Horizons Spacecraft Frames Kernel
===============================================================================

   This frame kernel contains the New Horizons spacecraft and science
   instruments.

Version and Date
-------------------------------------------------------------------------------

   The TEXT_KERNEL_ID stores version information of loaded project text
   kernels. Each entry associated with the keyword is a string that consists
   of four parts: the kernel name, version, entry date, and type. For example,
   the frames kernel might have an entry as follows:

           TEXT_KERNEL_ID += 'NEWHORIZONS V1.0.0 22-FEBRUARY-2007 IK'
                                  |          |         |          |
                                  |          |         |          |
              KERNEL NAME <-------+          |         |          |
                                             |         |          V
                             VERSION <-------+         |     KERNEL TYPE
                                                       |
                                                       V
                                                  ENTRY DATE

   New Horizons Frame Kernel Version:

           \begindata

           TEXT_KERNEL_ID += 'NEWHORIZONS_FRAMES V2.2.0 16-OCT-2012 FK'

           NAIF_BODY_NAME += ( 'NEW_HORIZONS' )
           NAIF_BODY_CODE += ( -98 )

           NAIF_BODY_NAME += ( 'NH' )
           NAIF_BODY_CODE += ( -98 )

           NAIF_BODY_NAME += ( 'NH_SPACECRAFT' )
           NAIF_BODY_CODE += ( -98 )

           \begintext

   Version 2.2.0 -- October 16, 2012 -- Lillian Nguyen, JHU/APL

            --   Updated the Alice airglow frame with in-flight alignment values.

   Version 2.1.1 -- January 15, 2009 -- Lillian Nguyen, JHU/APL

            --   Corrected typos in the text.

   Version 2.1.0 -- January 7, 2009 -- Lillian Nguyen, JHU/APL

            --   Updated the Alice SOC frame with in-flight alignment values.

   Version 2.0.0 -- August 4, 2008 -- Lillian Nguyen, JHU/APL

            --   Added frames for the two Autonomous Star Trackers and the
                 Fine Sun Sensor.

            --   Updated the frames heirarchy and spacecraft diagram.

   Version 1.1.3 -- May 21, 2008 -- Lillian Nguyen, JHU/APL

            --   Added diagrams for Alice and Ralph describing the layout of
                 the detectors.

   Version 1.1.2 -- April 15, 2008 -- Lillian Nguyen, JHU/APL

            --   Updated the LORRI boresight based on in-flight data.

   Version 1.1.1 -- March 18, 2008 -- Lillian Nguyen, JHU/APL

            --   Rotated the SOC frame such that the instrument +Y axis is
                 at the center of the 2x2 degree portion of the slit rather
                 than at the optical center of the slit.

   Version 1.1.0 -- July 12, 2007 -- Lillian Nguyen, JHU/APL

            --   PEPSSI frame renamed and a new PEPSSI frame defining a
                 coordinate system axis relabelling created.
            --   Individual frame created for each PEPSSI sector and detector.

   Version 1.0.1 -- April 11, 2007 -- Lillian Nguyen, JHU/APL

            --   Alice airglow frame was updated with higher precision values
                 and an additional rotation to shift the boresight.

   Version 1.0.0 -- February 22, 2007 -- Lillian Nguyen, JHU/APL

            --   Corrected spelling errors.
            --   Modified the frames hierarchy diagram.
            --   Clarified that the entire Alice slit is visible through
                 both Alice apertures and updated the field of view
                 definitions and diagrams appropriately.
            --   Noted that the standard acronym for the Alice Solar
                 Occultation Channel is SOCC.
            --   Removed NH_ASTR frame from NAIF body name to ID mapping.
            --   Promoting to version 1.0.0 denoting approval of kernel set
                 by instrument teams.

   Version 0.0.5 -- January 15, 2007 -- Lillian Nguyen, JHU/APL

            --   Draft Version. NOT YET APPROVED BY ALL INSTRUMENT TEAMS.
            --   Star tracker frame inserted between spacecraft frame and
                 nominal instrument frames to reflect a spacecraft frame
                 change due to a star tracker calibration.
            --   LORRI frame definition changed according to instrument team.
            --   Ralph frames defined for LEISA and for each MVIC focal plane
                 array.
            --   Alice Airglow frame updated with in-flight values.

   Version 0.0.4 -- October 4, 2006 -- Lillian Nguyen, JHU/APL

            --   Draft Version. NOT YET APPROVED BY INSTRUMENT TEAMS.
            --   Removed 3-letter frame names, updated frame tree.
            --   Corrected the PEPSSI frame definition.

   Version 0.0.3 -- April 4, 2006 -- Lillian Nguyen, JHU/APL

            --   Draft Version. NOT YET APPROVED BY INSTRUMENT TEAMS.
            --   Alice airglow and SOC frames redefined according to
                 orientation diagrams received from instrument team.
            --   SWAP and PEPSSI frames added.
            --   Ralph frames modified according to focal plane definitions
                 received from instrument team.

   Version 0.0.2 -- January 25, 2006 -- Lillian Nguyen, JHU/APL

            --   Draft Version. NOT YET APPROVED BY INSTRUMENT TEAMS.
            --   Includes the addition of frames for REX and SDC.
            --   LORRI frame redefined according to orientation diagram
                 received from instrument team.

   Version 0.0.1 -- November 16, 2005 -- Lillian Nguyen, JHU/APL

            --   Draft Version. NOT YET APPROVED BY INSTRUMENT TEAMS.

   Version 0.0.0 -- August 13, 2005 -- Brian Carcich

            --   Testing Kernel.


References
-------------------------------------------------------------------------------

            1.   ``C-kernel Required Reading''

            2.   ``Kernel Pool Required Reading''

            3.   ``Frames Required Reading''

            4.   nh_v000.tf (placeholder New Horizon SPICE frames kernel),
                 provided by Brian Carcich and dated 2005-08-13.

            5.   New Horizons Spacecraft Requirements Document, 7399-9004

            6.   New Horizons Spacecraft Configuration Drawings,
                 7399-0002_-_10-28-03.pdf (\\Aplfsfrontier\project\pluto)

            7.   New Horizons Spacecraft to PERSI/RALPH Interface
                 Control Document, Rev B, 7399-9201.

            8.   New Horizons System Alignment Report, 7399-9189, dated
                 12 December, 2005.

            9.   Instrument Vectors v2.xls (containing an update to
                 Table 12 of [8]), received in an e-mail from John Troll
                 dated 1/16/2006; and PEPSSI baffle vector, received in an
                 e-mail from John Troll on 2/8/2006.

           10.   ``Rotation Required Reading''

           11.   Alice Instrument Specification, 05310.02-ISPEC-01.

           12.   Spacecraft to Alice Interface Control Document (ICD),
                 7399-9046.

           13.   Ralph Instrument Specification, Rev. A ECR SWRI 5310-001.

           14.   LOng-Range Reconnaissance Imager (LORRI) User's Manual,
                 7400-9601, dated Jan. 10, 2006.

           15.   LORRI_orientation_1-9-06, received on 1/23/2006 by e-mail
                 from Hal Weaver along with a description of the LORRI frame
                 relative to the spacecraft frame. Also a phone conversation
                 with Hal clarifying the diagrams in the document.

           16.   E-mail exchange with David James (Laboratory for Atmospheric
                 and Space Physics at the University of Colorado (LASP)),
                 Jan. 26, 2006 - Feb. 1, 2006.

           17.   P-ALICE_Orientation_on_SC, received from Joel Parker in an
                 e-mail dated Jan. 25, 2006; discussions with Dave Slater on
                 Mar. 16 and 23, 2006, and e-mail exchange with Dave Slater on
                 Mar. 28-29, 2006 regarding the diagram.

           18.   Pluto Energetic Particle Spectrometer Science Investigation
                 (PEPSSI) Interface Control Document, 7399-9049, Rev. C.

           19.   E-mail dated Feb. 8, 2006 from John Troll containing measured
                 PEPSSI baffle vector.

           20.   New Horizons Spacecraft to SWAP Interface Control Document,
                 7399-9047 Rev. A.

           21.   New Horizons Critical Design Review Science Payload slides.

           22.   Document titled "RalphArrayPositions.doc", received from
                 Cathy Olkin by e-mail, Mar. 23, 2006, and e-mail exchange
                 concerning the document, Apr. 3-4, 2006.

           23.   Ralph Instrument Specification, Rev. A ECR SWRI 5310-001.

           24.   Discussions with Scott Turner regarding his analysis of
                 PEPSSI data containing evidence of sunlight.

           25.   PEPSSI mounting bracket mechanical drawing, JHU/APL
                 document 7399-0151.

           26.   E-mail from Scott Turner describing the PEPSSI mounting
                 specification.

           27.   E-mail discussions among Gabe Rogers, Scott Turner, Hal
                 Weaver, and Howard Taylor concerning instrument boresight
                 changes caused by a star tracker calibration.

           28.   "AST1_SPIN_CAL.dat", received from Gabe Rogers on 12/4/2006.

           29.   Discussions with Howard Taylor regarding LORRI instrument
                 frame definition and LORRI keywords, 12/21/2006.

           30.   E-mail exchange with Cathy Olkin on MVIC coordinate system
                 and LEISA size.

           31.   "RalphBoresights03.doc", received from Allen Lunsford
                 2/2/2007.

           32.   E-mail from Andrew Steffl regarding Alice pointing offsets,
                 received on 2/13/2007 and 3/22/2007.

           33.   E-mail from Cathy Olkin regarding the removal of the
                 NH_RALPH_MVIC frame and the introduction of the NH_RALPH
                 frame, received 2/22/2007.

           34.   E-mail from Maarten Versteeg clarifying that the entire
                 Alice slit is visible through both Alice apertures, received
                 2/22/2007, and from Joel Parker confirming that we should
                 change the Alice fields of view to the entire lollipop-shaped
                 slit, received 2/28/2007.

           35.   Telephone conversation with Hal Weaver about the Alice
                 instrument.

           36.   Discussion with Jon Vandegriff and Larry Brown about the
                 PEPSSI frames and fields of view, 6/21/2007.

           37.   E-mails from Henry Throop received on 2/6/2008 and 2/20/2008.

           38.   E-mails from Hal Weaver received on 4/9/2008 and 4/15/2008.

           39.   E-mails from Andrew Steffl containing optical and detector
                 parameters and detector layout information for Ralph,
                 received between 3/12/2008 and 5/21/2008.

           40.   E-mail from Gabe Rogers received on 6/30/2008 containing
                 spacecraft to body matrices and fields of view for the star
                 trackers and sun sensor.

           41.   'Autonomous Star Tracker Performance for the New Horizons
                 Mission', AIAA/AAS Astrodynamics Specialist Conference,
                 Keystone, CO, August 21-24 2006.

           42.   E-mail regarding Alice SOC alignment, received from
                 Andrew Steffl on 11/18/2008.

           43.   E-mail regarding Alice airglow alignment, received from
                 Andrew Steffl on 9/28/2012.

Contact Information
-------------------------------------------------------------------------------

   Lillian Nguyen, JHU/APL, (443)-778-5477, Lillian.Nguyen@jhuapl.edu


Implementation Notes
-------------------------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make use of
   this instrument kernel must ``load'' the kernel, normally during program
   initialization. Loading the kernel associates data items with their names
   in a data structure called the ``kernel pool''. The SPICELIB routine FURNSH,
   CSPICE routine furnsh_c, and IDL routine cspice_furnsh load SPICE kernels
   as shown below:

   FORTRAN (SPICELIB)

           CALL FURNSH ( 'kernel_name' )

   C (CSPICE)

           furnsh_c ( "kernel_name" )

   ICY (IDL)

           cspice_furnsh, 'kernel_name'

   In order for a program or subroutine to extract data from the pool, the
   SPICELIB routines GDPOOL, GCPOOL, and GIPOOL are used. See [2] for details.

   This file was created and may be updated with a text editor or word
   processor.


New Horizons Frames
-------------------------------------------------------------------------------

   The following New Horizons frames are defined in this kernel file:

           Frame Name                Relative To              Type     NAIF ID
           =======================   ===================      =======  =======

           Spacecraft frames:
           -----------------------------------
           NH_SPACECRAFT             J2000                    CK       -98000
           NH_ASTR                   NH_SPACECRAFT            FIXED    -98001
           NH_STAR_TRACKER_1         NH_SPACECRAFT            FIXED    -98010
           NH_STAR_TRACKER_2         NH_SPACECRAFT            FIXED    -98011
           NH_FINE_SUN_SENSOR        NH_SPACECRAFT            FIXED    -98012

           Alice Frames (-981xx):
           -----------------------------------
           NH_ALICE_SOC              NH_ASTR                  FIXED    -98100
           NH_ALICE_AIRGLOW          NH_SPACECRAFT            FIXED    -98101

           RALPH Frames (-982xx):
           -----------------------------------
           NH_RALPH                  NH_RALPH_MVIC_FT         FIXED    -98200
           NH_RALPH_LEISA            NH_SPACECRAFT            FIXED    -98201
           NH_RALPH_SIA              NH_ASTR                  FIXED    -98202
           NH_RALPH_MVIC_FT          NH_SPACECRAFT            FIXED    -98203
           NH_RALPH_MVIC_PAN2        NH_SPACECRAFT            FIXED    -98204
           NH_RALPH_MVIC_PAN1        NH_SPACECRAFT            FIXED    -98205
           NH_RALPH_MVIC_RED         NH_SPACECRAFT            FIXED    -98206
           NH_RALPH_MVIC_BLUE        NH_SPACECRAFT            FIXED    -98207
           NH_RALPH_MVIC_METHANE     NH_SPACECRAFT            FIXED    -98208
           NH_RALPH_MVIC_NIR         NH_SPACECRAFT            FIXED    -98209

           LORRI Frames (-983xx):
           -----------------------------------
           NH_LORRI                  NH_SPACECRAFT            FIXED    -98300
           NH_LORRI_1X1              NH_LORRI                 FIXED    -98301
           NH_LORRI_4X4              NH_LORRI                 FIXED    -98302

           PEPSSI Frames (-984xx):
           -----------------------------------
           NH_PEPSSI_ENG             NH_ASTR                  FIXED    -98400
           NH_PEPSSI                 NH_PEPSSI_ENG            FIXED    -98401
           NH_PEPSSI_S0              NH_PEPSSI                FIXED    -98402
           NH_PEPSSI_S1              NH_PEPSSI                FIXED    -98403
           NH_PEPSSI_S2              NH_PEPSSI                FIXED    -98404
           NH_PEPSSI_S3              NH_PEPSSI                FIXED    -98405
           NH_PEPSSI_S4              NH_PEPSSI                FIXED    -98406
           NH_PEPSSI_S5              NH_PEPSSI                FIXED    -98407
           NH_PEPSSI_D0              NH_PEPSSI                FIXED    -98408
           NH_PEPSSI_D1              NH_PEPSSI                FIXED    -98409
           NH_PEPSSI_D2              NH_PEPSSI                FIXED    -98410
           NH_PEPSSI_D3              NH_PEPSSI                FIXED    -98411
           NH_PEPSSI_D4              NH_PEPSSI                FIXED    -98412
           NH_PEPSSI_D5              NH_PEPSSI                FIXED    -98413
           NH_PEPSSI_D6              NH_PEPSSI                FIXED    -98414
           NH_PEPSSI_D7              NH_PEPSSI                FIXED    -98415
           NH_PEPSSI_D8              NH_PEPSSI                FIXED    -98416
           NH_PEPSSI_D9              NH_PEPSSI                FIXED    -98417
           NH_PEPSSI_D10             NH_PEPSSI                FIXED    -98418
           NH_PEPSSI_D11             NH_PEPSSI                FIXED    -98419

           REX Frames (-985xx):
           -----------------------------------
           NH_REX                    NH_ASTR                  FIXED    -98500

           SWAP Frames (-986xx):
           -----------------------------------
           NH_SWAP                   NH_ASTR                  FIXED    -98600

           SDC Frames (-987xx):
           -----------------------------------
           NH_SDC                    NH_ASTR                  FIXED    -98700


New Horizons Frames Hierarchy
-------------------------------------------------------------------------------

   The diagram below shows the New Horizons frames hierarchy:

       'J2000' INERTIAL
          |
          |<--- ck
          |
         'NH_SPACECRAFT'
              |
             'NH_STAR_TRACKER_1'
              |
             'NH_STAR_TRACKER_2'
              |
             'NH_FINE_SUN_SENSOR'
              |
             'NH_RALPH_MVIC_FT'
              |   |
              |  'NH_RALPH'
              |
             'NH_RALPH_MVIC_PAN2'
              |
             'NH_RALPH_MVIC_PAN1'
              |
             'NH_RALPH_MVIC_RED'
              |
             'NH_RALPH_MVIC_BLUE'
              |
             'NH_RALPH_MVIC_METHANE'
              |
             'NH_RALPH_MVIC_NIR'
              |
             'NH_RALPH_LEISA'
              |
             'NH_LORRI'
              |    |
              |   'NH_LORRI_1X1'
              |    |
              |   'NH_LORRI_4X4'
              |
             'NH_ALICE_AIRGLOW'
              |
             'NH_ASTR'
                  |
                 'NH_ALICE_SOC'
                  |
                 'NH_RALPH_SIA'
                  |
                 'NH_PEPSSI_ENG'
                  |    |
                  |   'NH_PEPSSI'
                  |         |
                  |        'NH_PEPSSI_S0'
                  |         |
                  |        'NH_PEPSSI_S1'
                  |         |
                  |        'NH_PEPSSI_S2'
                  |         |
                  |        'NH_PEPSSI_S3'
                  |         |
                  |        'NH_PEPSSI_S4'
                  |         |
                  |        'NH_PEPSSI_S5'
                  |         |
                  |        'NH_PEPSSI_D0'
                  |         |
                  |        'NH_PEPSSI_D1'
                  |         |
                  |        'NH_PEPSSI_D2'
                  |         |
                  |        'NH_PEPSSI_D3'
                  |         |
                  |        'NH_PEPSSI_D4'
                  |         |
                  |        'NH_PEPSSI_D5'
                  |         |
                  |        'NH_PEPSSI_D6'
                  |         |
                  |        'NH_PEPSSI_D7'
                  |         |
                  |        'NH_PEPSSI_D8'
                  |         |
                  |        'NH_PEPSSI_D9'
                  |         |
                  |        'NH_PEPSSI_D10'
                  |         |
                  |        'NH_PEPSSI_D11'
                  |
                 'NH_REX'
                  |
                 'NH_SWAP'
                  |
                 'NH_SDC'



Spacecraft Frame
-------------------------------------------------------------------------------

   From [5]: (Note: The figures referenced below can not be reproduced here.
   There is a diagram below that basically illustrates what is contained
   there.)

        ``Figure 3.1.1-1 shows the New Horizons observatory and its coordinate
        system. This coordinate system shall be the only coordinate system
        used for all observatory-level hardware and software development.
        Nominal thrust direction for an Atlas V or Delta IV launch is in the
        +Y direction. Positive observatory roll is defined as a right handed
        rotation about the +X axis, positive pitch is defined a right handed
        rotation about the +Z axis and positive yaw is defined as a right
        handed rotation about the +Y axis. The origin of the coordinate
        system is located at the center of the bottom of the observatory
        adapter ring.''


        +X view:
        --------
                                          o
                                         /|\
                                        / | \
                                       /  |  \
                                      /   |   \
                                     /    |    \
                                    /     |     \
                ___________________/______|______\__________________
                `-.                                   HGA(REX)   ,-'
                   `-.                                        ,-'
                      `-.                                  ,-'  __
                         `-.____________________________,-'    /  / PEPSSI
                 __________/_\________________________/_\_____|___|
        PERSI .-|                |               |                |______
        Alice | |                |      RTG      |                |     ||
              '-|                |     .-*-.     |                |_____|| SWAP
                |                |    /     \    |                |     ||
           |----|                |    \     /    |                |     ||
     PERSI |    |                |     "-.-"     |                |
     Ralph |___ |                |               |                |
           |    |________________|_______________|________________|
                                |         +X (out of page)
                               /__<------o_________\
                                 +Zsc    |       adapter ring
                                         |
                                         |
                                         V
                                          -Ysc




        +Y view:
        --------

                ______
                ------
                  ||   SWAP
                 ----
                _|__|______   __..---..__
               | |  \     _`-'           ``-.   HGA(REX)
        PEPSSI | ----  _'     `-_            `-.
               |     .'          `-_            `.
             .-|   ,                `-_           `.
       LORRI : |  .                    `-_          `.
             : | /                        `-_         \
             '-|.                            `-_       . _______   _______
               |'                .-*-.          `-_    ||+|+|+|+| |+|+|+|+|
               |                /     \            `|--`-------------------|
               |               !   o-----> +X       |  |                   |
               |                \  |  /           _,|--.-------------------|
      ASTR 1 \ |.                "-|-"         _,-     ||+|+|+|+| |+|+|+|+|
             \\|'                  |        _,-        ' -------   -------
       Star   \| '                 V     _,-          /    RTG (Radioisotope
     Trackers  |  `               +Z  _,-            .          Thermoelectric
              /|   `               _,-              -           Generator)
             //|    `.          _,-               .'
      ASTR 2 / |       '.    _,-              _.-'
               |__________',-__         __,,,''
                 |     |       '' --- ''
                 |     |
                 `-----'  PERSI (Alice above, Ralph below)



   Since the S/C bus attitude with respect to an inertial frame is provided
   by a C-kernel (see [1] for more information), this frame is defined as
   a CK-based frame.

           \begindata

           FRAME_NH_SPACECRAFT     = -98000
           FRAME_-98000_NAME       = 'NH_SPACECRAFT'
           FRAME_-98000_CLASS      = 3
           FRAME_-98000_CLASS_ID   = -98000
           FRAME_-98000_CENTER     = -98
           CK_-98000_SCLK          = -98
           CK_-98000_SPK           = -98

           \begintext


ASTR (autonomous star trackers) Frame
-------------------------------------------------------------------------------

   From [27], the definition of the spacecraft body frame changed after
   launch due to a calibration in the ASTR (star tracker) alignments to the
   principal axes. The star tracker boresight needed to be rotated relative
   to the original orientation in the pre-flight calibration report [8]
   in order to match the thruster firing directions with the spin axis of the
   spacecraft (i.e., to the principal moment of inertia).

   In order to rotate all of the instrument boresights by the same amount,
   we introduce a frame taking vectors from the ASTR frame to the spacecraft
   body frame. The nominal instrument frames will be linked to the ASTR frame
   rather than to the spacecraft frame.

   The ASTR to spacecraft body matrix is given [28] by:

            [ 0.99998698852861   0.00510125214304   0.00000025156926 ]
      DCM = [-0.00510125214304   0.99998698366466   0.00009862975258 ]
            [ 0.00000025156926  -0.00009862975258   0.99999999513605 ]


           \begindata

           FRAME_NH_ASTR            = -98001
           FRAME_-98001_NAME        = 'NH_ASTR'
           FRAME_-98001_CLASS       = 4
           FRAME_-98001_CLASS_ID    = -98001
           FRAME_-98001_CENTER      = -98
           TKFRAME_-98001_SPEC      = 'MATRIX'
           TKFRAME_-98001_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98001_MATRIX    = ( 0.99998698852861,
                                       -0.00510125214304,
                                        0.00000025156926,
                                        0.00510125214304,
                                        0.99998698366466,
                                       -0.00009862975258,
                                        0.00000025156926,
                                        0.00009862975258,
                                        0.99999999513605)

           \begintext


   New Horizons has two autonomous star trackers (ASTRs) on the spacecraft -X
   panel. Their boresights are separated by 90 degrees [41]. The star tracker
   to spacecraft body matrices are given in [40] as:

      Star Tracker 1 =

         [ 0.00509822454128  0.70789996418568  -0.70629430750392 ]
         [ 0.99998695603811 -0.00339039616065   0.00382007428188 ]
         [ 0.00030961293888 -0.70630457022434  -0.70790801536644 ]

      Star Tracker 2 =

         [ 0.00578813992901173 -0.705383466342775   -0.708802273448943   ]
         [ 0.999982550630013    0.00492038051247466  0.00326929519670704 ]
         [ 0.00118147011512493 -0.708808828433902    0.705399637696606   ],

   which translates to the following frame definitions:

           \begindata

           FRAME_NH_STAR_TRACKER_1  = -98010
           FRAME_-98010_NAME        = 'NH_STAR_TRACKER_1'
           FRAME_-98010_CLASS       = 4
           FRAME_-98010_CLASS_ID    = -98010
           FRAME_-98010_CENTER      = -98
           TKFRAME_-98010_SPEC      = 'MATRIX'
           TKFRAME_-98010_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98010_MATRIX    = ( 0.00509822454128,
                                        0.99998695603811,
                                        0.00030961293888,
                                        0.70789996418568,
                                       -0.00339039616065,
                                       -0.70630457022434,
                                       -0.70629430750392,
                                        0.00382007428188,
                                       -0.70790801536644)

           FRAME_NH_STAR_TRACKER_2  = -98011
           FRAME_-98011_NAME        = 'NH_STAR_TRACKER_2'
           FRAME_-98011_CLASS       = 4
           FRAME_-98011_CLASS_ID    = -98011
           FRAME_-98011_CENTER      = -98
           TKFRAME_-98011_SPEC      = 'MATRIX'
           TKFRAME_-98011_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98011_MATRIX    = ( 0.00578813992901173,
                                        0.999982550630013,
                                        0.00118147011512493,
                                       -0.705383466342775,
                                        0.00492038051247466,
                                       -0.708808828433902,
                                       -0.708802273448943,
                                        0.00326929519670704,
                                        0.705399637696606)

           \begintext


Sun Sensor Frame
-------------------------------------------------------------------------------

   The Fine Sun Sensor to spacecraft body matrix is given in [40] as:

      Fine Sun Sensor =

         [  0.702792970261541    0.711384357815347    0.0037863447564826   ]
         [ -0.00276516946126598 -0.00259068940063707  0.999992821057371    ]
         [  0.711389060071083   -0.702798394836006    0.000146380002554562 ],

   which translates to the following frame definition:

           \begindata

           FRAME_NH_FINE_SUN_SENSOR = -98012
           FRAME_-98012_NAME        = 'NH_FINE_SUN_SENSOR'
           FRAME_-98012_CLASS       = 4
           FRAME_-98012_CLASS_ID    = -98012
           FRAME_-98012_CENTER      = -98
           TKFRAME_-98012_SPEC      = 'MATRIX'
           TKFRAME_-98012_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98012_MATRIX    = ( 0.702792970261541,
                                       -0.00276516946126598,
                                        0.711389060071083,
                                        0.711384357815347,
                                       -0.00259068940063707,
                                       -0.702798394836006,
                                        0.0037863447564826,
                                        0.999992821057371,
                                        0.000146380002554562)

           \begintext


Alice Frames
-------------------------------------------------------------------------------

   The Alice instrument has two apertures, the Solar Occulation Channel, and
   the Airglow opening. Light travels through a lollipop-shaped slit
   described below for both apertures. Although [12], Rev A, defines the
   Solar Occulation Channel field of view as only the 2.0 x 2.0 degree "box"
   at the top of the lollipop-shaped slit and the Airglow field of view as the
   0.1 x 4.0 degree "stem" of the lollipop-shaped slit, the complete lollipop
   shape is visible through both apertures [34].

   The spectral resolution of airglow is degraded in the 2.0 x 2.0 degree
   "box" portion of the slit, but data is nevertheless still available in that
   portion of the slit. The 2.0 x 2.0 degree "box" portion of the slit is
   wider than the narrow "stem" portion to observe the sun during Solar
   Occulation Channel operations, but it is also possible for the sun to
   appear in the narrower "stem" portion [10]. Hence, the fields of view for
   both apertures have been defined here to be the entire lollipop-shaped
   slit even though they are defined differently in [3], Rev A.

Solar Occultation Channel (SOC) Frame Definition

   The SOC is also referred to as the SOCC (Solar OCcultation Channel) to
   distinguish it from the Science Operations Center. Since the documents
   to which this kernel refers still use the SOC acronym for the Solar
   Occultation Channel, we continue to use it here as well, although SOCC has
   become the standard term [34].

   The SOC frame is defined by the instrument team ([17] and [37]) such that
   the +Y axis in the instrument frame is the center of the 2.0 x 2.0 degree
   portion of the slit, and the X and Z axes are close to the spacecraft X and
   Z axes, respectively.

   From [12], the nominal SOC boresight is a line in the spacecraft YZ plane
   right-hand rotated by 2 degrees around the spacecraft X axis from the REX
   boresight (nominally the Y axis). We are given the alignment for the SOC
   boresight in [9] and will use that measured vector to determine the frame.

                            Z   ^
                             sc |
                                |                  SOC boresight
                                |            _.-   (optical boresight)
                                |        _.-'
                                |    _.-'   o
                                |_.-'    2.0
                                o-------------->
                             X                 Y
                              sc                sc

   The diagram below shows the projection onto the sky of the Alice entrance
   slit through the Alice optics for the SOC aperture. The SOC field of view
   is the entire lollipop-shaped slit illustrated below [34].

                Projection of Alice slit on the
                 sky through the SOC aperture
                                                       Spacecraft Axes
                               _
                              | |                             ^ +Z
                              | |                             |   sc
                              | |                             |
                              | |                             |
                              | |                             |
                              | |                             x--------->
                              | |                       +Y (in)          +X
                              | |                         sc               sc
                              | |
                    ---       |x| <---Alice optical
                     ^    o   | |     path center
                     | 2.0    | |
                     |    ____| |____
                     |   |           |
                     |   |           |
                    _v_  |     _     | /____  SOC
                         |           | \      boresight
                         |           |
                         |___________|

                 <------------------- wavelength


    The following diagram [39] illustrates the projections of the spacecraft
    axes through the SOCC aperture onto the Alice detector. The origin in
    the detector view is at the bottom left.

                  ________________________________________________
   increasing  ^ |                                                |
         rows  | |                                                |
               | |                  +Y (in)                       |
               | |  +X  <---------x   sc                          |
               | |    sc          |                               |
               | |                |                               |
               | |                |                               |
               | |                V  +Z                           |
               | |                     sc                         |
                 O________________________________________________|
                  ------------------------>
           [0,0]=[column, row]            increasing columns
                                          increasing wavelength

   The following frame definition uses nominal pre-launch calibration
   values. It remains here for reference but has been replaced with
   in-flight values, described below.

   From [9], the measured Alice optical path center in NH_ASTR coordinates
   is:

                                       [ -0.0011506 ]
           Alice optical path center = [  0.9993305 ]
                                       [  0.0365690 ]

   The SOC frame is defined such that the boresight (the center of the
   2.0 x 2.0 degree portion of the slit) is the instrument +Y axis [37]. To
   transform the optical center to the SOC boresight, the optical center is
   rotated -2.0 degrees about the NH_ASTR +X axis to obtain the instrument
   +Y axis:

                                 [ -0.00115060000000000 ]
           Alice SOC +Y Vector = [  0.99999797455532025 ]
                                 [  0.00167059166380266 ]

   The instrument +X vector is defined to be the NH_ASTR +X axis, so

                                 [ 1.0 ]
           Alice SOC +X Vector = [ 0.0 ]
                                 [ 0.0 ]

   The instrument +Z vector is determined by taking the cross product X x Y:

                                 [  0.0                 ]
           Alice SOC +Z Vector = [ -0.00167059271628372 ]
                                 [  0.99999860455901446 ]

   And we use that to adjust the +X vector to form an orthogonal frame:

                                         [ 0.99999933805964336 ]
           Alice SOC +X Vector = Y x Z = [ 0.00115059835766032 ]
                                         [ 0.00000192218391797 ]

   Using these three vectors, we define the rotation that takes vectors from
   the instrument frame to the NH_ASTR frame as

   [     ]   [ 0.99999933805964336 -0.00115060000000000  0.0                 ]
   [ ROT ] = [ 0.00115059835766032  0.99999797455532025 -0.00167059271628372 ]
   [     ]   [ 0.00000192218391797  0.00167059166380266  0.99999860455901446 ]

           FRAME_NH_ALICE_SOC       = -98100
           FRAME_-98100_NAME        = 'NH_ALICE_SOC'
           FRAME_-98100_CLASS       = 4
           FRAME_-98100_CLASS_ID    = -98100
           FRAME_-98100_CENTER      = -98
           TKFRAME_-98100_SPEC      = 'MATRIX'
           TKFRAME_-98100_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98100_MATRIX    = ( 0.99999933805964336,
                                        0.00115059835766032,
                                        0.00000192218391797,
                                       -0.00115060000000000,
                                        0.99999797455532025,
                                        0.00167059166380266,
                                        0.0,
                                       -0.00167059271628372,
                                        0.99999860455901446 )

   An in-flight alignment gives the following redefinition of the SOC frame
   [42]. Starting from the rotation matrix R1 taking vectors from the nominal
   NH_ALICE_SOC frame to the NH_SPACECRAFT frame, an additional rotation of
   0.42205843 degrees about the spacecraft Y axis followed by a rotation of
   0.013987654 degrees about the spacecraft Z axis are required to fit the
   alignment data. This results in the following rotation matrix, R2, taking
   vectors from the instrument frame to the NH_SPACECRAFT frame:

      [    ]   [                   ]   [                  ]   [    ]
      [ R2 ] = [ (0.013987654 deg) ] * [ (0.42205843 deg) ] * [ R1 ]
      [    ]   [                   ]Z  [                  ]Y  [    ]

               [  0.99996405571443  0.00418309873041 -0.00737488739974 ]
             = [ -0.00419478716187  0.99998996915870 -0.00157014096732 ]
               [  0.00736824536873  0.00160102061271  0.99997157244253 ]

           \begindata

           FRAME_NH_ALICE_SOC       = -98100
           FRAME_-98100_NAME        = 'NH_ALICE_SOC'
           FRAME_-98100_CLASS       = 4
           FRAME_-98100_CLASS_ID    = -98100
           FRAME_-98100_CENTER      = -98
           TKFRAME_-98100_SPEC      = 'MATRIX'
           TKFRAME_-98100_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98100_MATRIX    = (  0.99996405571443,
                                        -0.00419478716187,
                                         0.00736824536873,
                                         0.00418309873041,
                                         0.99998996915870,
                                         0.00160102061271,
                                        -0.00737488739974,
                                        -0.00157014096732,
                                         0.99997157244253 )

           \begintext


Airglow Channel Frame Definition

   The airglow frame is defined [17] such that -X axis in the instrument frame
   is the boresight, and +Y and +Z axes in the instrument frame are close to
   the spacecraft +Y and +Z axes, respectively.

   The diagram below shows the projection of the Alice slit onto the sky
   through the Alice optics for the airglow aperture. The spacecraft axes are
   shown to the right of the slit diagram. The airglow boresight is centered
   in the 6.0 degree long slit. Its field of view is rotated +2.0 degrees with
   respect to the spacecraft +X axis due to the 2.0 degree instrument tip to
   center the +Y axis (the REX antenna boresight) in the center of the SOC
   field of view [12, 17]. The airglow field of view is the entire lollipop-
   shaped slit illustrated below [34].

               Projection of Alice slit on the       Spacecraft Axes
               sky through the Airglow aperture
                                                             |   o/ +Z
                             _                               |2.0/    sc
                            | |                              |  /
                            | |                              | /
                            | |                              |/
                            | |                              o - - - - -
                            | |                      +X (out) `-._ 2.0 deg
                            | |                        sc          `-._
                            | |                                        ` +Y
                            | |                                            sc
                            | |
                            |+| <---Airglow           Airglow instrument axes
                            | |     boresight         are rotated +2.0 degrees
                            | |                       about the spacecraft +X
                        ____| |____                   axis.
                       |           |
                       |           |                  Instrument Axes
                       |           |
                       |           |                         ^ +Z
                       |           |                         |   inst
                       |___________|                         |
                                                             |
                    -------------------> wavelength          o-------> +Y
                                                        +X  (out)        inst
                                                          inst

    The following diagram [39] illustrates the projections of the spacecraft
    axes through the Airglow aperture onto the Alice detector. The origin in
    the detector view is at the bottom left.

    Note that the instrument tip about the spacecraft X axis is not depicted.
    The actual projected spacecraft axes are rotated by a nominal -2.0
    degrees about the spacecraft +X axis from the positions shown below.

                  ________________________________________________
   increasing  ^ |                                                |
         rows  | |                                                |
               | |                  +X  (out)                     |
               | |  +Y  <---------o   sc                          |
               | |    sc          |                               |
               | |                |                               |
               | |                |                               |
               | |                V  +Z                           |
               | |                     sc                         |
                 O________________________________________________|
                  ------------------------>
           [0,0]=[column, row]            increasing columns
                                          increasing wavelength


   Note that the following calculations use nominal pre-launch calibration
   values. These calculations remain here for reference, but have been
   replaced with in-flight calibration values, which are described below the
   nominal frame definition. The boresight has also changed from the center
   of the Alice slit to slightly off-center.

   The nominal frame is described here:

   We are given in [9] the measured boresight vector in spacecraft
   coordinates:

      Airglow boresight = -X     = [-0.9999836  0.0011311  0.0056122]
                            inst

   We must calculate the instrument Y and Z axes in spacecraft coordinates in
   order to define a rotation matrix, R, which takes vectors from the airglow
   instrument frame to vectors in the spacecraft frame.

   To calculate the instrument Y axis, we will rotate the spacecraft Y axis
   by +2.0 degrees about the spacecraft +X axis and use the projection of the
   rotated Y axis onto the plane perpendicular to the measured boresight as
   the instrument Y axis. The instrument Z axis is then determined using the
   cross product. Note that this calculation assumes there is no misalignment
   in the plane perpendicular to the boresight (that there is no twist in the
   misalignment). The rotation of the spacecraft Y axis is shown here:


                       +Z
                      |  SC
                      |
                      |         _. +Y
                      |     _.-' o   rotated
                      | _.-'    2
                      o'------------
                 +X  (out)           +Y
                   SC                  SC


   The rotated Y axis in spacecraft coordinates is

      +Y         =  [ 0.0  cos( 2.0 deg )  sin( 2.0 deg ) ]
        rotated

   We now calculate the projection of the rotated Y axis onto the plane
   perpendicular to the measured boresight to get the instrument Y axis:


                       -X     (Boresight)
                      ^  inst
                      |
                      |        / +Y
                      |      /     rotated
                      |    /
                      |  /
                      |/
                      o--------->
                 +Z  (out)       +Y
                   inst            inst


   Since the boresight vector is unit length, using the dot product and vector
   addition, we have

      +Y     =  +Y        -  Boresight * dot(+Y        , Boresight)
        inst      rotated                      rotated

   The SPICE routine VPERP does this calculation for us, yielding

      +Y     = [ 0.001326253357475  0.999390205835991  0.034892084075427 ]
        inst

   The instrument Z axis is determined using the cross product:

      +Z     =  +X     x  +Y
        inst      inst      inst

             = [ 0.005569311419949 -0.034898955455451  0.999375327731491 ]

   The rotation matrix R taking vectors in the instrument frame to vectors
   in the spacecraft frame is then

      [     ]   [  0.9999836  0.001326253357475  0.005569311419949 ]
      [  R  ] = [ -0.0011311  0.999390205835991 -0.034898955455451 ]
      [     ]   [ -0.0056122  0.034892084075427  0.999375327731491 ]

   This nominal frame definition is shown below. It remains here for
   reference only and will not be loaded into the SPICE kernel pool.

           FRAME_NH_ALICE_AIRGLOW   = -98101
           FRAME_-98101_NAME        = 'NH_ALICE_AIRGLOW'
           FRAME_-98101_CLASS       = 4
           FRAME_-98101_CLASS_ID    = -98101
           FRAME_-98101_CENTER      = -98
           TKFRAME_-98101_SPEC      = 'MATRIX'
           TKFRAME_-98101_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98101_MATRIX    = ( 0.9999836,
                                       -0.0011311,
                                       -0.0056122,
                                        0.001326253357475,
                                        0.999390205835991,
                                        0.034892084075427,
                                        0.005569311419949,
                                       -0.034898955455451,
                                        0.999375327731491)

   The following definition updates the nominal frame, but has since been
   replaced with more current values. These older values remain here for
   reference only.

   In-flight values for pointing offsets are described in [32] as:

   "The error bars are the standard deviation of the residuals. To convert
   from the s/c coordinate system to the Alice instrument coordinate system
   (the boresight is defined to be [-1,0,0] in both coordinate systems),
   perform the following rotations (in degrees):

            Rotation about s/c X  -1.717403537893697  +/-  0.14135753
            Rotation about new Z   0.368710896944916  +/-  0.013300878
            Rotation about new Y  -0.313630482588893  +/-  0.013886115

   Applying the inverse rotations, we arrive at the Alice boresight in
   spacecraft coordinates:

           [ -0.999964312690322  0.006435174723831  0.005473743878372 ]

   The 6 degree entrance slit is imaged onto rows 6-25 (inclusive). Therefore
   the center of the slit should fall exactly on the boundary between rows 15
   and 16. However, I believe it is preferable that when the Alice boresight
   aims toward a point source, the spectrum fall onto a single row (or as
   close to this as possible). Therefore, I have added an additional rotation
   of +0.10 degrees (1/3 of one 0.3 degree detector row) about the Y axis to
   put all the flux into row 16. In this case the Alice boresight in s/c
   coordinates is:

           [ -0.999972343138423  0.006435174723831  0.003728469461561 ]

           Rotation about new Y  -0.213630482588893  +/-  0.002959385400478"

   Note that the airglow boresight is no longer at the Alice slit center. The
   slit center is located at

           [ -0.999964312690322  0.006435174723831  0.005473743878372 ]

   Using the SPICE subroutine ROTMAT, the three rotations described above are
   calculated, leading to the following transformation, which takes vectors
   in the instrument frame to vectors in the spacecraft frame:

        [     ]   [  0.999972343138423  0.006543983365249  0.003534011879914 ]
        [  R  ] = [ -0.006435174723831  0.999530106264816 -0.029969237503143 ]
        [     ]   [ -0.003728469461561  0.029945666664166  0.999544574075370 ]

           FRAME_NH_ALICE_AIRGLOW   = -98101
           FRAME_-98101_NAME        = 'NH_ALICE_AIRGLOW'
           FRAME_-98101_CLASS       = 4
           FRAME_-98101_CLASS_ID    = -98101
           FRAME_-98101_CENTER      = -98
           TKFRAME_-98101_SPEC      = 'MATRIX'
           TKFRAME_-98101_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98101_MATRIX    = ( 0.999972343138423,
                                       -0.006435174723831,
                                       -0.003728469461561,
                                        0.006543983365249,
                                        0.999530106264816,
                                        0.029945666664166,
                                        0.003534011879914,
                                       -0.029969237503143,
                                        0.999544574075370)

   The frame defined above has been superceded by [43]. The above frame will
   not be loaded into the kernel pool and remains here for reference only.
   Updated values for the rotation matrix taking vectors from the instrument
   frame to the spacecraft frame are described in [43] as:

   "This new rotation matrix is the result of a re-analysis of data from the
   Alice boresight alignment scans made during the post-launch commissioning.
   Although the Alice slit is physically rectilinear, optical distortions
   (mostly coma) introduced by the primary mirror result in a slit that appears
   somewhat curved, when projected onto the sky. The initial analysis of the
   alignment data failed to take this into account, resulting in the calculated
   boresight being approximately 0.02 (20% of the 0.1 degree slit width)
   degrees from the centerline of the slit. The new rotation matrix to
   transform from the NH_SPACECRAFT coordinate system to the NH_ALICE_AIRGLOW
   coordinate system is obtained by making the following rotations (in order):

            Rotation about X:      -1.59926267084552 degrees
            Rotation about Z:       0.35521825089567 degrees
            Rotation about Y:      -0.23245579623587 degrees

   The Alice airglow boresight ([-1, 0, 0] in the instrument frame) is located
   at the following coordinates in the NH_SPACECRAFT frame:

           [ -0.99997255180979   0.00608399347241   0.00422855181362]

   Using the SPICE subroutine ROTMAT, the three rotations described above are
   calculated, leading to the following transformation, which takes vectors
   in the instrument frame to vectors in the spacecraft frame:

        [     ]   [  0.99997255180979    0.00619968832527    0.00405702990897]
        [  R  ] = [ -0.00608399347241    0.99959126350983   -0.02793368823199]
        [     ]   [ -0.00422855181362    0.02790823855932    0.99960154540200]"

           \begindata

           FRAME_NH_ALICE_AIRGLOW   = -98101
           FRAME_-98101_NAME        = 'NH_ALICE_AIRGLOW'
           FRAME_-98101_CLASS       = 4
           FRAME_-98101_CLASS_ID    = -98101
           FRAME_-98101_CENTER      = -98
           TKFRAME_-98101_SPEC      = 'MATRIX'
           TKFRAME_-98101_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98101_MATRIX    = ( 0.99997255180979,
                                       -0.00608399347241,
                                       -0.00422855181362,
                                        0.00619968832527,
                                        0.99959126350983,
                                        0.02790823855932,
                                        0.00405702990897,
                                       -0.02793368823199,
                                        0.99960154540200)

           \begintext


RALPH Frames
-------------------------------------------------------------------------------

   The RALPH instrument consists of Linear Etalon Imaging Spectral Array
   (LEISA) and Multispectral Visible Imaging Camera (MVIC). RALPH also has a
   Solar Illumination Aperture (SIA). RALPH is mounted on the +Z side of the
   New Horizons spacecraft.

   MVIC obtains science imaging data using frame transfer data acquisition and
   uses optical filters to provide spectral band discrimination in the blue,
   red, near infrared (NIR), methane, and two panchromatic bands [23].

   From [22], the instrument boresight is aligned with the center of the frame
   transfer array. A summary of the relative positions of the individual
   arrays is presented below, with detailed calculations following. To
   convert the numbers in Table 1 from microns to microradians use the
   conversion factor of 1.5208 microradian/micron.

   From [22] Table 1:

   The offset in microns of the individual focal plane arrays in Ralph
   -------------------------------------------------------------------
                     Offset of the array       Offset of the array
                     from the instrument       from the instrument
                     boresight in the MVIC     boresight in the MVIC
   Array             row direction (microns)   column direction (microns)
   -----             ---------------------     ---------------------
   Frame Transfer             0.0                      0.0
   Pan 2                   2041.0                      0.0
   Pan 1                   3354.0                      0.0
   Red                     4693.0                      0.0
   Blue                    5941.0                      0.0
   Methane                 7280.0                      0.0
   NIR                     8632.0                      0.0
   LEISA                   -218.0                   3334.0


   From [22] Table 2 (LEISA from [30]):

   Size in pixels of the MVIC and LEISA arrays
   -------------------------------------------------------------------
   Array              Rows        Columns
   -----              ----        -------
   Frame Transfer     128          5024
   Pan 2               32          5024
   Pan 1               32          5024
   Red                 32          5024
   Blue                32          5024
   Methane             32          5024
   NIR                 32          5024
   LEISA              256           256


   The detail of the calculations used in Table 1 for the MVIC arrays [22]:

   Pan 2:   (221 - 64) * 13 microns/pixel = 2041 microns
   Pan 1:   (322 - 64) * 13 microns/pixel = 3354 microns
   Red:     (425 - 64) * 13 microns/pixel = 4693 microns
   Blue:    (521 - 64) * 13 microns/pixel = 5941 microns
   Methane: (624 - 64) * 13 microns/pixel = 7280 microns
   NIR:     (728 - 64) * 13 microns/pixel = 8632 microns

   The first number in the calculations above is the MVIC row at which the
   filter is centered. Subtracted from that number is 64 in all cases, which
   is the MVIC row of the center of the frame transfer array. The MVIC pixels
   are 13 microns wide with a single pixel FOV of 19.8 microradians [22].

   The detail of the calculation used in Table 1 for LEISA [22]:

   The center of the MVIC frame transfer (FT) array corresponds to column
   44.65 and row 133.45 of LEISA. The distance from the instrument
   boresight (center of the FT array) to the center of the LEISA array is
   calculated here:

   In MVIC row direction (128 - 133.45)* 40 microns/pixel = -218 microns
   In MVIC col direction (128 -  44.65)* 40 microns/pixel = 3334 microns

   The LEISA scale is used here because this is the difference between two
   different LEISA pixel coordinates. The MVIC directions are used so that all
   offsets are presented in the same coordinate system. The LEISA pixels are
   40 microns wide with a single pixel FOV of 61 microradians.


Multispectral Visible Imaging Camera (MVIC) Frames Definition

   From [39]:

   MVIC Frame Transfer is a staring instrument. Each stare/read cycle
   produces 1 readout image that is translated into an image plane in
   the corresponding image file created by the SOC (Science Operations
   Center). Each image plane has roughly the same viewing geometry.
   All of Ralph's FOVs share a single telescope, nominally aligned to
   the spacecraft -X axis.

   When viewed by an observer looking out MVIC's Frame Transfer boresight,
   the spacecraft axes on the sky will look like:

   Diagram 1
   ---------
                     Sky View Looking out from MVIC Frame Transfer
                  _______________________________________________________
                 |                                                       |
                 |                             ^  +Y                     |
                 |                             |                         |
                 |                             |                         |
                 |                             |                         |
                 |                    <------- o                         |
                 |                   +Z         +X   (out)               |
                 |                                                       |
                 0_______________________________________________________|


   Displaying one MVIC Frame Transfer image plane using IDL after SOC
   processing will look like the following:

   Diagram 2
   ---------
                            MVIC Frame Transfer IDL Display
                  _______________________________________________________
                 |                                                       |
                 |                            ^ +Y                       |
              ^  |                            |                          |
              |  |                            |                          |
              |  |                            |                          |
              |  |                    <-------o                          |
   Increasing |  |                    +Z                                 |
   rows (128) |  |                                                       |
                 0_______________________________________________________|
                  -------------------------->
                  Increasing columns (5024)


   NOTE: the following calculations use nominal alignment values and have been
   preceded by in-flight values. The following two frames remain in this
   kernel for reference only. Only the frames enclosed within the
   "\begindata \begintext" block are used by SPICE.

   The base instrument frame, NH_RALPH_MVIC, describes the nominal instrument
   misalignment (note that the definition of this frame has changed after
   in-flight calibration - see below).

   Because the RALPH coordinate system is the same as the spacecraft coordinate
   system [7], the rotation matrix that takes vectors represented in the
   nominal MVIC frame into the spacecraft frame is the identity. We will
   adjust the rotation to take into account the measured alignment provided
   in [9].

   From [9], the measured MVIC boresight vector in spacecraft coordinates is:

                                        [ -0.9999688 ]
           MVIC Boresight Vector = -X = [  0.0078090 ]
                                        [ -0.0011691 ]

   and the detector direction (+Z axis in the instrument frame) is measured
   to be:

                                     [ -0.0011761 ]
           MVIC Detector Direction = [  0.0036511 ]
                                     [  0.9999926 ]

   Taking the cross product of these two vectors gives us:

                                    [  0.007813211258259 ]
           MVIC +Y Vector = Z x X = [  0.999962844813125 ]
                                    [ -0.003641802174272 ]

   And we use that to adjust the Z vector to form an orthogonal frame:

                                    [ -0.001140617758206 ]
           MVIC +Z Vector = X x Y = [  0.003650823069793 ]
                                    [  0.999992685214268 ]

   Using these three vectors, we define the rotation that takes vectors from
   the instrument frame to the spacecraft frame as

           [     ]   [  0.9999688  0.007813211258259 -0.001140617758206 ]
           [ ROT ] = [ -0.0078090  0.999962844813125  0.003650823069793 ]
           [     ]   [  0.0011691 -0.003641802174272  0.999992685214268 ]

           FRAME_NH_RALPH_MVIC      = -98200
           FRAME_-98200_NAME        = 'NH_RALPH_MVIC'
           FRAME_-98200_CLASS       = 4
           FRAME_-98200_CLASS_ID    = -98200
           FRAME_-98200_CENTER      = -98
           TKFRAME_-98200_SPEC      = 'MATRIX'
           TKFRAME_-98200_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98200_MATRIX    = (  0.9999688,
                                        -0.0078090,
                                         0.0011691,
                                         0.007813211258259,
                                         0.999962844813125,
                                        -0.003641802174272,
                                        -0.001140617758206,
                                         0.003650823069793,
                                         0.999992685214268 )

   Since the instrument boresight is aligned with the center of the frame
   transfer array [22], the rotation matrix taking vectors in the frame
   transfer array frame to the base instrument frame is the identity:

           FRAME_NH_RALPH_MVIC_FT   = -98203
           FRAME_-98203_NAME        = 'NH_RALPH_MVIC_FT'
           FRAME_-98203_CLASS       = 4
           FRAME_-98203_CLASS_ID    = -98203
           FRAME_-98203_CENTER      = -98
           TKFRAME_-98203_SPEC      = 'MATRIX'
           TKFRAME_-98203_RELATIVE  = 'NH_RALPH_MVIC'
           TKFRAME_-98203_MATRIX    = ( 1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0 )

   The above nominal NH_RALPH_MVIC frame has been updated in version 0.0.5 of
   this kernel. The calculations above had erroneously been done in single
   precision in earlier versions. Version 0.0.5 also connects NH_RALPH_MVIC to
   the NH_ASTR frame rather than the NH_SPACECRAFT frame to take into account
   the in-flight change to the star tracker alignment. Note that the
   NH_RALPH_MVIC and NH_RALPH_MVIC_FT frames defined above are in the text
   section of the kernel and will not be loaded into SPICE. They have been
   replaced by the frames defined below using in-flight measured values and
   remain in this kernel for reference only.

   The definitions below have been provided in [31]. [31] notes that the
   matrix values below contain an adjustment to correct a shift in values
   returned by SPICE. The MVIC and LEISA matrices may be updated in a future
   release once the cause of the shift is known. Note that the instrument
   reference frames are all defined with respect to the NH_SPACECRAFT frame.

   MVIC Pan Frame Transfer Array

           \begindata

           FRAME_NH_RALPH_MVIC_FT   = -98203
           FRAME_-98203_NAME        = 'NH_RALPH_MVIC_FT'
           FRAME_-98203_CLASS       = 4
           FRAME_-98203_CLASS_ID    = -98203
           FRAME_-98203_CENTER      = -98
           TKFRAME_-98203_SPEC      = 'MATRIX'
           TKFRAME_-98203_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98203_MATRIX    = ( 0.999915461106432,
                                       -0.012957863475922,
                                        0.001301425571368,
                                        0.012962551750366,
                                        0.999908274152575,
                                       -0.003840453099600,
                                       -0.001188689962478,
                                        0.003857294149614,
                                        0.999990855106924 )

           \begintext

   The NH_RALPH_MVIC frame has been removed [33]. A new frame, NH_RALPH,
   describes the Ralph instrument boresight, which is by definition the
   center of the frame transfer array, and is such set to the identity matrix
   with respect to the NH_RALPH_MVIC_FT frame.

   Basic boresight for the instrument

           \begindata

           FRAME_NH_RALPH           = -98200
           FRAME_-98200_NAME        = 'NH_RALPH'
           FRAME_-98200_CLASS       = 4
           FRAME_-98200_CLASS_ID    = -98200
           FRAME_-98200_CENTER      = -98
           TKFRAME_-98200_SPEC      = 'MATRIX'
           TKFRAME_-98200_RELATIVE  = 'NH_RALPH_MVIC_FT'
           TKFRAME_-98200_MATRIX    = ( 1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0 )

           \begintext


MVIC Time Delay Integration (TDI) Frames Frame Definitions
(includes PAN1, PAN2, RED, BLUE, METHANE, and NIR)

   From [39]:

   MVIC TDI is a scanning instrument. Each scan/read cycle produces 1 readout
   line that is translated into a single image row in the corresponding image
   file created by the SOC (Science Operations Center). All of Ralph's FOVs
   share a single telescope, nominally aligned to the spacecraft -X axis. The
   FOVs for the TDI frames are each slightly offset from each other, but the
   following diagrams are a valid approximation for all.

   When viewed by an observer looking out MVIC's TDI boresight, the spacecraft
   axes on the sky will look like:

   Diagram 1
   ---------
                        Sky View Looking out from MVIC TDI Frames

                                             ^ +Y
                                             |                     scan    ^
                                             |                   direction |
                                             |                             |
                  ___________________________|___________________________  |
                 |                +Z <-------o +X   (out)                | |
                 |_______________________________________________________| |


   Displaying the MVIC TDI image using IDL after SOC processing will look
   like the following:

   Diagram 2
   ---------
                                  MVIC TDI IDL Display
                  _______________________________________________________
                 |                                                       |
                 |                                                       |
                 |                                                       |
                 |                                                       |
              ^  |                                                       |
              |  |                            ^ +Y                       |
              |  |                            |                          |
              |  |                            |                          |
              |  |                            |                          |
              |  |                    <-------o                          |
   Increasing |  |                    +Z                                 |
   image row, |  |                                                       |
   scan time  |  |                                                       |
                 0_______________________________________________________|
                  -------------------------->
                  Increasing columns (5024)


   When viewed by an observer looking out MVIC's TDI boresights, each FOV
   is aligned slightly offset from each other, along the +Y direction:

   Diagram 3
   ---------
                        Sky View Looking out from MVIC TDI Frames

   NIR          |========================================================|

   METHANE      |========================================================|

   RED          |========================================================|

   BLUE         |========================================================|

   PAN1         |========================================================|

   PAN2         |========================================================|

                                           ^ +Y
                                           |                     scan    ^
                                           |                   direction |
                                           |                             |
                                +Z <-------o +X (out)

   TDI Arrays

           \begindata

           FRAME_NH_RALPH_MVIC_NIR     = -98209
           FRAME_-98209_NAME        = 'NH_RALPH_MVIC_NIR'
           FRAME_-98209_CLASS       = 4
           FRAME_-98209_CLASS_ID    = -98209
           FRAME_-98209_CENTER      = -98
           TKFRAME_-98209_SPEC      = 'MATRIX'
           TKFRAME_-98209_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98209_MATRIX    =  ( 0.999655573953295,
                                        -0.026218528636659,
                                         0.001363247322305,
                                         0.026223117930647,
                                         0.999649201897941,
                                        -0.003626292333940,
                                        -0.001204840860741,
                                         0.003661087780433,
                                         0.999991572287813 )

           FRAME_NH_RALPH_MVIC_METHANE     = -98208
           FRAME_-98208_NAME        = 'NH_RALPH_MVIC_METHANE'
           FRAME_-98208_CLASS       = 4
           FRAME_-98208_CLASS_ID    = -98208
           FRAME_-98208_CENTER      = -98
           TKFRAME_-98208_SPEC      = 'MATRIX'
           TKFRAME_-98208_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98208_MATRIX    =  ( 0.999706214601813,
                                        -0.024211181536640,
                                         0.001355962837282,
                                         0.024215758201663,
                                         0.999699832809316,
                                        -0.003629168417489,
                                        -0.001204837178782,
                                         0.003661233729783,
                                         0.999991571758176 )

           FRAME_NH_RALPH_MVIC_BLUE     = -98207
           FRAME_-98207_NAME        = 'NH_RALPH_MVIC_BLUE'
           FRAME_-98207_CLASS       = 4
           FRAME_-98207_CLASS_ID    = -98207
           FRAME_-98207_CENTER      = -98
           TKFRAME_-98207_SPEC      = 'MATRIX'
           TKFRAME_-98207_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98207_MATRIX    =  ( 0.999795403077788,
                                        -0.020196202570963,
                                         0.001341376615173,
                                         0.020200753919525,
                                         0.999789001887114,
                                        -0.003634876673948,
                                        -0.001204830694104,
                                         0.003661525649188,
                                         0.999991570697647 )

           FRAME_NH_RALPH_MVIC_RED     = -98206
           FRAME_-98206_NAME        = 'NH_RALPH_MVIC_RED'
           FRAME_-98206_CLASS       = 4
           FRAME_-98206_CLASS_ID    = -98206
           FRAME_-98206_CENTER      = -98
           TKFRAME_-98206_SPEC      = 'MATRIX'
           TKFRAME_-98206_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98206_MATRIX    =  ( 0.999752824372622,
                                        -0.022203736816693,
                                         0.001348672591773,
                                         0.022208300833224,
                                         0.999746432868338,
                                        -0.003632029868004,
                                        -0.001204833789898,
                                         0.003661379686231,
                                         0.999991571228120 )

           FRAME_NH_RALPH_MVIC_PAN1     = -98205
           FRAME_-98205_NAME        = 'NH_RALPH_MVIC_PAN1'
           FRAME_-98205_CLASS       = 4
           FRAME_-98205_CLASS_ID    = -98205
           FRAME_-98205_CENTER      = -98
           TKFRAME_-98205_SPEC      = 'MATRIX'
           TKFRAME_-98205_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98205_MATRIX    =  ( 0.999833950545631,
                                        -0.018188586893952,
                                         0.001334074936899,
                                         0.018193125555121,
                                         0.999827539694002,
                                        -0.003637708823840,
                                        -0.001204827891411,
                                         0.003661671618066,
                                         0.999991570166760 )

           FRAME_NH_RALPH_MVIC_PAN2 = -98204
           FRAME_-98204_NAME        = 'NH_RALPH_MVIC_PAN2'
           FRAME_-98204_CLASS       = 4
           FRAME_-98204_CLASS_ID    = -98204
           FRAME_-98204_CENTER      = -98
           TKFRAME_-98204_SPEC      = 'MATRIX'
           TKFRAME_-98204_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98204_MATRIX    =  ( 0.999868466620725,
                                        -0.016180897880494,
                                         0.001326767586392,
                                         0.016185423834895,
                                         0.999862046133615,
                                        -0.003640526306263,
                                        -0.001204825381830,
                                         0.003661817592276,
                                         0.999991569635460 )

           \begintext


Linear Etalon Imaging Spectral Array (LEISA) Frames Definition

   From [39]:

   LEISA is a scanning instrument. Each scan/read cycle produces 1 readout
   image that is translated into an image plane in the corresponding image
   file created by the SOC (Science Operations Center). All of Ralph's FOVs
   share a single telescope, nominally aligned to the spacecraft -X axis.

   When viewed by an observer looking out LEISA's boresight, the spacecraft
   axes on the sky will look like:

   Diagram 1
   ---------
                   Sky View Looking out from LEISA
                   _______________________________
                  |                               |
                  |                               |
                  |              ^ +Y             | ^
                  |              |                | |
                  |              |                | |
                  |              |                | | Scan
                  |              |                | | direction(s)
                  |      <-------o                | |
                  |     +Z         +X (out)       | |
                  |                               | |
                  |                               | |
                  |                               | |
                  |                               | v
                  |                               |
                  |_______________________________|


   Displaying one LEISA image plane using IDL after SOC processing will
   look like the following:

   Diagram 2
   ---------
                      LEISA IDL Display (1 image)
                   _______________________________
                  |                               |
                  |                +Y             |
                  |High Res      ^                | ^
                  |--------------|----------------| |
                ^ |Low Res       |                | |
                | |              |                | |S
                | |              |                | |c
                | |              x--------->      | |a
                | |                      +Z       | |n
                | |                               | |
                | |                               | |
   Increasing   | |                               | |
   rows (256),  | |                               | v
   wavelengths  | |                               |
                  0_______________________________|
                    --------------------->
                    Increasing columns (256)
                    Fixed wavelength


   Displaying one LEISA image row (constant wavelength) over all image
   planes using IDL after SOC processing will look like the following (note
   that the vertical dimension of the image is determined by the length of
   the scan; the horizontal dimension is the number of spatial pixels):

   Diagram 3
   ---------
                  LEISA IDL Display (1 wavelength)
                   _______________________________
                  |                               |
                  |                               |
                  |                               |
                  |                               |
                  |                               |
                  |                +Y  or -Y      |
                  |              ^ (depends on    |
                  |              | scan direction)|
                  |              |                |
                  |              |                |
                  |              |                |
                ^ |              |                |
                | |              x--------->      |
                | |                       +Z      |
                | |                               |
                | |                               |
   Increasing   | |                               |
   image plane, | |                               |
   scan time    | |                               |
                  0_______________________________|
                    --------------------->
                    Increasing columns (256)


   The LEISA frame is defined in [31] as

           \begindata

           FRAME_NH_RALPH_LEISA     = -98201
           FRAME_-98201_NAME        = 'NH_RALPH_LEISA'
           FRAME_-98201_CLASS       = 4
           FRAME_-98201_CLASS_ID    = -98201
           FRAME_-98201_CENTER      = -98
           TKFRAME_-98201_SPEC      = 'MATRIX'
           TKFRAME_-98201_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98201_MATRIX    =  ( 0.999833725668520,
                                        -0.018154907521815,
                                         0.001873657185926,
                                         0.018161563736818,
                                         0.999817822153710,
                                        -0.003715956943165,
                                        -0.001743042311673,
                                         0.003750674847578,
                                         0.999990409103958 )

           \begintext


Solar Illumination Aperture (SIA) Frame Definition

   We define the SIA frame such that +Z axis in the instrument frame is the
   boresight and +X in the instrument frame is aligned to the spacecraft +X
   axis. Looking down the spacecraft X axis, we have


                            Z   ^
                             sc |
                                |
                                |
                                |            _.- SIA boresight vector (+Z    )
                                |        _.-'                            inst
                                |    _.-'   o
                                |_.-'    2.0
                                o-------------->
                      X    = X                 Y
                       inst   sc                sc


                                          Plane X = 0

   The rotation matrix that transforms vectors in the instrument frame to the
   spacecraft frame can be defined by the single rotation

           [     ]   [              ]
           [ ROT ] = [ (90.0 - 2.0) ]
           [     ]   [              ]
                                     X

   where [x]  represents the rotation matrix of a given angle x about axis i.
            i

           \begindata

           FRAME_NH_RALPH_SIA       = -98202
           FRAME_-98202_NAME        = 'NH_RALPH_SIA'
           FRAME_-98202_CLASS       = 4
           FRAME_-98202_CLASS_ID    = -98202
           FRAME_-98202_CENTER      = -98
           TKFRAME_-98202_SPEC      = 'ANGLES'
           TKFRAME_-98202_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98202_ANGLES    = ( 0.0, 88.0, 0.0 )
           TKFRAME_-98202_AXES      = ( 3     1    3 )
           TKFRAME_-98202_UNITS     = 'DEGREES'

           \begintext


LOng Range Reconnaissance Imager (LORRI) Frames
-------------------------------------------------------------------------------

   The following diagrams are reproduced from [15] and [29].

   When viewed by an observer looking out LORRI's boresight, the spacecraft
   axes on the sky will look like:

   Diagram 1
   ---------
                  Sky View Looking out from LORRI
                 _________________________________
                |                                 |
                |                                 |
                |                ^ +Y             |
                |                |   sc           |
                |                |                |
                |                |                |
                |                |                |
                |       <--------o                |
                |     +Z           +X  (out)      |
                |       sc           sc           |
                |                                 |
                |                                 |
                |                                 |
                |                                 |
                |_________________________________|


   The LORRI optics inverts images in both the Y and Z directions, so that the
   projection of these spacecraft axes onto the LORRI CCD will look like the
   following: (Note that we are looking INTO the LORRI telescope in the
   diagram below, whereas above we were looking outwards, hence the position
   of the +Z axis does not appear to have changed when in fact it has flipped).

   Diagram 2
   ---------
                   Looking in at the LORRI CCD
                 _________________________________
                |                                 |       Spacecraft Axes
                |                                 |
                |                                 |              ^ +Y
                |                                 |              |   sc
  increasing  ^ |                                 |              |
   columns    | |                   p             |              x-----> +Z
              | |    p            +X  (in)        |        +X (in)         sc
              | |  +Z  <---------x  sc            |          sc
              | |    sc          |                |
              | |                |                |
              | |                |                |
              | |                |    p           |
              | |                V  +Y            |
              | |                     sc          |
                O_________________________________|
                 ------------------------>
          [0,0]=[column, row]            increasing rows

                                                 p       p
   Note that in Diagram 2, the axes are labeled Z   and Y   to clarify
                                                 sc      sc
   that although these are still spacecraft coordinates, they are the
   projections of the spacecraft axes from Diagram 1 onto the LORRI CCD, not
   the actual spacecraft axes. The actual spacecraft axes are depicted to the
   right of Diagram 2. The origin in the CCD view is at the bottom left, and
   the CCD storage area and serial register are to the left.

   The LORRI IDL display further inverts the image in Diagram 2 about the
   diagonal originating at [0,0]:

   Diagram 3
   ---------
                        LORRI IDL Display
                 _________________________________
                |                                 |       Spacecraft Axes
                |                                 |
                |                                 |              ^ +Z
                |                                 |              |   sc
  increasing  ^ |                                 |              |
        rows  | |                    p            |              o-----> +Y
              | |    p             +X  (out)      |        +X (out)        sc
              | |  +Y  <---------x   sc           |          sc
              | |    sc          |                |
              | |                |                |
              | |                |                |
              | |                |    p           |
              | |                V  +Z            |
              | |                     sc          |
                O_________________________________|
                 ------------------------>
          [0,0]=[column, row]            increasing columns



   Also provided here are the same set of three diagrams using the LORRI
   instrument axes, X , Y , Z , rather than the spacecraft axes.
                     L   L   L

   Diagram 1a
   ----------
                  Sky View Looking out from LORRI
                 _________________________________
                |                                 |
                |                                 |      Spacecraft Axes
                |                                 |
                |                                 |             ^ +Y
                |                                 |             |   sc
                |                                 |             |
                |                                 |       <-----o
                |                o--------->      |      +Z      +X  (out)
                |                |          Y     |        sc      sc
                |                |           L    |
                |                |                |
                |                |                |
                |                V X              |
                |                   L             |
                |_________________________________|


   Diagram 2a
   ----------
                   Looking in at the LORRI CCD
                 _________________________________
                |                                 |
                |                   p             |
                |                ^ X              |
                |                |  L             |
  increasing  ^ |                |                |
   columns    | |                |                |
              | |                |                |
              | |                x---------> p    |
              | |                           Y     |
              | |                            L    |
              | |                                 |
              | |                                 |
              | |                                 |
              | |                                 |
                O_________________________________|
                 ------------------------>
          [0,0]=[column, row]            increasing rows

   As in Diagram 2, the axes in Diagram 2a are the projections of the LORRI
   instrument axes through the optics onto the LORRI CCD.

   Diagram 3a
   ---------
                        LORRI IDL Display
                 _________________________________
                |                                 |
                |                   p             |
                |                ^ Y              |
                |                |  L             |
  increasing  ^ |                |                |
        rows  | |                |                |
              | |                |                |
              | |            p   o---------> p    |
              | |           Z (out)         X     |
              | |            L               L    |
              | |                                 |
              | |                                 |
              | |                                 |
              | |                                 |
                O_________________________________|
                 ------------------------>
          [0,0]=[column, row]            increasing columns



   Taken from [29], we have the following coordinate system definition for the
   LORRI frame:

   The -Z axis in instrument coordinates is defined to be the boresight and
   is approximately aligned with the spacecraft -X axis. The Y axis in
   instrument coordinates is approximately aligned with the spacecraft -Z axis
   and is in the direction of increasing rows. The X axis in instrument
   coordinates is approximately aligned with the spacecraft -Y axis and is in
   the direction of increasing columns.

   The following nominal frame definition remains here for reference only and
   will not be loaded into the SPICE kernel pool. See below for the updated
   frame.

   Using the measured vectors from [9], we have:

                                             [ -0.9999955 ]
           LORRI Boresight Vector (-Z    ) = [  0.0005485 ]
                                     inst    [ -0.0029601 ]

   and the detector direction (-Y axis in the instrument frame) in spacecraft
   coordinates is measured to be:

                             [ -0.0029702 ]
           LORRI -Y Vector = [  0.0069403 ]
                             [  0.9999715 ]

   Taking the cross product of these two vectors gives us:

                                     [ -0.000569028334549 ]
           LORRI +X Vector = Y x Z = [ -0.999975765451163 ]
                                     [  0.006938639428225 ]

   And we use that to adjust the Y vector to form an orthogonal frame:

                                     [  0.002956222326369 ]
           LORRI +Y Vector = Z x X = [ -0.006940292366278 ]
                                     [ -0.999971546140903 ]

   Using these three vectors, we define the rotation that takes vectors from
   the instrument frame to the spacecraft frame as

           [     ]   [ -0.000569028334549  0.002956222326369  0.9999955 ]
           [ ROT ] = [ -0.999975765451163 -0.006940292366278 -0.0005485 ]
           [     ]   [  0.006938639428225 -0.999971546140903  0.0029601 ]

           FRAME_NH_LORRI           = -98300
           FRAME_-98300_NAME        = 'NH_LORRI'
           FRAME_-98300_CLASS       = 4
           FRAME_-98300_CLASS_ID    = -98300
           FRAME_-98300_CENTER      = -98
           TKFRAME_-98300_SPEC      = 'MATRIX'
           TKFRAME_-98300_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98300_MATRIX    = ( -0.000569028334549,
                                        -0.999975765451163,
                                         0.006938639428225,
                                         0.002956222326369,
                                        -0.006940292366278,
                                        -0.999971546140903,
                                         0.9999955,
                                        -0.0005485,
                                         0.0029601 )


   The updated, in-flight value for the LORRI boresight is given in [38] as:

                                             [ -0.99998064 ]
           LORRI Boresight Vector (-Z    ) = [  0.00543141 ]
                                     inst    [ -0.00303788 ]

   The new LORRI +Y vector is the component of the nominal LORRI +Y vector
   in the NH_SPACECRAFT frame perpendicular to the updated boresight vector.
   The LORRI X axis completes the right-handed orthogonal frame:

           [     ]   [ -0.005452680629036  0.002999533810427  0.99998064 ]
           [ ROT ] = [ -0.999960367261253 -0.007054338553346 -0.00543141 ]
           [     ]   [  0.007037910250677 -0.999970619120629  0.00303788 ]

           \begindata

           FRAME_NH_LORRI           = -98300
           FRAME_-98300_NAME        = 'NH_LORRI'
           FRAME_-98300_CLASS       = 4
           FRAME_-98300_CLASS_ID    = -98300
           FRAME_-98300_CENTER      = -98
           TKFRAME_-98300_SPEC      = 'MATRIX'
           TKFRAME_-98300_RELATIVE  = 'NH_SPACECRAFT'
           TKFRAME_-98300_MATRIX    = ( -0.005452680629036,
                                        -0.999960367261253,
                                         0.007037910250677,
                                         0.002999533810427,
                                        -0.007054338553346,
                                        -0.999970619120629,
                                         0.99998064,
                                        -0.00543141,
                                         0.00303788 )
           \begintext

   LORRI has two binning modes - 1x1 and 4x4. Separate frames are defined
   below for each of those modes. The frames are identical to the NH_LORRI
   frame, hence the identity rotation.

           \begindata

           FRAME_NH_LORRI_1X1       = -98301
           FRAME_-98301_NAME        = 'NH_LORRI_1X1'
           FRAME_-98301_CLASS       = 4
           FRAME_-98301_CLASS_ID    = -98301
           FRAME_-98301_CENTER      = -98
           TKFRAME_-98301_SPEC      = 'MATRIX'
           TKFRAME_-98301_RELATIVE  = 'NH_LORRI'
           TKFRAME_-98301_MATRIX    = ( 1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0 )

           FRAME_NH_LORRI_4X4       = -98302
           FRAME_-98302_NAME        = 'NH_LORRI_4X4'
           FRAME_-98302_CLASS       = 4
           FRAME_-98302_CLASS_ID    = -98302
           FRAME_-98302_CENTER      = -98
           TKFRAME_-98302_SPEC      = 'MATRIX'
           TKFRAME_-98302_RELATIVE  = 'NH_LORRI'
           TKFRAME_-98302_MATRIX    = ( 1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0 )
           \begintext


Pluto Energetic Particle Spectrometer Science Investigation (PEPSSI) Frames
-------------------------------------------------------------------------------

   From [18], PEPSSI is mounted on the spacecraft +Y panel in the -X/-Z corner
   of the spacecraft's top panel. It has been mounted on a bracket to provide
   a clear field of view.

   The PEPSSI coordinate system is defined in Figure A-4 of [18] such that
   the instrument Z axis is the normal vector to the baffle shielding the
   the instrument from the high gain antenna, the instrument Y axis is the
   instrument look direction, and the instrument X axis completes the right-
   handed frame. Three coarse reproductions of Figure A-4 are shown below.
   The coordinate system is displayed above each diagram. Figure A-5 of [18]
   shows the instrument mounted to its bracket. The mounting bracket is
   represented by the column of "B"s in the diagrams below):

                +Z                     -X                        +Z
               ^  inst                ^  inst                   ^  inst
               |                      |                         |
               |                      |                         |
               x----> +X              o----> +Y                 o----> +Y
         +Y (in)        inst      +Z  (out)     inst        +X  (out)    inst
           inst                     inst  __                  inst
                                         |  ``-.
                                         |      `.
       ____________________   B _________|_       '              ____________
      |____________________|  B|           `.      \    B ______|____________|
            |        |        B|             \      .   B|         |____|
       ^    |        |        B|              |     |   B|         |       ^
       |    |        |        B|             /      '   B|         |  ^    |
       |    |        |        B|_________ _,'      /    B|         |  |    |
    baffle  |        |        B          |        ,     B|         |  | baffle
    points  |________|                   |      ,'      B|_________|  |
    into                      baffle --> |__..-`        B           sensor
    page

   The PEPSSI alignment is given in [9] as a single vector - the measured
   normal to the instrument baffle. This vector defines the instrument +Z
   axis:

                                         [ 0.4574230384 ]
          PEPSSI baffle normal (+Z  ) =  [ 0.2112844736 ]
                                  inst   [ 0.8637841369 ]

  The nominal instrument look direction is determined by analyzing the
  mechanical drawings of the instrument mounting bracket [25]. Figure A-5
  of [18] shows PEPSSI mounted on the sloped face of the bracket, with its
  look direction (instrument +Y axis) normal to the sloped face. Two views
  of the mounting bracket in the spacecraft coordinate frame are [25]:

                      ^ Y                             ^ Y
                      |  sc                           |  sc
                      |                               |
            X  <------x Z (in)                 X (in) x------> Z
             sc          sc                     sc             sc

         _ _ _ _ _ _ _____                    _ _ _ _ _ _ ______
            o    _.-'     |                      o    _.-'      |
      28.343 _.-'        _|                27.832 _.-'          |
          .-'        _.-' |                    .-'              |
          |      _.-'     |                    |                |
          |  _.-'         |                    |                |
          .-'_____________|                    |________________|
      ////////////////////////              ////////////////////////
             spacecraft                           spacecraft

   The face of the bracket on which PEPSSI is mounted is the outward-facing
   sloped surface in the above left diagram. The normal to the face is the
   instrument +Y axis. To calculate the normal, we first use the given angles
   to find two vectors in the plane of the bracket face:

       V1 = [ cos(28.343 deg), -sin(28.343 deg), 0  ]

       V2 = [ 0, -sin(27.832 deg), -cos(27.832 deg) ]

   The normal to the bracket face (instrument +Y) is given in spacecraft
   coordinates by the cross product:

                         [  0.430539299735951 ]
       Y     = V1 x V2 = [  0.798162645175740 ]
        inst             [ -0.421393288068217 ]

   The instrument X axis is orthogonal to the plane containing the
   instrument Y and Z axes:

                              [  0.778475068630558 ]
       X     = Y    x  Z    = [ -0.564648724992781 ]
        inst    inst    inst  [ -0.274132057382342 ]

   The Y axis is adjusted to form an orthogonal frame:

                              [  0.429814764127889 ]
       Y     = Z    x  X    = [  0.797828733864454 ]
        inst    inst    inst  [ -0.422763030500455 ]

   Using these three vectors, we define the rotation that takes vectors from
   the instrument frame to the spacecraft frame as

           [     ]   [  0.778475068630558  0.429814764127889 0.4574230384 ]
           [ ROT ] = [ -0.564648724992781  0.797828733864454 0.2112844736 ]
           [     ]   [ -0.274132057382342 -0.422763030500455 0.8637841369 ]

           \begindata

           FRAME_NH_PEPSSI_ENG      = -98400
           FRAME_-98400_NAME        = 'NH_PEPSSI_ENG'
           FRAME_-98400_CLASS       = 4
           FRAME_-98400_CLASS_ID    = -98400
           FRAME_-98400_CENTER      = -98
           TKFRAME_-98400_SPEC      = 'MATRIX'
           TKFRAME_-98400_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98400_MATRIX    = ( 0.778475068630558
                                       -0.564648724992781
                                       -0.274132057382342
                                        0.429814764127889
                                        0.797828733864454
                                       -0.422763030500455
                                        0.4574230384
                                        0.2112844736
                                        0.8637841369 )

           \begintext

   Note that it was determined ([24], [26]) that PEPSSI was incorrectly
   mounted on the spacecraft. The above frame definition describes the
   actual mounting on the spacecraft, which is different from the intended
   mounting specification, described in [26]. [24] also suggests that the
   Euler rotations in Figure A-5 of [18] describe neither the mounting
   specification nor the actual mounting, but rather a permutation of the
   mounting specification.

   The PEPSSI frame defined above has been named NH_PEPSSI_ENG to denote that
   its coordinate system is defined in the engineering diagrams found in [18].
   The NH_PEPSSI frame below is a rotation of the NH_PEPSSI_ENG frame that is
   more suitable for data analysis [36]. In the NH_PEPSSI frame, the +Z axis
   is the boresight, and the -Y axis is the normal vector to the baffle. The
   NH_PEPSSI frame defined in this way is also referred to in the instrument
   kernel as the frame for defining the boresight vectors and boundary corner
   vectors of the PEPSSI sectors.

           \begindata

           FRAME_NH_PEPSSI          = -98401
           FRAME_-98401_NAME        = 'NH_PEPSSI'
           FRAME_-98401_CLASS       = 4
           FRAME_-98401_CLASS_ID    = -98401
           FRAME_-98401_CENTER      = -98
           TKFRAME_-98401_SPEC      = 'MATRIX'
           TKFRAME_-98401_RELATIVE  = 'NH_PEPSSI_ENG'
           TKFRAME_-98401_MATRIX    = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        0.0
                                       -1.0
                                        0.0
                                        1.0
                                        0.0  )

           \begintext

   As a convenience, an individual frame is defined below for each of the six
   PEPSSI sectors and twelve detectors. The boresight is defined to be the
   +Z axis in each frame.

   The following sector frames were determined by rotating the NH_PEPSSI frame
   about the NH_PEPSSI Y axis by the number of degrees required to move the
   NH_PEPSSI boresight (NH_PEPSSI +Z axis) into the center of the sector's
   field of view. Refer to the PEPSSI instrument kernel, nh_pepssi.ti, for
   more details on the rotation angles and physical locations of sectors S0
   through S5.

            \begindata

            FRAME_NH_PEPSSI_S0      = -98402
            FRAME_-98402_NAME       = 'NH_PEPSSI_S0'
            FRAME_-98402_CLASS      = 4
            FRAME_-98402_CLASS_ID   = -98402
            FRAME_-98402_CENTER     = -98
            TKFRAME_-98402_SPEC     = 'MATRIX'
            TKFRAME_-98402_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98402_MATRIX   = (0.382683432365090
                                        0.000000000000000
                                        0.923879532511287
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.923879532511287
                                        0.000000000000000
                                        0.382683432365090 )

            FRAME_NH_PEPSSI_S1      = -98403
            FRAME_-98403_NAME       = 'NH_PEPSSI_S1'
            FRAME_-98403_CLASS      = 4
            FRAME_-98403_CLASS_ID   = -98403
            FRAME_-98403_CENTER     = -98
            TKFRAME_-98403_SPEC     = 'MATRIX'
            TKFRAME_-98403_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98403_MATRIX   = (0.760405965600031
                                        0.000000000000000
                                        0.649448048330184
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.649448048330184
                                        0.000000000000000
                                        0.760405965600031 )

            FRAME_NH_PEPSSI_S2      = -98404
            FRAME_-98404_NAME       = 'NH_PEPSSI_S2'
            FRAME_-98404_CLASS      = 4
            FRAME_-98404_CLASS_ID   = -98404
            FRAME_-98404_CENTER     = -98
            TKFRAME_-98404_SPEC     = 'MATRIX'
            TKFRAME_-98404_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98404_MATRIX   = (0.972369920397677
                                        0.000000000000000
                                        0.233445363855905
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.233445363855905
                                        0.000000000000000
                                        0.972369920397677 )

            FRAME_NH_PEPSSI_S3      = -98405
            FRAME_-98405_NAME       = 'NH_PEPSSI_S3'
            FRAME_-98405_CLASS      = 4
            FRAME_-98405_CLASS_ID   = -98405
            FRAME_-98405_CENTER     = -98
            TKFRAME_-98405_SPEC     = 'MATRIX'
            TKFRAME_-98405_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98405_MATRIX   = (0.972369920397677
                                        0.000000000000000
                                       -0.233445363855905
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.233445363855905
                                        0.000000000000000
                                        0.972369920397677 )

            FRAME_NH_PEPSSI_S4      = -98406
            FRAME_-98406_NAME       = 'NH_PEPSSI_S4'
            FRAME_-98406_CLASS      = 4
            FRAME_-98406_CLASS_ID   = -98406
            FRAME_-98406_CENTER     = -98
            TKFRAME_-98406_SPEC     = 'MATRIX'
            TKFRAME_-98406_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98406_MATRIX   = (0.760405965600031
                                        0.000000000000000
                                       -0.649448048330184
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.649448048330184
                                        0.000000000000000
                                        0.760405965600031 )

            FRAME_NH_PEPSSI_S5      = -98407
            FRAME_-98407_NAME       = 'NH_PEPSSI_S5'
            FRAME_-98407_CLASS      = 4
            FRAME_-98407_CLASS_ID   = -98407
            FRAME_-98407_CENTER     = -98
            TKFRAME_-98407_SPEC     = 'MATRIX'
            TKFRAME_-98407_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98407_MATRIX   = (0.382683432365090
                                        0.000000000000000
                                       -0.923879532511287
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.923879532511287
                                        0.000000000000000
                                        0.382683432365090 )

           \begintext


   The following detector frames were determined by rotating the NH_PEPSSI
   frame about the NH_PEPSSI Y axis by the number of degrees required to move
   the NH_PEPSSI boresight (NH_PEPSSI +Z axis) into the center of the
   detector's field of view. Refer to the PEPSSI instrument kernel,
   nh_pepssi.ti, for more details on the rotation angles and physical
   locations of detectors D0 through D11.

           \begindata

            FRAME_NH_PEPSSI_D0      = -98408
            FRAME_-98408_NAME       = 'NH_PEPSSI_D0'
            FRAME_-98408_CLASS      = 4
            FRAME_-98408_CLASS_ID   = -98408
            FRAME_-98408_CENTER     = -98
            TKFRAME_-98408_SPEC     = 'MATRIX'
            TKFRAME_-98408_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98408_MATRIX   = (0.279829014030992
                                        0.000000000000000
                                        0.960049854385929
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.960049854385929
                                        0.000000000000000
                                        0.279829014030992 )

            FRAME_NH_PEPSSI_D1      = -98409
            FRAME_-98409_NAME       = 'NH_PEPSSI_D1'
            FRAME_-98409_CLASS      = 4
            FRAME_-98409_CLASS_ID   = -98409
            FRAME_-98409_CENTER     = -98
            TKFRAME_-98409_SPEC     = 'MATRIX'
            TKFRAME_-98409_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98409_MATRIX   = (0.480988768919388
                                        0.000000000000000
                                        0.876726755707508
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.876726755707508
                                        0.000000000000000
                                        0.480988768919388 )

            FRAME_NH_PEPSSI_D2      = -98410
            FRAME_-98410_NAME       = 'NH_PEPSSI_D2'
            FRAME_-98410_CLASS      = 4
            FRAME_-98410_CLASS_ID   = -98410
            FRAME_-98410_CENTER     = -98
            TKFRAME_-98410_SPEC     = 'MATRIX'
            TKFRAME_-98410_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98410_MATRIX   = (0.685182990326359
                                        0.000000000000000
                                        0.728370969882400
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.728370969882400
                                        0.000000000000000
                                        0.685182990326359 )

            FRAME_NH_PEPSSI_D3      = -98411
            FRAME_-98411_NAME       = 'NH_PEPSSI_D3'
            FRAME_-98411_CLASS      = 4
            FRAME_-98411_CLASS_ID   = -98411
            FRAME_-98411_CENTER     = -98
            TKFRAME_-98411_SPEC     = 'MATRIX'
            TKFRAME_-98411_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98411_MATRIX   = (0.826589749127189
                                        0.000000000000000
                                        0.562804927695069
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.562804927695069
                                        0.000000000000000
                                        0.826589749127189 )

            FRAME_NH_PEPSSI_D4      = -98412
            FRAME_-98412_NAME       = 'NH_PEPSSI_D4'
            FRAME_-98412_CLASS      = 4
            FRAME_-98412_CLASS_ID   = -98412
            FRAME_-98412_CENTER     = -98
            TKFRAME_-98412_SPEC     = 'MATRIX'
            TKFRAME_-98412_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98412_MATRIX   = (0.941176015256371
                                        0.000000000000000
                                        0.337916718003327
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.337916718003327
                                        0.000000000000000
                                        0.941176015256371 )

            FRAME_NH_PEPSSI_D5      = -98413
            FRAME_-98413_NAME       = 'NH_PEPSSI_D5'
            FRAME_-98413_CLASS      = 4
            FRAME_-98413_CLASS_ID   = -98413
            FRAME_-98413_CENTER     = -98
            TKFRAME_-98413_SPEC     = 'MATRIX'
            TKFRAME_-98413_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98413_MATRIX   = (0.992004949679715
                                        0.000000000000000
                                        0.126198969135830
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                       -0.126198969135830
                                        0.000000000000000
                                        0.992004949679715 )

            FRAME_NH_PEPSSI_D6      = -98414
            FRAME_-98414_NAME       = 'NH_PEPSSI_D6'
            FRAME_-98414_CLASS      = 4
            FRAME_-98414_CLASS_ID   = -98414
            FRAME_-98414_CENTER     = -98
            TKFRAME_-98414_SPEC     = 'MATRIX'
            TKFRAME_-98414_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98414_MATRIX   = (0.992004949679715
                                        0.000000000000000
                                       -0.126198969135830
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.126198969135830
                                        0.000000000000000
                                        0.992004949679715 )

            FRAME_NH_PEPSSI_D7      = -98415
            FRAME_-98415_NAME       = 'NH_PEPSSI_D7'
            FRAME_-98415_CLASS      = 4
            FRAME_-98415_CLASS_ID   = -98415
            FRAME_-98415_CENTER     = -98
            TKFRAME_-98415_SPEC     = 'MATRIX'
            TKFRAME_-98415_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98415_MATRIX   = (0.941176015256371
                                        0.000000000000000
                                       -0.337916718003327
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.337916718003327
                                        0.000000000000000
                                        0.941176015256371 )

            FRAME_NH_PEPSSI_D8      = -98416
            FRAME_-98416_NAME       = 'NH_PEPSSI_D8'
            FRAME_-98416_CLASS      = 4
            FRAME_-98416_CLASS_ID   = -98416
            FRAME_-98416_CENTER     = -98
            TKFRAME_-98416_SPEC     = 'MATRIX'
            TKFRAME_-98416_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98416_MATRIX   = (0.826589749127189
                                        0.000000000000000
                                       -0.562804927695069
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.562804927695069
                                        0.000000000000000
                                        0.826589749127189 )

            FRAME_NH_PEPSSI_D9      = -98417
            FRAME_-98417_NAME       = 'NH_PEPSSI_D9'
            FRAME_-98417_CLASS      = 4
            FRAME_-98417_CLASS_ID   = -98417
            FRAME_-98417_CENTER     = -98
            TKFRAME_-98417_SPEC     = 'MATRIX'
            TKFRAME_-98417_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98417_MATRIX   = (0.685182990326359
                                        0.000000000000000
                                       -0.728370969882400
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.728370969882400
                                        0.000000000000000
                                        0.685182990326359 )

            FRAME_NH_PEPSSI_D10     = -98418
            FRAME_-98418_NAME       = 'NH_PEPSSI_D10'
            FRAME_-98418_CLASS      = 4
            FRAME_-98418_CLASS_ID   = -98418
            FRAME_-98418_CENTER     = -98
            TKFRAME_-98418_SPEC     = 'MATRIX'
            TKFRAME_-98418_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98418_MATRIX   = (0.480988768919388
                                        0.000000000000000
                                       -0.876726755707508
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.876726755707508
                                        0.000000000000000
                                        0.480988768919388 )

            FRAME_NH_PEPSSI_D11     = -98419
            FRAME_-98419_NAME       = 'NH_PEPSSI_D11'
            FRAME_-98419_CLASS      = 4
            FRAME_-98419_CLASS_ID   = -98419
            FRAME_-98419_CENTER     = -98
            TKFRAME_-98419_SPEC     = 'MATRIX'
            TKFRAME_-98419_RELATIVE = 'NH_PEPSSI'
            TKFRAME_-98419_MATRIX   = (0.279829014030992
                                        0.000000000000000
                                       -0.960049854385929
                                        0.000000000000000
                                        1.000000000000000
                                        0.000000000000000
                                        0.960049854385929
                                        0.000000000000000
                                        0.279829014030992 )

           \begintext


Radio Science Experiment (REX) Frames
-------------------------------------------------------------------------------

   We are defining the REX coordinate system such that the +Z axis in
   instrument coordinates is the boresight, the +X axis in instrument
   coordinates is aligned with the spacecraft +X axis, and the +Y axis in
   instrument coordinates is aligned with the spacecraft -Z axis. The
   measured boresight vector is given in [9] as

                                           [  0.0000823 ]
           REX Boresight Vector (+Z    ) = [  1.0       ]
                                   inst    [ -0.0001249 ]

   Since we defined the instrument +X axis to be the spacecraft +X axis, we
   have:

                           [ 1.0 ]
           REX +X Vector = [ 0.0 ]
                           [ 0.0 ]

   The instrument +Y vector is determined by taking the cross product Z x X:

                           [  0.0               ]
           REX +Y Vector = [ -0.000124900004202 ]
                           [ -0.999999992199995 ]

   And we use that to adjust the +X vector to form an orthogonal frame:

                                   [  0.999999996613355 ]
           REX +X Vector = Y x Z = [ -0.000082299999378 ]
                                   [  0.000000010279270 ]

   Using these three vectors, we define the rotation matrix that takes vectors
   from the instrument frame to the spacecraft frame as

           [     ]    [  0.999999996613355  0.0                0.0000823 ]
           [ ROT ]  = [ -0.000082299999378 -0.000124900004202  1.0       ]
           [     ]    [  0.000000010279270 -0.999999992199995 -0.0001249 ]

           \begindata

           FRAME_NH_REX             = -98500
           FRAME_-98500_NAME        = 'NH_REX'
           FRAME_-98500_CLASS       = 4
           FRAME_-98500_CLASS_ID    = -98500
           FRAME_-98500_CENTER      = -98
           TKFRAME_-98500_SPEC      = 'MATRIX'
           TKFRAME_-98500_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98500_MATRIX    = (  0.999999996613355,
                                        -0.000082299999378,
                                         0.000000010279270,
                                         0.0,
                                        -0.000124900004202,
                                        -0.999999992199995,
                                         0.0000823,
                                         1.0,
                                        -0.0001249 )

           \begintext


Solar Wind Around Pluto (SWAP) Frames
-------------------------------------------------------------------------------

   From [20], SWAP is mounted on the -Z panel of the New Horizons Spacecraft
   and uses the same coordinate system as the spacecraft. The rotation matrix
   that takes vectors represented in the nominal SWAP frame into the spacecraft
   frame is the identity. We will adjust the rotation to take into account the
   measured alignment provided in [9]:

                                     [  0.9999672 ]
           SWAP Measured +X Vector = [  0.0080965 ]
                                     [ -0.0000296 ]

                                     [  0.0000469 ]
           SWAP Measured +Z Vector = [ -0.0021315 ]
                                     [  0.9999977 ]

   Taking the cross product of these two vectors gives us:

                                                [ -0.0080964188 ]
           SWAP +Y Vector (Boresight) = Z x X = [  0.9999649500 ]
                                                [  0.0021318100 ]

   And we use that to adjust the Z vector to form an orthogonal frame:

                                    [  0.000046859163 ]
           SWAP +Z Vector = X x Y = [ -0.002131500500 ]
                                    [  0.999997730000 ]

           [     ]    [  0.9999672 -0.0080964188  0.000046859163 ]
           [ ROT ]  = [  0.0080965  0.9999649500 -0.002131500500 ]
           [     ]    [ -0.0000296  0.0021318100  0.999997730000 ]

           \begindata

           FRAME_NH_SWAP            = -98600
           FRAME_-98600_NAME        = 'NH_SWAP'
           FRAME_-98600_CLASS       = 4
           FRAME_-98600_CLASS_ID    = -98600
           FRAME_-98600_CENTER      = -98
           TKFRAME_-98600_SPEC      = 'MATRIX'
           TKFRAME_-98600_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98600_MATRIX    = ( 0.9999672,
                                        0.0080965,
                                       -0.0000296,
                                       -0.0080964188,
                                        0.9999649500,
                                        0.0021318100,
                                        0.000046859163,
                                       -0.002131500500,
                                        0.999997730000 )

           \begintext


Student Dust Counter (SDC) Frames
-------------------------------------------------------------------------------

   The SDC coordinate system is defined [16] such that the boresight is the
   instrument +Z axis. This corresponds to the spacecraft -Y axis. The
   instrument X axis is defined such that it is along the detector's long
   dimension, with instrument +X corresponding to spacecraft +Z.  The
   instrument Y axis is defined such that it is along the detector's short
   dimension, with instrument +Y corresponding to the spacecraft -X axis. The
   rotation matrix that takes vectors from the instrument frame to the
   spacecraft frame is then:

           [     ]    [ 0 -1  0 ]
           [ ROT ]  = [ 0  0 -1 ]
           [     ]    [ 1  0  0 ]

           \begindata

           FRAME_NH_SDC             = -98700
           FRAME_-98700_NAME        = 'NH_SDC'
           FRAME_-98700_CLASS       = 4
           FRAME_-98700_CLASS_ID    = -98700
           FRAME_-98700_CENTER      = -98
           TKFRAME_-98700_SPEC      = 'MATRIX'
           TKFRAME_-98700_RELATIVE  = 'NH_ASTR'
           TKFRAME_-98700_MATRIX    = ( 0.0,
                                        0.0,
                                        1.0,
                                       -1.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                       -1.0,
                                        0.0 )

           \begintext


