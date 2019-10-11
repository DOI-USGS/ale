KPL/FK

Trace Gas Orbiter (TGO) Spacecraft Frames Kernel
===============================================================================

   This frame kernel contains a complete set of frame definitions for the
   ExoMars 2016 Trace Gas Orbiter Spacecraft (TGO) including definitions for
   the TGO structures and TGO science instrument frames. This kernel also
   contains NAIF ID/name mapping for the TGO instruments.


Version and Date
------------------------------------------------------------------------

   Version 1.8 -- August 8, 2019 --  Marc Costa Sitja, ESAC/ESA
                                     Boris Semenov, NAIF/JPL

      Corrected TGO_ACS_TIRVIM_SCAN_BASE frame to TGO_ACS_TIRVIM_SCAN.
      Removed TGO_NOMAD_LNO_SCAN references in favour of TGO_NOMAD_LNO_FMM.
      Renamed TGO_ACS_TIRVIM_SCAN_OCC_BSR references to
      TGO_ACS_TIRVIM_SCAN_OCC_BS. Corrected several typos and diagrams.

   Version 1.7 -- March 6, 2019 --  Marc Costa Sitja, ESAC/ESA
                                    Marty Jean-Charles, CNES

      Updated SA+Z_ZERO and SA-Z_ZERO frames; the normal of the SA was
      antiparallel to the Sun vector.

   Version 1.6 -- July 3, 2018 --  Marc Costa Sitja, ESAC/ESA
                                   Nikolay Ignatiev, IKI

      Corrected typo in TGO_ACS_TIRVIM_SUN frame definition and added
      TGO_ACS_TIRVIM_SUN_BSR ID.

   Version 1.5 -- June 18, 2018 --  Marc Costa Sitja, ESAC/ESA

      Updated NOMAD UVIS Occultation and NOMAD SO Boresight alignments.

   Version 1.4 -- May 15, 2018 -- Stepanov Tulyakov, EPFL
                                  Marc Costa Sitja, ESAC/ESA

      Updated CaSSIS rotations of CRU with respect to SC and FSA with
      respect to TEL.
      Added LGAs reference frames, corrected STR IDs and added
      Structures IDs.

   Version 1.3 -- March 29, 2018 -- Marc Costa Sitja, ESAC/ESA
                                    Boris Semenov, NAIF/JPL

      Added Star Tracker reference frames and spacecraft reference
      frame for planning CKs.

   Version 1.2 -- July 20, 2017 -- Marc Costa Sitja, ESAC/ESA

      Corrected typo in NOMAD UVIS reference frame.

   Version 1.1 -- July 20, 2017 -- Marc Costa Sitja, ESAC/ESA
                                   Bernhard Geiger, ESAC/ESA

      Updated ACS frame definitions, added TIRVIM frames and
      implemented boresight misalignments.
      Corrected typos in NOMAD frame and ID descriptions.

   Version 1.0 -- January 15, 2017 -- Marc Costa Sitja, ESAC/ESA

      Corrected several HGA IDs from -226 to -143.

   Version 0.9 -- December 15, 2016 -- Marc Costa Sitja, ESAC/ESA

      Updated NOMAD_SO definitions with Alejandro Cardesin.

   Version 0.8 -- September 26, 2016 -- Marc Costa Sitja, ESAC/ESA

      Corrected NOMAD UVIS Nadir Boresight misalignment.
      Renamed the NOMAD LNO Scan mirror frame (TGO_NOMAD_LNO_SCAN) to
      NOMAD LNO Flip Mirror Mechanism frame (TGO_NOMAD_LNO_FMM).
      Renamed the NOMAD LNO nadir and occultation frames
      (TGO_NOMAD_LNO_*) to NOMAD LNO nadir and occultation operations
      frames (TGO_NOMAD_LNO_OPS_*).
      Minor edits to some text after NOMAD instrument team review.
      Added reference [18].

   Version 0.7 -- August 11, 2016 -- Marc Costa Sitja, ESAC/ESA
                                     Bernhard Geiger, ESAC/ESA
                                     Alejandro Cardesin, ESAC/ESA

      Added references [13], [14], [15], [16] and [17].
      Updated Spacecraft drawings with Main Engine.
      Corrected TGO Solar Array Frames and TGO High Gain Antenna
      definitions as described in reference [13].
      Separated the ACS NIR and the TRIVIM frames definitions sections.
      Corrected rotation of the CaSSIS Filter Strip Assembly Frame.
      Updated the NAIF IDs for CaSSIS definitions.
      Corrected several typos and updated text and diagram information.
      Updated NOMAD frames orientation with information provided by
      I. Thomas ([15]).
      Removed references to flip/scanning mirror in ACS NIR channel
      frame definitions.
      Updated ACS frames orientation with information provided by
      A. Trokhimovskiy ([16]).
      Corrected TGO_ACS_TIRVIM_SCAN frame definition.

   Version 0.6 -- June 6, 2016 -- Jorge Diaz del Rio, ODC Space

      Updated comments to exchange the Science Operations Frame Definitions
      Kernel file name by the description of the data.

   Version 0.5 -- May 31, 2016 -- Jorge Diaz del Rio, ODC Space

      TGO_SA*_GIMBAL frame renamed to TGO_SA*_ZERO. Corrected typos in
      comments. Added ``EXOMARS 2016 TGO'' and ``TRACE GAS ORBITER'' as
      synonyms for TGO.

   Version 0.4 -- May 20, 2016 -- Jorge Diaz del Rio, ODC Space
                                  Anton Ledkov, IKI

      Preliminary Version. Added ACS and NOMAD frames. Modified FREND
      frame chain upon request from FREND Instrument Team. Corrected
      CaSSIS CRU and FSA frames orientations. Added CaSSIS filter
      name/ID mappings. Added SA and HGA frames. Added list of science
      operations frames.

   Version 0.3 -- March 17, 2016 -- Jorge Diaz del Rio, ODC Space

      Preliminary Version. Corrected rotations in TGO_CASSIS_CRU and in
      TGO_CASSIS_FSA frame definitions. Added FREND frames.

   Version 0.2 -- March 11, 2016 -- Jorge Diaz del Rio, ODC Space

      Preliminary Version. Added CaSSIS frames.

   Version 0.1 -- January 26, 2016 --  Jorge Diaz del Rio, ODC Space

      Preliminary Version. Added TGO_SPACECRAFT frame for its use with
      test CK kernel.

   Version 0.0 -- December 17, 2015 -- Jorge Diaz del Rio, ODC Space

      Preliminary Version. Only TGO Name to NAIF ID mappings for their
      use with test SPK kernel.

   Version 0.0-draft -- May 26, 2015 -- Anton Ledkov, IKI

      Draft Version.


References
------------------------------------------------------------------------

    1. ``Frames Required Reading'', NAIF

    2. ``Kernel Pool Required Reading'', NAIF

    3. ``C-Kernel Required Reading'', NAIF

    4. ``ExoMars: Science Operations Centre - Flight Dynamics - Pointing
       Timeline-ICD,'' EXM-GS-ICD-ESC-50003, Issue 1.4, 15 December 2015

    5. ``CaSSIS Rotation Axis Determination Report'', EXM-CA-TRE-UBE-00112
       Issue 0.7, 1 March 2016

    6. ``The Color and Stereo Surface Imaging System (CaSSIS) for ESA's
       Trace Gas Orbiter.'' Eighth International Conference on Mars (2014)
       N. Thomas et al.

    7. ``FREND Mechanical ICD Drawings,'' EXM-FR-DRW-IKI-0020, Issue 1.2,
       1 March 2015

    8. Email from FREND PM (Alexey Malakhov) on March 17, 2016 (Re. TGO FREND
       FK/IK approach)

    9. ``High Resolution Middle Infrared Spectrometer, a Part of Atmospheric
       Chemistry Suite (ACS) for ExoMars 2016 Trace Gas Orbiter'',
       International Conference on Space Optics, Tenerife 7-10 October 2014

   10. ``NOMAD Experiment ICD'', EXM-PL-ICD-ESA-00025, Issue 2.7 2014-09-23

   11. ``Atmospheric Chemistry Suite (ACS): a Set of Infrared Spectrometers
       for Atmospheric Measurements onboard ExoMars Trace Gas Orbiter'', A.
       Trokhimovskiy et al.

   12. TGO Science Operations Frames Definition Kernel, latest version.

   13. ``EXOMARS OMB frame definitions and conventions'',
       EXM-OM-TNO-AF-0361, Issue 3, 2011-10-14, Thales Alenia Space.

   14. ``EXOMARS Spacecraft Mechanical Interface Control Document'',
       EXM-MS-ICD-AI-0019, Issue 12, 2015-08-10, Thales Alenia Space.

   15. Email from Ian Thomas <ian.thomas@aeronomie.be> ``[EM16-SOC]
       [TGO] [SGS] [EM16.NOMAD] SPICE review and misalignment update by
       25th July'' on 25 July 2016.

   16. Email from Alexander Trokhimovskiy <a.trokh@gmail.com>
       ``Re: [EM16-SOC] [TGO] [SGS] [EM16.ACS] [EM16.NOMAD] SPICE review
       and misalignment update by 25th July'' on 27 July 2016.

   17. Email from Alexander Trokhimovskiy <a.trokh@gmail.com>
       ``Re: [EM16-SOC] [TGO] [SGS] [EM16.ACS] [EM16.NOMAD] SPICE review
       and misalignment update by 25th July'' on 11 August 2016.

   18. Email from Ian Thomas <ian.thomas@aeronomie.be> ``UVIS Nadir
       Boresight Correction'' on 16 September 2016.

   19. ``Boresight Alignment'', ExoMars 2016 Confluence Page,
       https://issues.cosmos.esa.int/exomarswiki/display/OE/Boresight+Alignment
       Bernhard Geiger (ESAC/ESA), accessed on 19th July 2017.

   20. Email from Bernhard Geiger (ESAC/ESA) ``TGO Star Tracker boresights and
       frames'' on 7th November 2017.

   21. ``ICD, EXOMARS HGA-A'', EXM-OM-ICD-MDA-0041-7, Louis-Philippe Lebel,
       MacDonald, Dettwiler and Associates Corporation on 21st January 2015.

   22. Email from Ian Thomas <ian.thomas@aeronomie.be>``New SO and UVIS solar
       boresight vectors' on 13 June 2018.


Contact Information
------------------------------------------------------------------------

   If you have any questions regarding this file contact the
   ESA SPICE Service (ESS) at ESAC:

           Marc Costa Sitja
           (+34) 91-8131-457
           esa_spice@sciops.esa.int, marc.costa@esa.int

   or SPICE support at IKI:

           Alexander Abbakumov
           +7 (495) 333-40-13
           aabbakumov@romance.iki.rssi.ru

   or NAIF at JPL:

           Boris Semenov
           +1 (818) 354-8136
           Boris.Semenov@jpl.nasa.gov


Implementation Notes
------------------------------------------------------------------------

  This file is used by the SPICE system as follows: programs that make use
  of this frame kernel must "load" the kernel normally during program
  initialization. Loading the kernel associates the data items with
  their names in a data structure called the "kernel pool".  The SPICELIB
  routine FURNSH loads a kernel into the pool as shown below:

    FORTRAN: (SPICELIB)

      CALL FURNSH ( frame_kernel_name )

    C: (CSPICE)

      furnsh_c ( frame_kernel_name );

    IDL: (ICY)

      cspice_furnsh, frame_kernel_name

    MATLAB: (MICE)

         cspice_furnsh ( 'frame_kernel_name' )

    PYTHON: (SPICEYPY)*

         furnsh( frame_kernel_name )

  In order for a program or routine to extract data from the pool, the
  SPICELIB routines GDPOOL, GIPOOL, and GCPOOL are used.  See [2] for
  more details.

  This file was created and may be updated with a text editor or word
  processor.

  * SPICEPY is a non-official, community developed Python wrapper for the
    NAIF SPICE toolkit. Its development is managed on Github.
    It is available at: https://github.com/AndrewAnnex/SpiceyPy


TGO NAIF ID Codes -- Summary Section
------------------------------------------------------------------------

   The following names and NAIF ID codes are assigned to the TGO spacecraft,
   its structures and science instruments (the keywords implementing these
   definitions are located in the section "TGO NAIF ID Codes -- Definition
   Section" at the end of this file):

   TGO Spacecraft and Spacecraft Structures names/IDs:

            TGO                      -143     (synonyms: EXOMARS 2016 TGO,
                                                         EXOMARS TGO,
                                                    and  TRACE GAS ORBITER)
            TGO_STR-1                -143031
            TGO_STR-2                -143032

   ACS names/IDs:

            TGO_ACS                  -143100
            TGO_ACS_NIR_NAD          -143111
            TGO_ACS_NIR_OCC          -143112
            TGO_ACS_MIR              -143120
            TGO_ACS_TIRVIM           -143130
            TGO_ACS_TIRVIM_BBY       -143131
            TGO_ACS_TIRVIM_SPC       -143132
            TGO_ACS_TIRVIM_NAD       -143133
            TGO_ACS_TIRVIM_OCC       -143134
            TGO_ACS_TIRVIM_SUN       -143140

   FREND names/IDs:

            TGO_FREND                -143200
            TGO_FREND_HE             -143210
            TGO_FREND_SC             -143220

   NOMAD names/IDs:

            TGO_NOMAD                -143300
            TGO_NOMAD_LNO            -143310
            TGO_NOMAD_LNO_OPS_NAD    -143311
            TGO_NOMAD_LNO_OPS_OCC    -143312
            TGO_NOMAD_SO             -143320
            TGO_NOMAD_UVIS_NAD       -143331
            TGO_NOMAD_UVIS_OCC       -143332

   CaSSIS names/IDs:

            TGO_CASSIS               -143400
            TGO_CASSIS_PAN           -143421
            TGO_CASSIS_RED           -143422
            TGO_CASSIS_NIR           -143423
            TGO_CASSIS_BLU           -143424


TGO Frames
------------------------------------------------------------------------

   The following TGO frames are defined in this kernel file:

        Name                      Relative to                 Type    NAIF ID
      ======================    ==========================   =======  =======

   TGO Spacecraft and Spacecraft Structures frames:
   ------------------------------------------------
      TGO_SPACECRAFT              J2000,                     CK       -143000
                                  TGO_SPACECRAFT_PLAN
      TGO_SPACECRAFT_PLAN         J2000                      CK       -143002
      TGO_SA+Z_ZERO               TGO_SPACECRAFT             FIXED    -143010
      TGO_SA+Z                    TGO_SA+Z_ZERO              CK       -143011
      TGO_SA-Z_ZERO               TGO_SPACECRAFT             FIXED    -143012
      TGO_SA-Z                    TGO_SA-Z_ZERO              CK       -143013
      TGO_HGA_EL                  TGO_SPACECRAFT             CK       -143021
      TGO_HGA_AZ                  TGO_HGA_EL                 CK       -143022
      TGO_HGA                     TGO_HGA_AZ                 FIXED    -143025
      TGO_LGA+Z                   TGO_SPACECRAFT             FIXED    -143031
      TGO_LGA-Z                   TGO_SPACECRAFT             FIXED    -143032
      TGO_LGA-X                   TGO_SPACECRAFT             FIXED    -143033
      TGO_STR-1                   TGO_SPACECRAFT             FIXED    -143041
      TGO_STR-2                   TGO_SPACECRAFT             FIXED    -143042

   ACS frames:
   -----------
      TGO_ACS_NIR_BASE            TGO_SPACECRAFT             FIXED    -143100
      TGO_ACS_NIR_NAD             TGO_ACS_NIR_BASE           FIXED    -143111
      TGO_ACS_NIR_OCC             TGO_ACS_NIR_BASE           FIXED    -143112
      TGO_ACS_MIR                 TGO_SPACECRAFT             FIXED    -143120
      TGO_ACS_TIRVIM_BASE         TGO_SPACECRAFT             FIXED    -143105
      TGO_ACS_TIRVIM_SCAN_ROT     TGO_ACS_TIRVIM_BASE        FIXED    -143106
      TGO_ACS_TIRVIM_SCAN         TGO_ACS_TIRVIM_SCAN_ROT    CK       -143107
      TGO_ACS_TIRVIM              TGO_ACS_TIRVIM_SCAN        FIXED    -143130
      TGO_ACS_TIRVIM_SCAN_BBY     TGO_ACS_TIRVIM_BASE        FIXED    -143131
      TGO_ACS_TIRVIM_SCAN_SPC     TGO_ACS_TIRVIM_BASE        FIXED    -143132
      TGO_ACS_TIRVIM_SCAN_NAD     TGO_ACS_TIRVIM_BASE        FIXED    -143133
      TGO_ACS_TIRVIM_SCAN_OCC     TGO_ACS_TIRVIM_BASE        FIXED    -143134
      TGO_ACS_TIRVIM_SCAN_OCC_BS  TGO_ACS_TIRVIM_BASE        FIXED    -143135
      TGO_ACS_TIRVIM_SUN          TGO_ACS_TIRVIM_BASE        FIXED    -143140
      TGO_ACS_TIRVIM_SUN_BSR      TGO_ACS_TIRVIM_BASE        FIXED    -143141

   FREND frames:
   -------------
      TGO_FREND                   TGO_SPACECRAFT             FIXED    -143200

   NOMAD frames:
   -------------
      TGO_NOMAD_LNO_BASE          TGO_SPACECRAFT             FIXED    -143300
      TGO_NOMAD_LNO_FMM           TGO_NOMAD_LNO_BASE         CK       -143305
      TGO_NOMAD_LNO               TGO_NOMAD_LNO_BASE         FIXED    -143310
      TGO_NOMAD_LNO_OPS_NAD       TGO_NOMAD_LNO_BASE         FIXED    -143311
      TGO_NOMAD_LNO_OPS_OCC       TGO_NOMAD_LNO_BASE         FIXED    -143312
      TGO_NOMAD_SO                TGO_SPACECRAFT             FIXED    -143320
      TGO_NOMAD_UVIS_BASE         TGO_SPACECRAFT             FIXED    -143330
      TGO_NOMAD_UVIS_NAD          TGO_NOMAD_UVIS_BASE        FIXED    -143331
      TGO_NOMAD_UVIS_OCC          TGO_NOMAD_UVIS_BASE        FIXED    -143332

   CaSSIS frames:
   --------------
      TGO_CASSIS_CRU              TGO_SPACECRAFT             FIXED    -143400
      TGO_CASSIS_TEL              TGO_CASSIS_CRU             CK       -143410
      TGO_CASSIS_FSA              TGO_CASSIS_TEL             FIXED    -143420


   In addition, the following frames, in use by the ExoMars 2016 mission, are
   defined in another kernel:

        Name                      Relative to                 Type    NAIF ID
      ======================    ==========================   =======  =======

   ExoMars 2016 mission science operations frames (1):
   ---------------------------------------------------
      TGO_MARS_NPO                J2000                    DYNAMIC   -143900

      (1) This frame is defined in the ExoMars 2016 Science Operations
          Frame Definitions kernel file (see [12]). This frame can be used
          'as is' or to define default TGO attitude profiles. In order
          to use it for the latter together with this frames kernel,
          additional fixed-offset frames kernel(s) need to be loaded. See
          the section ``Using this frame'' in the comment area of this file
          for further details.


ExoMars 2016 Frames Hierarchy
--------------------------------------------------------------------------

   The diagram below shows the ExoMars 2016 frames hierarchy (except
   for science operations frames):


                               "J2000" INERTIAL
           +-----------------------------------------------------+
           |                          |      |                   |
           |<-pck                     |      |                   |<-pck
           |                          |      |                   |
           v                          |      |                   v
       "IAU_MARS"                     |      |              "IAU_EARTH"
     MARS BODY-FIXED                  |      |<-ck        EARTH BODY-FIXED
     ---------------                  |      |            ----------------
                                      |      v
       "TGO_LGA+Z"    "TGO_LGA+X"     |   "TGO_SPACECRAFT_PLAN"
       -----------    -----------     |   ---------------------
           ^               ^          |      |
           |               |          |<-ck  |<-ck
           |<-fixed        |<-fixed   |      |
           |               |          |      |
           |  "TGO_LGA-Z"  |          |      |
           |  -----------  |          |      |
           |      ^        |          |      |
           |      |        |          |      |
           |      |<-fixed |          v      v
           |      |        |     "TGO_SPACECRAFT"
           +------------------------------------------------------------+
           |              |           .      |          |               |
           |<-fixed       |<-fixed    .      |<-fixed   |<-fixed    ck->|
           |              |           .      |          |               |
           v              |           .      v          v               v
   "TGO_SA+Z_ZERO"        |           . "TGO_STR-1"  "TGO_STR-2"  "TGO_HGA_EL"
   ---------------        |           . -----------  -----------  ------------
           |              |           .                                 |
           |<-ck          v           .                             ck->|
           |      "TGO_SA-Z_ZERO"     .                                 |
           v      ---------------     .                                 v
      "TGO_SA+Z"          |           .                           "TGO_HGA_AZ"
      ----------          |<-ck       .                           ------------
                          |           .                                 |
                          v           .                          fixed->|
                     "TGO_SA-Z"       .                                 |
                     ----------       .                                 v
                                      .                             "TGO_HGA"
                                      .                             ---------
                                      V
                Individual instrument frame trees are provided
                      in the other sections of this file


   Please refer to the ACS, CaSSIS, FREND and NOMAD sections for the frame
   hierarchy of each payload; and to the TGO science operations frame
   definitions kernel [12] for further details on these frame definitions.


TGO Spacecraft and Spacecraft Structures Frames
------------------------------------------------------------------------

   This section of the file contains the definitions of the spacecraft
   and spacecraft structures frames.

   DISCLAIMER: The origin of the frames specified in the following
   definitions are not implemented. The ``true'' origin of all frames
   is in the center of the TGO_SPACECRAFT frame, the center of which
   is defined by the position given by the SPK (ephemeris) kernel in
   use.


TGO Spacecraft Frames
--------------------------------------

   According to [4] the TGO spacecraft reference frame -- TGO_SPACECRAFT --
   is defined as follows:

      -  +X axis is perpendicular to the launch vehicle interface plane
         and points towards the Schiaparelli Entry, Descent and Landing
         Demonstrator Module (EDM) attachment point;

      -  -Y axis is perpendicular to the payload Science Deck and points
         towards the payload side; representing the reference spacecraft
         line of sight towards Mars during science operations;

      -  +Z axis completes the right-handed frame.

      -  the origin of this frame is located at the centre of the launch
         vehicle interface ring: at the bottom of the interface cylinder and
         the top of the launch vehicle specific interface frame.


   These diagrams illustrate the TGO_SPACECRAFT frame:

   -X S/C side (Main Engine side) view:
   ------------------------------------

                                    ^
                                    | toward Mars
                                    |

                               Science Deck
                             ._____________.
   .__  _______________.     |             |     .______________  ___.
   |  \ \               \    |             |    /               \ \  |
   |  / /                \   |     ___     |   /                / /  |
   |  \ \               +Zsc |    / _ +Xsc | .'                 \ \  |
   |  / /                  <--------x)     |o |                 / /  |
   |  \ \                 .' |    \_|_/    | `.                 \ \  |
   |  / /                /   | Main |Engine|   \                / /  |
   .__\ \_______________/    | (ME) |      |    \_______________\ \__.
       +Z Solar Array        ._____ v +Ysc .      -Z Solar Array
                                 ._____.
                               .'       `.
                              /           \
                             .   `.   .'   .             +Xsc is into
                             |     `o'     |              the page.
                             .      |      .
                              \     |     /
                               `.       .'
                            HGA  ` --- '


   -Y S/C side (Science Deck side) view:
   -------------------------------------
                                  _____
                                 /     \  EDM
                                |       |
                             ._____________.
                             |   |     |   |
                             |nom|     |acs|
                             |___'     '___|
   o==/ /==================o<|             |>o==================/ /==o
     +Z Solar Array          |--.       .--|        -Z Solar Array
                             |ca|       |fr|
                             |--' +Xsc  '--|
                             |      ^      |
                             |      |      |
                             |      |      |
                        +Zsc .______|______.
                            <-------x\  `. ME
                              /______+Ysc_\
                           HGA    `.|.'                 +Ysc is into the
                                                         page.

      ``nom'' corresponds to ``NOMAD'';
      ``acs'' corresponds to ``ACS'';
      ``ca'' corresponds to ``CaSSIS''; and
      ``fr'' corresponds to ``FREND''.


   +Z S/C side view:
   -----------------

                         ._________________.
                         |                 |
                        ============o=     |
                         | SA+Z          +Zsc_..--,
                         |            <-----o-..__|
                         |          +Xsc   ||     ME
                         ._________________.|
                                         o-o|/|
                                          \|V +Ysc
                                           o  | :
                                            \ |/
                                             \|
                                                 HGA


   Since the S/C bus attitude with respect to an inertial frame is provided
   by a C-kernel (see [3] for more information), this frame is defined as
   a CK-based frame.

   These sets of keywords define the TGO_SPACECRAFT frame:

   \begindata

      FRAME_TGO_SPACECRAFT             = -143000
      FRAME_-143000_NAME               = 'TGO_SPACECRAFT'
      FRAME_-143000_CLASS              =  3
      FRAME_-143000_CLASS_ID           = -143000
      FRAME_-143000_CENTER             = -143
      CK_-143000_SCLK                  = -143
      CK_-143000_SPK                   = -143
      OBJECT_-143_FRAME                = 'TGO_SPACECRAFT'

   \begintext


   An additional S/C bus reference frame is defined in order to accommodate
   the C-kernels that have been generated with a fictional SCLK kernel. These
   CK kernels contain predicted data and are used for long and mid term
   planning.

   The before-mentioned CKs are generated with a fictional SCLK kernel due to
   the fact that successive updates of the real SCLK kernel will lead to
   erroneous results for the predicted data provided by those kernels after
   the last Time Correlation Packet that the real SCLK contains. The
   alternative of re-generating the planning CKs with the latest SCLK kernel
   is not considered.

   In order to be able to use the long and mid term planning CKs with the
   measured and short term planning CKs the planning CKs are generated with the
   fictional SCLK and are defined relative to the TGO spacecraft planning
   reference frame -- TGO_SPACECRAFT_PLAN --. Those planning CKs are then
   appended with a CK segment generated with the real SCLK that maps the
   TGO_SPACECRAFT_PLAN to the TGO_SPACECRAFT reference frame thus allowing
   to use both planning and measured CK files together with correct results.

   Note that when new SCLK are available the segment boundaries of the
   planning CKs will be affected. Due to this reason, the mapping segments
   boundaries are adjusted inwards by a minute on each side to get a better
   chance of them always being within the original CK segment boundaries.

   The TGO_SPACECRAFT_PLAN frame is defined as a CK-based frame. These sets of
   keywords define the TGO_SPACECRAFT_PLAN frame.

   \begindata

      FRAME_TGO_SPACECRAFT_PLAN        = -143002
      FRAME_-143002_NAME               = 'TGO_SPACECRAFT_PLAN'
      FRAME_-143002_CLASS              =  3
      FRAME_-143002_CLASS_ID           = -143002
      FRAME_-143002_CENTER             = -143
      CK_-143002_SCLK                  = -143999
      CK_-143002_SPK                   = -143

   \begintext


TGO Solar Array Frames
--------------------------------------------------------------------------

   TGO solar arrays are articulated (having one degree of freedom),
   therefore the Solar Array frames, TGO_SA+Z and TGO_SA-Z, are
   defined as CK frames with their orientation given relative to
   TGO_SA+Z_ZERO and TGO_SA-Z_ZERO respectively.

   TGO_SA+Z_ZERO and TGO_SA-Z_ZERO are two ``fixed-offset'' frames,
   defined with respect to TGO_SPACECRAFT, as follows:

      -  +Z is parallel to the longest side of the array, positively
         oriented from the yoke to the end of the wing;

      -  +X is antiparallel to the spacecraft bus +X axis, pointing
         on the opposite direction to the EDM attachment side;

      -  +Y completes the right-handed frame.

      -  the origin of the frame is located at the yoke geometric
         center.


   Both Solar Array frames (TGO_SA+Z and TGO_SA-Z) are defined as
   follows:

      -  +Z is parallel to the longest side of the array, positively oriented
         from the yoke to the end of the wing;

      -  +X is normal to the solar array plane, the solar cells facing +Z;

      -  +Y completes the right-handed frame;

      -  the origin of the frame is located at the yoke geometric center.


   The axis of rotation is parallel to the Z axis of the spacecraft and the
   solar array frames.

   This diagram illustrates the TGO_SA+Z_ZERO, TGO_SA+Z, TGO_SA-Z_ZERO
   and TGO_SA-Z frames:

   -X S/C side (Main Engine side) view:
   ------------------------------------

                                    ^
                                    | toward Mars
                                    |

                     +Ysa+z0  Science Deck
                     +Ysa+z  ._____________.
   .__  _______________.    ^|             |     .______________  ___.
   |  \ \               \   ||             |    /               \ \  |
   |  / /       +Zsa+z0  \  ||      _  +Xsa-z0 /     +Zsa-z0    / /  |
   |  \ \       +Zsa+z    `.||+Zsc/ _  +Xsa-z.'     +Zsa-z      \ \  |
   |  / /           <-------o| <----x)+Xsc |o-------->          / /  |
   |  \ \                 .' +Xsa+z0|_/    ||`.                 \ \  |
   |  / /                /   +Xsa+z |      ||  \                / /  |
   .__\ \_______________/    |      v +Ysc |v   \_______________\ \__.
       +Z Solar Array        .____________+Ysa-z0   -Z Solar Array
                        +Zsc     ._____.  +Ysa-z0
                               .'       `.
                              /           \
                             .   `.   .'   .          +Xsc is into the page;
                             |     `o'     |          +Xsa+z, +Xsa+z0,
                             .      |      .          +Xsa-z and +Xsa-z0 are
                              \     |     /            out of the page.
                               `.       .'
                            HGA  ` --- '


   These sets of keywords define solar array frames:

   \begindata

      FRAME_TGO_SA+Z_ZERO             = -143010
      FRAME_-143010_NAME              = 'TGO_SA+Z_ZERO'
      FRAME_-143010_CLASS             =  4
      FRAME_-143010_CLASS_ID          = -143010
      FRAME_-143010_CENTER            = -143
      TKFRAME_-143010_RELATIVE        = 'TGO_SPACECRAFT'
      TKFRAME_-143010_SPEC            = 'ANGLES'
      TKFRAME_-143010_UNITS           = 'DEGREES'
      TKFRAME_-143010_AXES            = (     3,      1,     2 )
      TKFRAME_-143010_ANGLES          = ( 180.0,    0.0,   0.0 )

      FRAME_TGO_SA+Z                  = -143011
      FRAME_-143011_NAME              = 'TGO_SA+Z'
      FRAME_-143011_CLASS             =  3
      FRAME_-143011_CLASS_ID          = -143011
      FRAME_-143011_CENTER            = -143
      CK_-143011_SCLK                 = -143
      CK_-143011_SPK                  = -143

      FRAME_TGO_SA-Z_ZERO             = -143012
      FRAME_-143012_NAME              = 'TGO_SA-Z_ZERO'
      FRAME_-143012_CLASS             =  4
      FRAME_-143012_CLASS_ID          = -143012
      FRAME_-143012_CENTER            = -143
      TKFRAME_-143012_RELATIVE        = 'TGO_SPACECRAFT'
      TKFRAME_-143012_SPEC            = 'ANGLES'
      TKFRAME_-143012_UNITS           = 'DEGREES'
      TKFRAME_-143012_AXES            = (      1,      2,     3 )
      TKFRAME_-143012_ANGLES          = (  180.0,    0.0,  180.0 )

      FRAME_TGO_SA-Z                  = -143013
      FRAME_-143013_NAME              = 'TGO_SA-Z'
      FRAME_-143013_CLASS             =  3
      FRAME_-143013_CLASS_ID          = -143013
      FRAME_-143013_CENTER            = -143
      CK_-143013_SCLK                 = -143
      CK_-143013_SPK                  = -143

   \begintext


TGO High Gain Antenna Frame
---------------------------

   The TGO High Gain Antenna is attached to the +Y panel of the S/C bus
   in the corner with the -X panel by a gimbal providing two degrees of
   freedom and it articulates during flight to track Earth. To incorporate
   rotations in the gimbal the HGA frame chain includes three frames:
   TGO_HGA_EL, TGO_HGA_AZ, and TGO_HGA.

   The first two frames are defined as CK-based frames and are
   co-aligned with the spacecraft frame in the zero gimbal position. In
   a non-zero position the TGO_HGA_EL is rotated from the spacecraft
   frame by an elevation angle about +Y and the TGO_TGO_AZ frame is
   rotated from the TGO_HGA_EL frame by an azimuth angle about +Z. These
   rotations are stored in separated segments in CK files.

   In [21] TGO_HGA frame is equivalent to the ``High Gain Antenna Functional
   Frame (HGAF)''.

   The TGO_HGA frame is defined as follows:

      -  +Z axis is in the antenna boresight direction;

      -  +X axis points from the gimbal toward the antenna dish
         symmetry axis;

      -  +Y axis completes the right hand frame;

      -  the origin of the frame is located at the phase center
         (theoretical and nominal location).

   The TGO_HGA frame is defined a fixed offset frame relative to the
   TGO_HGA_AZ frame and is rotated by -90 degrees about +X from it.

   This diagram illustrates the TGO_HGA frames in the zero gimbal
   position:

   -X S/C side (Main Engine side) view:
   ------------------------------------
                                    ^
                                    | toward Mars
                                    |

                               Science Deck
                             ._____________.
   .__  _______________.     |             |     .______________  ___.
   |  \ \               \    |             |    /               \ \  |
   |  / /                \   |     __      |   /                / /  |
   |  \ \                 +Zsc    / _ +Xsc | .'                 \ \  |
   |  / /                    <------x)     |o |                 / /  |
   |  \ \                 .' |    \_|_/    | `.                 \ \  |
   |  / /                /   |      |      |   \                / /  |
   .__\ \_______________+Zhga_az(*) |      |    \_______________\ \__.
       +Z Solar Array   +Zhga_el(*) v_+Ysc .     -Z Solar Array
                             <------x +Xhga_az(*) +Xhga_el(*)
                               .'   |   `.
                              /     | +Yhga_az(*)
                             .   `. v +Zhga_el(*)       +Xsc, +Xhga_az(*)
                             | Zhga o------->            and +Xhga_el(*) are
                             .      |      . +Yhga       into the page; +Zhga
                              \     |     /              is out of the page.
                               `.   |   .'
                            HGA  ` -V- '
                                      +Xhga


  * The TGO_HGA_AZ and TGO_HGA_EL frames are in zero gimbal position.


   This set of keywords defines the HGA frame as a CK frame:

   \begindata

      FRAME_TGO_HGA_EL                 = -143021
      FRAME_-143021_NAME               = 'TGO_HGA_EL'
      FRAME_-143021_CLASS              =  3
      FRAME_-143021_CLASS_ID           = -143021
      FRAME_-143021_CENTER             = -143
      CK_-143021_SCLK                  = -143
      CK_-143021_SPK                   = -143

      FRAME_TGO_HGA_AZ                 = -143022
      FRAME_-143022_NAME               = 'TGO_HGA_AZ'
      FRAME_-143022_CLASS              =  3
      FRAME_-143022_CLASS_ID           = -143022
      FRAME_-143022_CENTER             = -143
      CK_-143022_SCLK                  = -143
      CK_-143022_SPK                   = -143

      FRAME_TGO_HGA                    =  -143025
      FRAME_-143025_NAME               = 'TGO_HGA'
      FRAME_-143025_CLASS              =  4
      FRAME_-143025_CLASS_ID           =  -143025
      FRAME_-143025_CENTER             =  -143
      TKFRAME_-143025_RELATIVE         = 'TGO_HGA_AZ'
      TKFRAME_-143025_SPEC             = 'ANGLES'
      TKFRAME_-143025_UNITS            = 'DEGREES'
      TKFRAME_-143025_ANGLES           = (   0.000,  90.000,   0.000 )
      TKFRAME_-143025_AXES             = (   2,       1,       3     )

   \begintext


TGO Low Gain Antennae Frames
----------------------------

   The low gain antenna is an essential component of the S/C in an
   emergency case. During and after separation, full antenna coverage is
   necessary before a stable attitude is achieved, requiring three antennas
   with spherical coverage. In case of S/C survival mode, the LGAs provide
   the communication capability for recovery by ground operation.

   TGO has three Low Gain Antennae installed in the +Z, -Z and -X panels of
   the S/C bus -- TGO_LGA+Z, TGO_LGA-Z, TGO_LGA-X -- and are defined as
   ``fixed-offset'', defined with respect to the TGO_SPACECRAFT frame as
   follows (from [13]):

      -  +X axis is in the antenna boresight direction (nominally
         co-aligned to the spacecraft +Z, -Z and -X axis);

      -  +Y axis is in the direction of the spacecraft +X axis;

      -  +Z completes the right hand frame;

      -  the origin of the frame is defined as a reference mounting hole
         of each LGA.


   This diagram illustrates the TGO Low Gain Antennae frames:

   -X S/C side (Main Engine side) view:
   ------------------------------------

                                    ^
                                    | toward Mars
                                    |

                               Science Deck
                             ._____________.
   .__  _______________.     |             |     .______________  ___.
   |  \ \            +Zlga+z ^             ^ +Zlga-z            \ \  |
   |  / /                \   | +Zlga-x     |   /                / /  |
   |  \ \                 `. | ^  / _ \    | .'                 \ \  |
   |  / /     +Xlga+z <------o | | +Ylga-z x------> +Xlga-z     / /  |
   |  \ \               +Ylga+z|  \___/    | `.                 \ \  |
   |  / /                /   | o----> +Ylga-x  \                / /  |
   .__\ \_______________/    +Xlga-x       |    \_______________\ \__.
       +Z Solar Array        ._____________.      -Z Solar Array
                                 ._____.
                               .'       `.
        <--------x +Xsc       /           \          The TGO_SPACECRAFT frame
      +Zsc       |           .   `.   .'   .         origin is conveniently
                 |           |     `o'     |         moved.
                 |           .      |      .
                 v            \     |     /
                +Ysc           `.       .'           +Xsc, +Ylga-z is into
                            HGA  ` --- '              the page; +Xlga-x and
                                                     +Ylga+z are out of
                                                      the page.


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_LGA+Z                   =  -143031
      FRAME_-143031_NAME                = 'TGO_LGA+Z'
      FRAME_-143031_CLASS               =  4
      FRAME_-143031_CLASS_ID            =  -143031
      FRAME_-143031_CENTER              =  -143
      TKFRAME_-143031_RELATIVE          = 'TGO_SPACECRAFT'
      TKFRAME_-143031_SPEC              = 'ANGLES'
      TKFRAME_-143031_UNITS             = 'DEGREES'
      TKFRAME_-143031_AXES              = (     2,     1,       3 )
      TKFRAME_-143031_ANGLES            = (   0.0, -90.0,   -90.0 )

      FRAME_TGO_LGA-Z                   =  -143032
      FRAME_-143032_NAME                = 'TGO_LGA-Z'
      FRAME_-143032_CLASS               =  4
      FRAME_-143032_CLASS_ID            =  -143032
      FRAME_-143032_CENTER              =  -143
      TKFRAME_-143032_RELATIVE          = 'TGO_SPACECRAFT'
      TKFRAME_-143032_SPEC              = 'ANGLES'
      TKFRAME_-143032_UNITS             = 'DEGREES'
      TKFRAME_-143032_AXES              = (     2,     1,       3 )
      TKFRAME_-143032_ANGLES            = (   0.0, -90.0,   +90.0 )

      FRAME_TGO_LGA-X                   =  -143033
      FRAME_-143033_NAME                = 'TGO_LGA-X'
      FRAME_-143033_CLASS               =  4
      FRAME_-143033_CLASS_ID            =  -143033
      FRAME_-143033_CENTER              =  -143
      TKFRAME_-143033_RELATIVE          = 'TGO_SPACECRAFT'
      TKFRAME_-143033_SPEC              = 'ANGLES'
      TKFRAME_-143033_UNITS             = 'DEGREES'
      TKFRAME_-143033_AXES              = (     3,     1,       2 )
      TKFRAME_-143033_ANGLES            = (   0.0, +90.0,   180.0 )

   \begintext


TGO Star Trackers Frames
--------------------------------------------------------------------------

   There are two Star Trackers (STRs) mounted  mounted close to the PLM
   Instruments, on a dedicated panel attached to the 2 PLM Bench panels with
   their boresight canted 42 degrees from the spacecraft +X axis and with 5 deg
   above the XZ plane towards the +Y axis. The nominal and redundant STR are
   in mirror position with respect to the XY plane. This layout ensures a
   large angle between STR boresight and Sun whatever mission phase and that
   all S/C appendages (especially the Solar Arrays) are maintained outside of
   the Sun Exclusion Angle. The redundant STR is not co-aligned with main unit
   to minimize the risk of failure propagation in case of an unexpected
   external straylight perturbation. The angle between the two boresight is
   83 degrees. The on-board software manages fully autonomously the power-on
   of the STR until operational tracking mode is reached.

   The Star Tracker STR-1 and STR-2 frames -- TGO_STR-1 and TGO_STR-2 -- are
   defined as follows:

      -  +Z axis is anti-parallel to the direction of an incoming collimated
         light ray which is parallel to the optical axis;

      -  +X axis is in the plane formed by the +Z axis and the vector from
         the detector centre pointing along the positively counted detector
         rows, perpendicular to the +Z axis

      -  +Y axis completes the right hand frame;

      -  the origin of the frame is is defined as the intersection of the
         mounting interface plane and the +X Panel mounting hole.


   These diagrams illustrate the Star Trackers frames:


   -Y S/C side (Science Deck side) view:
   -------------------------------------

                    +Zstr-2       _____       +Zstr-1
                         ^       /     \  EDM  ^
                          \     |       |     /
                           \ ._____________. /
                            \|   |     |   |/
                   +Xstr-2  .o   |     |   x. +Ystr-1
                          .' |___'     '___| '.
   o==/ /===============.' o |             |>o='. ==============/ /==o
                       V     | -.       .--|     V
                  -Ystr-2    |  |       |  |     +Xstr-1
                             |--' +Xsc  '--|
                             |      ^      |
                             |      |      |            Ysc and +Ystr-1 are
                             |      |      |            into the page.
                        +Zsc .______|______.            Xstr-2 is out of the
                            <-------x   `. ME           page.
                              /______+Ysc \
                           HGA    `.|.'


   The rotation matrices from the Star Tracker frames to the S/C frame are
   the matrices that result of the following quaternions (from [20]):


     Quaternion = (  0.40858701, 0.01988630, -0.91164303, -0.03958660  )
         STR-1 -> SC          q_0         q_1          q_2          q_3


     Quaternion = (  0.65510702, 0.31692499, -0.25880700,  0.63514697  )
         STR-2 -> SC          q_0         q_1          q_2          q_3


   This is incorporated by the frame definitions below.

   \begindata

      FRAME_TGO_STR-1               = -143041
      FRAME_-143041_NAME            = 'TGO_STR-1'
      FRAME_-143041_CLASS           =  4
      FRAME_-143041_CLASS_ID        = -143041
      FRAME_-143041_CENTER          = -143
      TKFRAME_-143041_RELATIVE      = 'TGO_SPACECRAFT'
      TKFRAME_-143041_SPEC          = 'MATRIX'
      TKFRAME_-143041_MATRIX        = ( -0.66532202, -0.00390928, -0.74654627,
                                        -0.06860763,  0.99607487,  0.05592719,
                                         0.74339734,  0.08842836, -0.66297875 )

      FRAME_TGO_STR-2               = -143042
      FRAME_-143042_NAME            = 'TGO_STR-2'
      FRAME_-143042_CLASS           =  4
      FRAME_-143042_CLASS_ID        = -143042
      FRAME_-143042_CENTER          = -143
      TKFRAME_-143042_RELATIVE      = 'TGO_SPACECRAFT'
      TKFRAME_-143042_SPEC          = 'MATRIX'
      TKFRAME_-143042_MATRIX        = (  0.05921395, -0.99622389,  0.06349536,
                                         0.66813407, -0.00770686, -0.74400099,
                                         0.74168091,  0.08647865,  0.66515477 )

   \begintext


ACS Frames
------------------------------------------------------------------------

   This section of the file contains the definitions of the Atmospheric
   Chemistry Suite (ACS) instrument frames.


ACS Frame Tree
~~~~~~~~~~~~~~

   The diagram below shows the ACS frame hierarchy.


                               "J2000" INERTIAL
           +-----------------------------------------------------+
           |                          |                          |
           |<-pck                     |                          |<-pck
           |                          |                          |
           v                          |                          v
       "IAU_MARS"                     |                     "IAU_EARTH"
     MARS BODY-FIXED                  |<-ck               EARTH BODY-FIXED
     ---------------                  |                   ----------------
                                      v
                               "TGO_SPACECRAFT"
                    +-----------------------------------------+
                    |                    |                    |
                    |<-fixed             |<-fixed             |<-fixed
                    |                    |                    |
                    v                    v                    |
           "TGO_ACS_NIR_BASE"      "TGO_ACS_MIR"              |
         +---------------------+   -------------              |
         |                     |                              |
         |<-fixed              |<-fixed                       |
         |                     |                              |
         v                     v                              |
   "TGO_ACS_NIR_NAD"    "TGO_ACS_NIR_OCC"                     |
   -----------------    -----------------                     |
                                                              |
                                                              v
                                                 "TGO_ACS_TIRVIM_BASE"
   +-------------------------------------------------------------------------+
   |      |      |      |      |                    |         |              |
   |<-fxd |<-fxd |<-fxd |<-fxd |<-fxd        fixed->|  fixed->|       fixed->|
   |      |      |      |      |                    |         |              |
   |      |      |      |      v                    |         v              |
   |      |      |      | "TGO_ACS_TIRVIM_SCAN_BBY" |  "TGO_ACS_TIRVIM_SUN"  |
   |      |      |      | ------------------------- |  --------------------  |
   |      |      |      v                           |                        v
   |      |      |   "TGO_ACS_TIRVIM_SCAN_SPC"      |  "TGO_ACS_TIRVIM_SUN_BSR"
   |      |      |   -------------------------      |  ------------------------
   |      |      v                                  |
   |      |     "TGO_ACS_TIRVIM_SCAN_NAD"           |
   |      |     -------------------------           |
   |      v                                         v
   |     "TGO_ACS_TIRVIM_SCAN_OCC"      "TGO_ACS_TIRVIM_SCAN_ROT"
   |     -------------------------      -------------------------
   v                                                |
   "TGO_ACS_TIRVIM_SCAN_OCC_BS"                 ck->|
   ----------------------------                     |
                                                    v
                                          "TGO_ACS_TIRVIM_SCAN"
                                          ---------------------
                                                    |
                                             fixed->|
                                                    |
                                                    v
                                            "TGO_ACS_TIRVIM"
                                            ----------------


ACS TIRVIM Base Frame
~~~~~~~~~~~~~~~~~~~~~

   The ACS Thermal Infrared V-shape Interferometer Mounting Spectrometer
   (TIRVIM) is rigidly mounted on the spacecraft Science Deck. Therefore,
   the base frame associated with it -- the ACS TIRVIM Base frame,
   TGO_ACS_TIRVIM_BASE --  is specified as a fixed offset frame
   with its orientation given relative to the TGO_SPACECRAFT frame.

   The ACS TIRVIM Base frame are defined as follows (from [9]):

      -  +X axis is along the nominal spectrometer mirror rotation axis, and
         it is nominally co-aligned with the spacecraft +Z axis;

      -  +Z axis is co-aligned with the -Y spacecraft axis and it is along
         the spectrometer boresight in "nadir" position;

      -  +Y axis completes the right-handed frame;

      -  the origin of this frame is located at the intersection of the
         spectrometer scanning mirror rotation axis and mirror central axis.


   These diagrams illustrate the nominal TGO_ACS_TIRVIM_BASE frame with
   respect to the spacecraft frame.


   -X S/C side (Main Engine side) view:
   ------------------------------------


                                    ^
                                    | toward Mars
                                    |
                                    |    ^ +Zacs_tirvim_base
                                         |
                           Science deck  |
                             .___________|_.
   .__  _______________.     |   <-------o +Yacs_tirvim_base____  ___.
   |  \ \               \    | +Xacs_tirvim_base/               \ \  |
   |  / /                \   |     ___     |   /                / /  |
   |  \ \               +Zsc |    / _ +Xsc | .'                 \ \  |
   |  / /                  <--------x)     |o |                 / /  |
   |  \ \                 .' |    \_|_/    | `.                 \ \  |
   |  / /                /   |      |      |   \                / /  |
   .__\ \_______________/    |      |      |    \_______________\ \__.
       +Z Solar Array        ._____ v +Ysc .      -Z Solar Array
                                 ._____.
                               .'       `.
                              /           \
                             .   `.   .'   .          +Xsc is into the page;
                             |     `o'     |          +Yacs_tirvim_base is
                             .      |      .           out of the page.
                              \     |     /
                               `.       .'
                            HGA  ` --- '


   -Y S/C side (Science Deck side) view:
   -------------------------------------
                                  _____
                                 /     \  EDM
                                |       |
                             ._____________.
                             |+Xacs_tirvim_base
                             |    <------o | +Zacs_tirvim_base
                      +Zsc   |        '__|_|
   o==/ /==================o<|           | |>o==================/ /==o
     +Z Solar Array          |           | |        -Z Solar Array
                             |           v +Yacs_tirvim_base
                             |    +Xsc     |
                             |      ^      |
                             |      |      |
                             |      |      |
                        +Zsc .______|______.            +Ysc is into the
                            <-------x   `. ME            page;
                              /______+Ysc \             +Zacs_tirvim_base
                           HGA    `.|.'                  is out of the page.


   Nominally, a rotation of -90 degrees about +Y spacecraft axis and then
   a rotation of  90 degrees about the +X resulting axis are required to
   align the TGO_SPACECRAFT to the TGO_ACS_TIRVIM_BASE frame.


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_ACS_TIRVIM_BASE        =  -143105
      FRAME_-143105_NAME               = 'TGO_ACS_TIRVIM_BASE'
      FRAME_-143105_CLASS              =   4
      FRAME_-143105_CLASS_ID           =  -143105
      FRAME_-143105_CENTER             =  -143
      TKFRAME_-143105_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143105_SPEC             = 'ANGLES'
      TKFRAME_-143105_UNITS            = 'DEGREES'
      TKFRAME_-143105_AXES             = ( 3,     2,      1   )
      TKFRAME_-143105_ANGLES           = ( 0.0,  90.0,  -90.0 )

   \begintext


ACS TIRVIM Scanning Mirror frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The Thermal Infrared V-shape Interferometer Mounting Spectrometer
   (TIRVIM) has a single-axis scanning mirror that provide the possibility
   of observation in nadir and any position from it (360 degrees of positive
   rotation only) (see [11]).

   Since this scanning mirrors rotate with respect to the TIRVIM base,
   the TGO_ACS_TIRVIM_SCAN frame is defined as a CK frame with its orientation
   provided in a CK file relative to the TGO_ACS_TIRVIM_SCAN_ROT frames. The
   TGO_ACS_TIRVIM_SCAN_ROT frame is then defined to incorporate misalignments
   in the Scanner rotation axis.

   The ACS TIRVIM scanning mirror rotation frame and the ACS TIRVIM scanning
   mirror frames -- TGO_ACS_TIRVIM_SCAN_ROT and TGO_ACS_TIRVIM_SCAN -- are
   defined as (from [9]):

      -  +X axis is along the nominal spectrometer scanning mirror rotation
         axis, and it is nominally co-aligned with the spectrometer base +X
         axis;

      -  +Z axis is parallel to the scanning mirror boresight that defines
         the spectrometer boresight; in 'nadir' scanner position is co-aligned
         with the -Y spacecraft axis -- this is the 'fixed' position for
         the TGO_ACS_TIRVIM_SCAN_ROT frame --; in 'space' scanner position is
         co-aligned with the +X spacecraft axis;

      -  +Y axis completes the right-handed frame;

      -  the origin of this frame is located at the intersection of the
         spectrometer scanning mirror rotation axis and mirror central axis.


   For an arbitrary scanner angle, the scanning mirror frame base is rotated
   by this angle about the +X axis with respect to its rotation frame. The
   sense of rotation is:

      (1) nadir - (2) space - (3) black body - (4) occultation - (1) nadir

   These diagrams illustrate the TGO_ACS_TIRVIM frames for nominal scanner
   positions 'nadir' (~0.0 degrees), equivalent to TGO_ACS_TIRVIM_SCAN_ROT,
   solar 'occultation' (~67.07 deg from the spacecraft -Y axis to the -X in
   the XY plane) and 'space' (~90.00 deg from the spacecraft -Y axis to the
   +X in the XY plane). All diagrams are +Z S/C side view:


    Scanner in 'nadir' position        Scanner in 'occ' position
    ---------------------------        -------------------------

      (1)         +Zbase                 (4)    +Zbase
                ^ +Zscan                           ^
                |                                  |         +Zscan
                |       Science                    |_~67deg .^
                | ACS     Deck                     |  \  .'    Science Deck
          ._____|___________.                ._ACS_| .'________.
          |     o------->   |                |     o------->   |
         ============o= +Ybase             ========\===o= +Ybase
        SA+Z            +Yscan_..--,        SA+Z     \         |__..--,
          |            <-----o-..__|         |       \    <-----o-..__|
          |          +Xsc   ||     ME        |        v  +Xsc  ||     ME
          ._________________.|               ._________+Yscan_.||
                          o-o|/|                             o-o|/|
                           \|V +Ysc                           \|V +Ysc
                            o  | :                             o  | :
                             \ |/                               \ |/
                              \|                                 \|
                                  HGA                               HGA


     Scanner in 'space' position        Scanner in 'black body' position
     ---------------------------        --------------------------------

       (2)        +Zbase                   (3)       +Zbase
                ^ +Yscan                           ^
                |                                  |
                |       Science                    |       Science
        ~90deg.-| ACS     Deck            ~180deg.-| ACS     Deck
    +Zscan _/___|___________.                ._/___|___________.
        <-------o------->   |              <-------o------->   |
         ============o= +Ybase          +Yscan=\===|====o= +Ybase
        SA+Z                |__..--,        SA+Z'._|           |__..--,
          |            <-----o-..__|         |     |      <-----o-..__|
          |          +Xsc   ||     ME        |     V     +Xsc   ||     ME
          ._________________.|               .___+Zscan_________.|
                          o-o|/|                              o-o|/|
                           \|V +Ysc                            \|V +Ysc
                            o  | :                              o  | :
                             \ |/                                \ |/
                              \|                                  \|
                                  HGA                                 HGA


                                              +Zsc, +Xbase, and +Xscan are
                                               out of the page.


   Nominally the TGO_ACS_TIRVIM_SCAN_ROT frame is equivalent to the
   TGO_ACS_TIRVIM_BASE frame.

   These sets of keywords define the TIRVIM scanning mirror frame as a
   CK based frame and the TIRVIM scanning mirror rotation frame as a
   fixed-offset frame. Since the SPICE frames subsystem calls for specifying
   the reverse transformation--going from the instrument or structure frame to
   the base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed compared
   to the above text, and the signs associated with the rotation angles
   assigned to the TKFRAME_*_ANGLES keyword are the opposite from what is
   written in the above text.:

   \begindata

      FRAME_TGO_ACS_TIRVIM_SCAN_ROT    =  -143106
      FRAME_-143106_NAME               = 'TGO_ACS_TIRVIM_SCAN_ROT'
      FRAME_-143106_CLASS              =   4
      FRAME_-143106_CLASS_ID           =  -143106
      FRAME_-143106_CENTER             =  -143
      TKFRAME_-143106_RELATIVE         = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143106_SPEC             = 'ANGLES'
      TKFRAME_-143106_UNITS            = 'DEGREES'
      TKFRAME_-143106_AXES             = ( 3,     2,      1   )
      TKFRAME_-143106_ANGLES           = ( 0.0,   0.0,    0.0 )

      FRAME_TGO_ACS_TIRVIM_SCAN        = -143107
      FRAME_-143107_NAME               = 'TGO_ACS_TIRVIM_SCAN'
      FRAME_-143107_CLASS              =  3
      FRAME_-143107_CLASS_ID           = -143107
      FRAME_-143107_CENTER             = -143
      CK_-143107_SCLK                  = -143
      CK_-143107_SPK                   = -143

   \begintext


ACS TIRVIM Detector Frames
~~~~~~~~~~~~~~~~~~~~~~~~~~

   Since the TIRVIM detector receives radiation through the scanner
   and has a single pixel, its frame, TGO_ACS_TIRVIM is defined to be
   nominally co-aligned with the TIRVIM scanner frame TGO_ACS_TIRVIM_SCAN.
   This frame is introduced to allow incorporating into the TIRVIM frame
   chain any misalignment between the scanner boresight direction and the
   detector view directions.

   The following in-flight calibrated rotation axis misalignment, provided as
   the boresight in 'nadir' position was provided by A. Trokhimovskiy on 10th
   May, 2017 [19] please note that the scanner channel in 'nadir' position was
   not measured, but inferred from an offset in the 'solar direction':

      ACS_TIRVIM Boresight: ( 0.0, -0.99905, -0.04362 )

   The boresights is defined relative to the TGO_SPACECRAFT frame. Given
   the boresight the rotation from the TGO_ACS_TIRVIM_SCAN frame to the
   TGO_ACS_TIRVIM frames determined from the in-flight calibration data
   can be represented by the following rotation angles in degrees:

       nad
      M    = |0.0|  * |-2.5000306232819502|  * |3.6615346000217E-15|
       base       Z                        Y                        X


   This set of keywords define the TIRVIM frame as a fixed-offset frame. Since
   the SPICE frames subsystem calls for specifying the reverse transformation
   --going from the instrument or structure frame to the base frame-- as
   compared to the description given above, the order of rotations assigned to
   the TKFRAME_*_AXES keyword is also reversed compared to the above text, and
   the signs associated with the rotation angles assigned to the
   TKFRAME_*_ANGLES keyword are the opposite from what is written in the above
   text.:

   \begindata

      FRAME_TGO_ACS_TIRVIM             =  -143110
      FRAME_-143110_NAME               = 'TGO_ACS_TIRVIM'
      FRAME_-143110_CLASS              =   4
      FRAME_-143110_CLASS_ID           =  -143110
      FRAME_-143110_CENTER             =  -143
      TKFRAME_-143110_RELATIVE         = 'TGO_ACS_TIRVIM_SCAN'
      TKFRAME_-143110_SPEC             = 'ANGLES'
      TKFRAME_-143110_UNITS            = 'DEGREES'
      TKFRAME_-143110_AXES             = ( 3,     2,      1   )
      TKFRAME_-143110_ANGLES           = (
                            0.0,  2.5000306232819502, 0.000000000000003661535
                                         )

   \begintext


ACS TIRVIM fixed scanner position frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The ACS TIRVIM scanning mirror can be used in several positions for
   external observations, being the most common 'nadir' and
   cold 'space.' In addition, and through the usage of an off-pointed
   periscope and an off-axis parabolic mirror, a solar 'occultation'
   observation can be performed in parallel to the use of the scanning
   mirror. For the solar 'occultation' position a boresight for the position
   of the best spectral resolution within the FoV is also provided. This
   boresight will be used for pointings driven by TIRVIM using the scanner
   channel. The 'black body' position of the scanning mirror is also
   considered.

   Therefore, for the ACS TIRVIM a set of 'fixed-mirror-position'
   frames -- 'nadir', 'solar occultation', 'solar occultation best spectral
   resolution', 'space' and 'black body' -- are defined as fixed-offset
   frames to allow computing these orientations without needing to use a CK:

       Frame Name                  Fixed-mirror-position
      ---------------------------- ------------------------------------------
       TGO_ACS_TIRVIM_SCAN_BBY     Black Body
       TGO_ACS_TIRVIM_SCAN_SPC     Cold Space
       TGO_ACS_TIRVIM_SCAN_NAD     Nadir
       TGO_ACS_TIRVIM_SCAN_OCC     Solar Occultation
       TGO_ACS_TIRVIM_SCAN_OCC_BS  Solar Occultation Best Spectral Resolution


   Each of these 'fixed-position' frames are defined as a fixed
   offset frame with respect to the corresponding base frame for each of
   the spectrometers as follows (from [9]):

      -  +X axis is along the nominal spectrometer mirror rotation
         axis, and it is nominally co-aligned with the spectrometer
         base +X axis;

      -  +Z axis is parallel to the scanning mirror boresight that defines
         the spectrometer boresight at a particular angle;

      -  +Y axis completes the right-handed frame;

      -  the origin of this frame is located at the intersection of the
         spectrometer scanning mirror rotation axis and mirror central axis.


   These diagrams illustrate fixed mirror pointing directions co-aligned
   with the +Z axis of the corresponding 'fixed-mirror-position' frame
   -- the 'solar occultation best spectral resolution' and 'black body'
   positions are not represented --:

   +Z S/C side view
   ----------------

                  +Z*base                        +Z*base
                ^ +Z*nad                           ^
                |                                  |         +Z*occ
                |       Science                    |_~67deg .^
                | ACS     Deck                     |  \  .'    Science Deck
          ._____|___________.                ._ACS_| .'________.
          |     o------->   |                |     o------->   |
         ============o= +Y*base             ========\===o= +Y*base
         SA+Z             +Y*nad.--,        SA+Z     \         |__..--,
          |            <-----o-..__|         |        \   <-----o-..__|
          |          +Xsc   ||     ME        |         v +Xsc  ||     ME
          ._________________.|               .______+Y*occ_____.|
                          o-o|/|                             o-o|/|
                           \|V +Ysc                           \|V +Ysc
                            o  | :                             o  | :
                             \ |/                               \ |/
                              \|                                 \|
                                  HGA                               HGA


                +Zbase
                +Zbase                             +Zbase
                ^ +Y*spc                           ^ +Y*bby
                |                                  |
                |       Science                    |       Science
        ~90deg.-| ACS     Deck            ~180deg.-| ACS     Deck
    +Z*spc _/___|___________.                ._/___|___________.
        <-------o------->   |                ||    o------->   |
         ============o= +Ybase              ===\===|====o= +Ybase
         SA+Z               |__..--,        SA+Z'._|    +Xsc   |__..--,
          |            <-----o-..__|         |     |      <-----o-..__|
          |          +Xsc   ||     ME        |     v +Z*bby    ||     ME
          ._________________.|               ._________________.|
                          o-o|/|                             o-o|/|
                           \|V +Ysc                           \|V +Ysc
                            o  | :                             o  | :
                             \ |/                               \ |/
                              \|                                 \|
                                  HGA                               HGA


                                              +Zsc, +Xbase, +X*nad, +X*occ,
                                              +X*bby and +X*spc are out of
                                              the page;


      ``*base'' corresponds to ``acs_tirvim_base'';
      ``*nad'' corresponds to ``acs_tirvim_scan_nad'';
      ``*occ'' corresponds to ``acs_tirvim_scan_occ'';
      ``*spc'' corresponds to ``acs_tirvim_scan_spc'' and
      ``*bby'' corresponds to ``acs_tirvim_scan_bby''.


   These 'fixed-position' frames are nominally rotated about the
   +X axis of the corresponding spectrometer base frames by the following
   angles:

      Frame name                    Rotation Angle, deg
      ---------------------------  -------------------
      TGO_ACS_TIRVIM_SCAN_BBY        180.00
      TGO_ACS_TIRVIM_SCAN_SPC        +90.00
      TGO_ACS_TIRVIM_SCAN_NAD          0.00
      TGO_ACS_TIRVIM_SCAN_OCC        -67.07
      TGO_ACS_TIRVIM_SCAN_OCC_BS     -67.07


   The following in-flight calibrated rotation axis misalignment, provided as
   the boresight in 'nadir' position was provided by A. Trokhimovskiy on 25th
   and 27th April and on the 10th May, 2017 [19]:

      ACS_TIRVIM_SCAN_NAD Boresight:     (  0.00000, -0.99905, -0.04362 )

      ACS_TIRVIM_SCAN_OCC Boresight:     ( -0.89699, -0.44081, -0.04362 )

      ACS_TIRVIM_SCAN_OCC_BS Boresight:  ( -0.90139, -0.43102, -0.04028 )

   The boresights is defined relative to the TGO_SPACECRAFT frame. Given
   the boresight the rotation from the TGO_ACS_TIRVIM_BASE frame to the
   TGO_ACS_TIRVIM_SCAN_ROT frames determined from the in-flight calibration
   data can be represented by the following rotation angles in degrees:

       nad
      M    = |0.0|  * |-2.5000306232819502|  * |3.6615346000217E-15|
       base       Z                        Y                        X

       occ
      M    = |0.0|  * |-2.4990260652816447|  * |-63.829000275807154|
       base       Z                        Y                        X

       bsr
      M    = |0.0|  * |-2.3086086582635210|  * |-64.444150970533260|
       base       Z                        Y                        X


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_ACS_TIRVIM_SCAN_BBY    =  -143131
      FRAME_-143131_NAME               = 'TGO_ACS_TIRVIM_SCAN_BBY'
      FRAME_-143131_CLASS              =   4
      FRAME_-143131_CLASS_ID           =  -143131
      FRAME_-143131_CENTER             =  -143
      TKFRAME_-143131_RELATIVE         = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143131_SPEC             = 'ANGLES'
      TKFRAME_-143131_UNITS            = 'DEGREES'
      TKFRAME_-143131_AXES             = ( 3,     2,      1    )
      TKFRAME_-143131_ANGLES           = ( 0.0,   0.0,  180.00 )

      FRAME_TGO_ACS_TIRVIM_SCAN_SPC    =  -143132
      FRAME_-143132_NAME               = 'TGO_ACS_TIRVIM_SCAN_SPC'
      FRAME_-143132_CLASS              =   4
      FRAME_-143132_CLASS_ID           =  -143132
      FRAME_-143132_CENTER             =  -143
      TKFRAME_-143132_RELATIVE         = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143132_SPEC             = 'ANGLES'
      TKFRAME_-143132_UNITS            = 'DEGREES'
      TKFRAME_-143132_AXES             = ( 3,     2,      1    )
      TKFRAME_-143132_ANGLES           = ( 0.0,   0.0,  -90.00 )

      FRAME_TGO_ACS_TIRVIM_SCAN_NAD    =  -143133
      FRAME_-143133_NAME               = 'TGO_ACS_TIRVIM_SCAN_NAD'
      FRAME_-143133_CLASS              =   4
      FRAME_-143133_CLASS_ID           =  -143133
      FRAME_-143133_CENTER             =  -143
      TKFRAME_-143133_RELATIVE         = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143133_SPEC             = 'ANGLES'
      TKFRAME_-143133_UNITS            = 'DEGREES'
      TKFRAME_-143133_AXES             = ( 3,     2,      1   )
      TKFRAME_-143133_ANGLES           = (
                            0.0,  2.5000306232819502, 0.000000000000003661535
                                         )

      FRAME_TGO_ACS_TIRVIM_SCAN_OCC    =  -143134
      FRAME_-143134_NAME               = 'TGO_ACS_TIRVIM_SCAN_OCC'
      FRAME_-143134_CLASS              =   4
      FRAME_-143134_CLASS_ID           =  -143134
      FRAME_-143134_CENTER             =  -143
      TKFRAME_-143134_RELATIVE         = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143134_SPEC             = 'ANGLES'
      TKFRAME_-143134_UNITS            = 'DEGREES'
      TKFRAME_-143134_AXES             = ( 3,     2,      1    )
      TKFRAME_-143134_ANGLES           = (
                             0.0,  2.4990260652816447,  +63.829000275807154
                                          )

      FRAME_TGO_ACS_TIRVIM_SCAN_OCC_BS  =  -143135
      FRAME_-143135_NAME                = 'TGO_ACS_TIRVIM_SCAN_OCC_BS'
      FRAME_-143135_CLASS               =   4
      FRAME_-143135_CLASS_ID            =  -143135
      FRAME_-143135_CENTER              =  -143
      TKFRAME_-143135_RELATIVE          = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143135_SPEC              = 'ANGLES'
      TKFRAME_-143135_UNITS             = 'DEGREES'
      TKFRAME_-143135_AXES              = ( 3,     2,      1    )
      TKFRAME_-143135_ANGLES            = (
                              0.0,   2.3086086582635210,  +64.444150970533260
                                          )

   \begintext


ACS TIRVIM Sun Channel frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The ACS TIRVIM instrument has a Sun Channel for which two reference
   frames are defined: the ACS TIRVIM Sun Channel frame that is defined by the
   center of the FoV and the ACS TIRVIM Sun Channel Best Spectral Resolution
   -- TGO_ACS_TIRVIM_SUN_BSR and TGO_ACS_TIRVIM_SUN -- which is defined by a
   boresight to be used for pointings driven by TIRVIM (using the Sun channel)
   that corresponds to the pixel of best spectral resolution of the slit. Note
   that the sensitivity is not uniform within the FOV and peaks close to the
   MIR slit.

   Each of these frames are defined as a fixed offset frame with respect to
   the corresponding base frame for each of the spectrometers as follows (from
   [9]):

      -  +X axis is nominally co-aligned with the spectrometer
         base +X axis;

      -  +Z axis is parallel to the Sun Channel boresight (the
         nominal and the BSR one respectively);

      -  +Y axis completes the right-handed frame;

      -  the origin of this frame is located at the intersection of the
         spectrometer scanning mirror rotation axis and mirror central axis.


   These diagram illustrates fixed mirror pointing directions co-aligned
   with the +Z axis of the corresponding 'fixed-mirror-position' frame:

   +Z S/C side view
   ----------------

          +Z*base
            ^
            |         +Zsun
            |_~67deg .^
            |  \  .'    Science Deck
      ._ACS_| .'________.
      |     o------->   |
     ========\===o= +Y*base
     SA+Z     \         |__..--,
      |        \   <-----o-..__|
      |         v +Xsc   ||     ME
      ._______+Ysun______.|
                       o-o|/|
                        \|V +Ysc
                         o  | :         +Zsc and +Xsun
                          \ |/           and +Xspc are out of the page.
                           \|
                               HGA


   Nominally, a rotation of 90 degrees about +X spacecraft axis and then
   a rotation of -67.07 degrees about the +Y resulting axis are required to
   align the TGO_SPACECRAFT to the TGO_ACS_MIR frames.

   The following in-flight calibrated misalignment boresight was provided
   by A. Trokhimovskiy on April 25 and 27, 2017 [19]:

      ACS_TIRVIM_SUN Boresight:      ( -0.92070, -0.39000, -0.01420 )

      ACS_TIRVIM_SUN_BSR Boresight:  ( -0.92169, -0.38739, -0.02042 )


   These boresights are defined relative to the TGO_SPACECRAFT frame. Given
   these boresights the rotation from the TGO_ACS_TIRVIM_BASE frame to the
   TGO_ACS_TIRVIM_SUN and TGO_ACS_TIRVIM_SUN_BSR frames determined from the
   in-flight calibration data can be represented by the following rotation
   angles in degrees:

       sun
      M    = |0.0|  * |-0.8136314295043187|  * |-67.04293381739603|
       base       Z                        Y                       X

       bsr
      M    = |0.0|  * |-1.1700608817724396|  * |-67.20278942918976|
       base       Z                        Y                       X


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_ACS_TIRVIM_SUN         =  -143140
      FRAME_-143140_NAME               = 'TGO_ACS_TIRVIM_SUN'
      FRAME_-143140_CLASS              =   4
      FRAME_-143140_CLASS_ID           =  -143140
      FRAME_-143140_CENTER             =  -143
      TKFRAME_-143140_RELATIVE         = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143140_SPEC             = 'ANGLES'
      TKFRAME_-143140_UNITS            = 'DEGREES'
      TKFRAME_-143140_AXES             = ( 3,     2,      1   )
      TKFRAME_-143140_ANGLES           = (
                   0.0, +0.8136314295043187,  +67.04293381739603
                                         )

      FRAME_TGO_ACS_TIRVIM_SUN_BSR     =  -143141
      FRAME_-143141_NAME               = 'TGO_ACS_TIRVIM_SUN_BSR'
      FRAME_-143141_CLASS              =   4
      FRAME_-143141_CLASS_ID           =  -143141
      FRAME_-143141_CENTER             =  -143
      TKFRAME_-143141_RELATIVE         = 'TGO_ACS_TIRVIM_BASE'
      TKFRAME_-143141_SPEC             = 'ANGLES'
      TKFRAME_-143141_UNITS            = 'DEGREES'
      TKFRAME_-143141_AXES             = ( 3,     2,      1    )
      TKFRAME_-143141_ANGLES           = (
                   0.0, +1.1700608817724396,  +67.20278942918976
                                         )

   \begintext


ACS Near Infrared (NIR) Base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The ACS Near Infrared (NIR) spectrometer is rigidly mounted on the
   spacecraft Science Deck. Therefore, the base frame associated with
   it -- the ACS NIR Base frame, TGO_ACS_NIR_BASE -- is specified as
   a fixed offset frame with its orientation given relative to the
   TGO_SPACECRAFT frame.

   The ACS NIR Base frame is defined as follows (from [9]):

      -  +X axis is along the nominal spectrometer boresight, and
         it is nominally co-aligned with the spacecraft +Z axis;

      -  +Z axis is co-aligned with the -Y spacecraft axis and it is along
         the spectrometer boresight in "nadir" position;

      -  +Y axis completes the right-handed frame;

      -  the origin of this frame is located at geometrical centre of the
         first folding mirror at the entry optics of the spectrometer.


   These diagrams illustrate the nominal TGO_ACS_NIR_BASE frame with respect
  to the spacecraft frame.


   -X S/C side (Main Engine side) view:
   ------------------------------------
                                    ^
                                    | toward Mars
                                    |
                                    |    ^ +Zacs_nir_base
                                         |
                           Science deck  |
                             .___________|_.
   .__  _______________.     |   <-------o +Yacs_nir_base_______  ___.
   |  \ \               \    | +Xacs_nir_base   /               \ \  |
   |  / /                \   |     ___     |   /                / /  |
   |  \ \               +Zsc |    / _ +Xsc | .'                 \ \  |
   |  / /                  <--------x)     |o |                 / /  |
   |  \ \                 .' |    \_|_/    | `.                 \ \  |
   |  / /                /   |      |      |   \                / /  |
   .__\ \_______________/    |      |      |    \_______________\ \__.
       +Z Solar Array        ._____ v +Ysc .      -Z Solar Array
                                 ._____.
                               .'       `.
                              /           \
                             .   `.   .'   .        +Xsc is into the page;
                             .      |      .        +Yacs_nir_base is out
                              \     |     /          of the page.
                               `.       .'
                            HGA  ` --- '


   -Y S/C side (Science Deck side) view:
   -------------------------------------
                                  _____
                                 /     \  EDM
                                |       |
                             ._____________.
                             |+Xacs_nir_base
                             |    <------o | +Zacs_nir_base
                      +Zsc   |        '__|_|
   o==/ /==================o<|           | |>o==================/ /==o
     +Z Solar Array          |           | |        -Z Solar Array
                             |           v +Yacs_nir_base
                             |    +Xsc     |
                             |      ^      |
                             |      |      |
                             |      |      |
                        +Zsc .______|______.            +Ysc is into the
                            <-------x   `. ME            page;
                              /______+Ysc \             +Zacs_nir_base
                           HGA    `.|.'                  is out of the page.



   Nominally, a rotation of -90 degrees about +Y spacecraft axis and then
   a rotation of  90 degrees about the +X resulting axis are required to
   align the TGO_SPACECRAFT to the TGO_ACS_NIR_BASE frame.


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_ACS_NIR_BASE           =  -143100
      FRAME_-143100_NAME               = 'TGO_ACS_NIR_BASE'
      FRAME_-143100_CLASS              =   4
      FRAME_-143100_CLASS_ID           =  -143100
      FRAME_-143100_CENTER             =  -143
      TKFRAME_-143100_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143100_SPEC             = 'ANGLES'
      TKFRAME_-143100_UNITS            = 'DEGREES'
      TKFRAME_-143100_AXES             = ( 3,     2,      1   )
      TKFRAME_-143100_ANGLES           = ( 0.0,  90.0,  -90.0 )

   \begintext


ACS NIR nadir and occultation position frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The ACS NIR spectrometer is capable of performing parallel nadir and
   limb/solar occultation measurements (see [11]) using the two periscope
   apertures (one of them "looking" nadir) and an off-axis parabolic
   mirror that defines the NIR occultation boresight to be at 67.07
   degrees from -Y spacecraft axis towards the -X spacecraft axis in the
   XY plane, therefore two frames -- TGO_ACS_NIR_NAD and TGO_ACS_NIR_OCC
   -- are defined as fixed-offset frames to allow computing the
   orientation of the ACS NIR field-of-view in both cases.

   Each of these 'fixed-position' frames is defined as a fixed
   offset frame with respect to the corresponding base frame for each of
   the spectrometers as follows (from [9]):

      -  +X axis is along the nominal spectrometer boresight, and it is
         nominally co-aligned with the spectrometer base +X axis;

      -  +Z axis is parallel to the spectrometer detector array's lines;

      -  +Y axis completes the right-handed frame;

      -  the origin of this frame is located at geometrical centre of the
         first folding mirror at the entry optics of the spectrometer.


   These diagram illustrates fixed mirror pointing directions co-aligned
   with the +Z axis of the corresponding 'fixed-mirror-position' frame:

   +Z S/C side view
   ----------------

                  +Z*base                        +Z*base
                ^ +Z*nad                           ^
                |                                  |         +Z*occ
                |       Science                    |_~67deg .^
                | ACS     Deck                     |  \  .'    Science Deck
          ._____|___________.                ._ACS_| .'________.
          |     o------->   |                |     o------->   |
         ============o= +Y*base             ========\===o= +Y*base
         SA+Z             +Y*nad.--,        SA+Z     \         |__..--,
          |            <-----o-..__|         |        \   <-----o-..__|
          |          +Xsc   ||     ME        |         v +Xsc  ||     ME
          ._________________.|               .______+Y*occ_____.|
                          o-o|/|                             o-o|/|
                           \|V +Ysc                           \|V +Ysc
                            o  | :                             o  | :
                             \ |/                               \ |/
                              \|                                 \|
                                  HGA                               HGA


                +Zbase
                ^ +Yspc                        +Zsc, +X*base, +X*nad, +X*occ
                |                               and +Xspc are out of the page.
                |       Science
         90deg.-| ACS     Deck
    +Zspc  _/___|___________.
        <-------o------->   |
         ============o= +Ybase
         SA+Z               |__..--,
          |            <-----o-..__|
          |          +Xsc   ||     ME
          ._________________.|
                          o-o|/|
                           \|V +Ysc
                            o  | :
                             \ |/
                              \|
                                  HGA


      ``*base'' corresponds to ``acs_nir_base'';
      ``*nad'' corresponds to ``acs_nir_nad''; and
      ``*occ'' corresponds to ``acs_nir_occ''.


   These 'fixed-position' frames are nominally rotated about the
   +X axis of the corresponding spectrometer base frames by the following
   angles:

      Frame name              Rotation Angle, deg
      ----------------------  -------------------
      TGO_ACS_NIR_NAD           0.00
      TGO_ACS_NIR_OCC         -67.07


   The following in-flight calibrated misalignment boresights were provided
   by A. Trokhimovskiy on June 13, 2016 [17]. The boresight vector is provided
   as is and the the cross vector, that completes the reference frame is
   defined as the composition of the boresight and the reference vector: :

      ACS_NIR_OCC Boresight:         ( -0.9231, -0.3845, -0.0069 )

      ACS_NIR_OCC Reference Vector:  ( -0.9220, -0.3860,  0.0025 )


   This boresight is relative to the TGO_SPACECRAFT frame. Given this
   boresight the rotation from the TGO_ACS_NIR_BASE frame to the
   TGO_ACS_NIR_OCC frame determined from the in-flight calibration
   data can be represented by the following rotation angles in degrees:

      occ
     M  = |-10.888905099180327| * |-0.3953437251552117| * |-67.38674625597875|
      base                    Z                       Y                      X


   The TGO_ACS_NIR_NAD misalignment will updated during the Science phase.
   For the time being the available measurements are not sufficient to
   determine the boresight reliably according to Alexander Trokhimovskiy
   on 29th June 2017.

   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_ACS_NIR_NAD            =  -143111
      FRAME_-143111_NAME               = 'TGO_ACS_NIR_NAD'
      FRAME_-143111_CLASS              =   4
      FRAME_-143111_CLASS_ID           =  -143111
      FRAME_-143111_CENTER             =  -143
      TKFRAME_-143111_RELATIVE         = 'TGO_ACS_NIR_BASE'
      TKFRAME_-143111_SPEC             = 'ANGLES'
      TKFRAME_-143111_UNITS            = 'DEGREES'
      TKFRAME_-143111_AXES             = ( 3,     2,      1   )
      TKFRAME_-143111_ANGLES           = ( 0.0,   0.0,    0.0 )

      FRAME_TGO_ACS_NIR_OCC            =  -143112
      FRAME_-143112_NAME               = 'TGO_ACS_NIR_OCC'
      FRAME_-143112_CLASS              =   4
      FRAME_-143112_CLASS_ID           =  -143112
      FRAME_-143112_CENTER             =  -143
      TKFRAME_-143112_RELATIVE         = 'TGO_ACS_NIR_BASE'
      TKFRAME_-143112_SPEC             = 'ANGLES'
      TKFRAME_-143112_UNITS            = 'DEGREES'
      TKFRAME_-143112_AXES             = ( 1,     2,      3      )
      TKFRAME_-143112_ANGLES           = ( +67.38674625597875,
                                           +0.3953437251552117,
                                           +10.888905099180327   )

   \begintext


ACS High Resolution Middle Infrared Spectrometer (MIR) frame:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The ACS High Resolution Middle Infrared Spectrometer (MIR) is rigidly
   mounted on the spacecraft Science Deck. Therefore, the frame associated
   with it -- the ACS MIR frame, TGO_ACS_MIR -- is specified as a fixed
   offset frame with its orientation given relative to the TGO_SPACECRAFT
   frame.

   The ACS MIR spectrometer operates in occultation mode only and its
   boresight is pointing +67.07 deg from the spacecraft -Y axis to the -X
   in the XY plane.

   Therefore, the ACS MIR frame is defined as follows (from [9]):

      -  +Z axis is parallel to the spectrometer boresight, pointing
         at +67.07 degrees from the spacecraft -Y axis towards the -X
         axis in the XY plane;

      -  +X axis is parallel to the spectrometer detector array's lines,
         and it is nominally co-aligned with the spacecraft +Z axis;

      -  +Y axis completes the right handed-frame

      -  the origin of this frame is located at geometrical centre of the
         first folding mirror at the entry optics of the spectrometer.


   This diagrams illustrates the nominal TGO_ACS_MIR frame with respect
   to the spacecraft frame.

   +Z S/C side view
   ----------------

                'Nadir'  Towards Mars
                         ^
               +Xmir ^   |         +Zmir
                      \  |_~67deg .^
                       \ |  \  .'    Science Deck
                   ._ACS\| .'________.
                   |     o           |
                  ============o=======
                  SA+Z               |__..--,
                   |            <-----o-..__|
                   |          +Xsc   ||     ME
                   ._________________.|
                                   o-o|/|
                                    \|V +Ysc
                                     o  | :
                                      \ |/
                                       \|
                                           HGA

                                    +Zsc and +Ymir are out of the page.


   Nominally, a rotation of 90 degrees about +X spacecraft axis and then
   a rotation of -67.07 degrees about the +Y resulting axis are required to
   align the TGO_SPACECRAFT to the TGO_ACS_MIR frames.

   The following in-flight calibrated misalignment boresight was provided
   by A. Trokhimovskiy on May 10, 2017 [19]:

     ACS_MIR Boresight:   ( -0.9215, -0.3884, -0.0003 )


   This boresight is relative to the TGO_SPACECRAFT frame. Given this
   boresight the rotation from the TGO_SPACECRAFT frame to the
   TGO_ACS_MIR frame determined from the in-flight calibration
   data can be represented by the following rotation angles in degrees:

       mir
      M    = |0.0|  * |-67.14521755516077|  * |90.0442552276922|
       base       Z                      Y                        X


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_ACS_MIR                =  -143120
      FRAME_-143120_NAME               = 'TGO_ACS_MIR'
      FRAME_-143120_CLASS              =   4
      FRAME_-143120_CLASS_ID           =  -143120
      FRAME_-143120_CENTER             =  -143
      TKFRAME_-143120_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143120_SPEC             = 'ANGLES'
      TKFRAME_-143120_UNITS            = 'DEGREES'
      TKFRAME_-143120_AXES             = ( 1,     2,      3    )
      TKFRAME_-143120_ANGLES           = ( -90.0442552276922,
                                           +67.14521755516077,
                                           +0.000000000000000  )

   \begintext


FREND Frames
------------------------------------------------------------------------

   This section of the file contains the definitions of the
   Fine-Resolution Epithermal Neutron Detector (FREND) instrument
   frames.


FREND Frame Tree
~~~~~~~~~~~~~~~~

   The diagram below shows the FREND frame hierarchy.


                               "J2000" INERTIAL
           +-----------------------------------------------------+
           |                          |                          |
           |<-pck                     |                          |<-pck
           |                          |                          |
           V                          |                          V
       "IAU_MARS"                     |                     "IAU_EARTH"
     MARS BODY-FIXED                  |<-ck               EARTH BODY-FIXED
     ---------------                  |                   ----------------
                                      V
                               "TGO_SPACECRAFT"
                               ----------------
                                      |
                                      |<-fixed
                                      |
                                      V
                                 "TGO_FREND"
                                 -----------


FREND base frame
~~~~~~~~~~~~~~~~

   The Fine-Resolution Epithermal Neutron Detector (FREND) is rigidly mounted
   on the spacecraft Science Deck. The base frame -- TGO_FREND, associated to
   it, maps the TGO spacecraft reference axis defined in the mechanical
   drawings and it is specified as a fixed-offset frame with its orientation
   aligned to the TGO_SPACECRAFT frame in order to simplify the science
   operations and commanding of the instrument, as requested by the FREND
   Instrument Team (see [8]).

   The FREND base frame is defined by the detector design and its mounting on
   the spacecraft as follows (from [8]):

      -  -Y axis is along the nominal FREND 3He counters and stilbene-based
         scintillator boresights; and it is nominally co-aligned with the
         spacecraft -Y axis;

      -  +Z axis is nominally co-aligned with the spacecraft +Z axis;

      -  +X axis completes the right-handed frame;

      -  the origin of the frame is located at the geometrical center of
         the FREND stilbene scintillator detector.

   These diagrams illustrate the TGO_FREND frame with respect to the
   TGO_SPACECRAFT frame:

   -X S/C side (HGA side) view:
   ----------------------------
                                    ^
                                    | toward Mars
                                    |

                                    Science deck
                             ._____________.
   .__  _______________      |    <-------x|+Zfrend_____________  ___.
   |  \ \               \    |   +Xfrend  ||    /               \ \  |
   |  / /                \   |     ___    ||   /                / /  |
   |  \ \                 `. |    / _+Xsc || .'                 \ \  |
   |  / /             +Zsc <--------x) |  v|o |                 / /  |
   |  \ \                 .' |    \_|_+Yfrend`.                 \ \  |
   |  / /                /   |      |      |   \                / /  |
   .__\ \_______________/    |      |      |    \_______________\ \__.
       +Z Solar Array        ._____ v +Ysc .      -Z Solar Array
                               .'       `.
                              /           \
                             .   `.   .'   .             +Xsc and +Xfrend
                             |     `o'     |              are into the page.
                             .      |      .
                              \     |     /
                               `.       .'
                            HGA  ` --- '


   -Y S/C side (Science Deck side) view:
   -------------------------------------

                                  _____
                                 /     \  EDM
                                |       |
                             ._____________.
                             |             |
                             |            ^|+Xfrend
                      +Zsc   |            ||
   o==/ /==================o |            | >o==================/ /==o
     +Z Solar Array          |          .-||        -Z Solar Array
                         +Zfrend <--------x|+Yfrend
                             |    +Xsc     |
                             |      ^      |
                             |      |      |
                             |      |      |
                        +Zsc .______|______.            +Ysc is into the
                            <-------x   `. ME            page;
                              /______+Ysc \             +Ysc and +Yfrend
                           HGA    `.|.'                  are into the page.


   -Y FREND side (spacecraft Science Deck) view:
   ---------------------------------------------


    +Ysc                         .-----------.
       X--------->              |  Dosimeter  |
       |       +Zsc             |             |
       |        .---------------------------------------------.
       |        |        _____________________________        |
       | +Xsc   |     .'                               '.     |
       v        |   .'   . --- .               . --- .   '.   |
                |  /   /         \           /         \   \  |
                | .   .           .         .           .   . |
                | |   |           |         |           |   | |
                | |   .           .         .           .   | |
                | |    \         /           \         /    | |
                | |      . ___ .     . - .     . ___ .      | |
                | |                /       \   +Zfrend      | |
                | |           +Yfrend  x--------->          | |
                | |                \   |   /                | |
                | |      . --- .     . | .     . --- .      | |
                | |    /         \     |     /         \    | |
                | |   .           .    |    .           .   | |
                | |   |           |    v    |           |   | |
                | .   .           .    +Xfrend          .   . |
                |  \   \         /           \         /   /  |
                |    .   . ___ .               . ___ .   .    |
                |     '.                               .'     |
                |        '---------------------------'        |
                '---------------------------------------------'


                                                    +Ysc and +Yfrend are
                                                     into the page.


   Nominally, the TGO_FREND and the TGO_SPACECRAFT frames are co-aligned.

   \begindata

      FRAME_TGO_FREND                  =  -143200
      FRAME_-143200_NAME               = 'TGO_FREND'
      FRAME_-143200_CLASS              =   4
      FRAME_-143200_CLASS_ID           =  -143200
      FRAME_-143200_CENTER             =  -143
      TKFRAME_-143200_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143200_SPEC             = 'ANGLES'
      TKFRAME_-143200_UNITS            = 'DEGREES'
      TKFRAME_-143200_AXES             = ( 1,   2,      3   )
      TKFRAME_-143200_ANGLES           = ( 0.0, 0.0,    0.0 )

   \begintext


NOMAD Frames
------------------------------------------------------------------------

   This section of the file contains the definitions of the Nadir and
   Occultation for MArs Discovery (NOMAD) instrument frames.


NOMAD Frame Tree
~~~~~~~~~~~~~~~~

   The diagram below shows the NOMAD frame hierarchy.


                               "J2000" INERTIAL
           +-----------------------------------------------------+
           |                          |                          |
           |<-pck                     |<-ck                      |<-pck
           |                          |                          |
           V                          |                          V
       "IAU_MARS"                     |                     "IAU_EARTH"
     MARS BODY-FIXED                  |                   EARTH BODY-FIXED
     ---------------                  |                   ----------------
                                      V
                               "TGO_SPACECRAFT"
                        +---------------------------+
                        |             |             |
                        |<-fixed      |<-fixed      |
                        |             |             |
                        |             V             |
                        |       "TGO_NOMAD_SO"      |
                        |       --------------      |
                        v                           |
               "TGO_NOMAD_LNO_BASE"                 |
            +------------------------+              |
            |             |          |              |
            |<-fixed      |<-ck      |<-fixed       |
            |             |          |              |
            V             |          V              |
  "TGO_NOMAD_LNO_OPS_NAD" | "TGO_NOMAD_LNO_OPS_OCC" |
  ----------------------- | ----------------------  |
                          |                         |
                          v                         |<-fixed
                "TGO_NOMAD_LNO_FMM"                 |
                --------------------                v
                         |                "TGO_NOMAD_UVIS_BASE"
                         |<-fixed       +-----------------------+
                         |              |                       |
                         v              |                       |
                  "TGO_NOMAD_LNO"       |<-fixed                |<-fixed
                  ---------------       |                       |
                                        v                       v
                               "TGO_NOMAD_UVIS_NAD"    "TGO_NOMAD_UVIS_OCC"
                               --------------------    --------------------


NOMAD LNO Base Frame
~~~~~~~~~~~~~~~~~~~~

   The NOMAD Limb Nadir and Occultation (LNO) spectrometer is rigidly mounted
   on the spacecraft Science Deck. Therefore, the base frame associated with
   it -- the NOMAD LNO Base frame, TGO_NOMAD_LNO_BASE --  is specified as a
   fixed offset frame with its orientation given relative to the
   TGO_SPACECRAFT frame.

   The NOMAD LNO Base frame is defined as follows (from [10]):

      -  +Y axis is along the nominal spectrometer flip mirror rotation
         axis, and it is nominally co-aligned with the spacecraft +Z axis;

      -  +Z axis is co-aligned with the -Y spacecraft axis and it is along
         the spectrometer boresight in "zero" scanning position;

      -  +X axis completes the right-handed frame, and it is parallel to the
         detector array lines and the wide side of the slit;

      -  the origin of this frame is located at the intersection of the
         spectrometer flip mirror rotation axis and mirror central axis.


   These diagrams illustrate the nominal TGO_NOMAD_LNO_BASE and the
   TGO_NOMAD_UVIS_BASE frame with respect to the spacecraft frame.


   -X S/C side (Main Engine side) view:
   ------------------------------------


                                    ^
                                    | toward Mars
                                    |
            +Znomad_lno_base  ^     |
                              |
                              | Science deck
        +Ynomad_lno_base     .|____________.
   .__  _____________<--------x            |     _______________  ___.
   |  \ \               \  +Xnomad_lno_base|    /               \ \  |
   |  / /                \   |     ___     |   /                / /  |
   |  \ \                 `. |    / _+Xsc || .'                 \ \  |
   |  / /             +Zsc <--------x) |  v|o |                 / /  |
   |  \ \                 .' |    \_|_+Yfrend`.                 \ \  |
   |  / /                /   |      |      |   \                / /  |
   .__\ \_______________/    |      |      |    \_______________\ \__.
       +Z Solar Array        ._____ v +Ysc .      -Z Solar Array
                               .'       `.
                              /           \
                             .   `.   .'   .             +Xsc and
                             |     `o'     |             +Xnomad_lno_base
                             .      |      .              are into the page.
                              \     |     /
                               `.       .'
                            HGA  ` --- '


   -Y S/C side (Science Deck side) view:
   -------------------------------------

                                  _____
              +Xnomad_lno_base ^ /     \ EDM
                               ||       |
                             ._|___________.
            +Ynomad_lno_base | | |         |
                        <------o |         |
                             |__+Znomad_lno_base
   o==/ /==================o<|             |>o==================/ /==o
     +Z Solar Array          |             |        -Z Solar Array
                             |             |
                             |    +Xsc     |
                             |      ^      |
                             |      |      |
                             |      |      |
                        +Zsc .______|______.            +Ysc is into the
                            <-------x   `. ME            page;
                              /______+Ysc \             +Znomad_lno_base is
                           HGA    `.|.'                  out of the page.


   Nominally, a single rotation of 90 degrees about +X spacecraft axis is
   required to align the TGO_SPACECRAFT to the TGO_NOMAD_LNO_BASE frame.


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_NOMAD_LNO_BASE         =  -143300
      FRAME_-143300_NAME               = 'TGO_NOMAD_LNO_BASE'
      FRAME_-143300_CLASS              =   4
      FRAME_-143300_CLASS_ID           =  -143300
      FRAME_-143300_CENTER             =  -143
      TKFRAME_-143300_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143300_SPEC             = 'ANGLES'
      TKFRAME_-143300_UNITS            = 'DEGREES'
      TKFRAME_-143300_AXES             = ( 3,     2,      1   )
      TKFRAME_-143300_ANGLES           = ( 0.0,   0.0,  -90.0 )

   \begintext


NOMAD UVIS Base Frame
~~~~~~~~~~~~~~~~~~~~~

   The NOMAD Ultraviolet and Visible Spectrometer (UVIS) is rigidly mounted
   on the spacecraft Science Deck. Therefore, the base frames associated with
   it -- the NOMAD UVIS Base frame, TGO_NOMAD_UVIS_BASE --  are specified as
   a fixed offset frame with its orientation given relative to the
   TGO_SPACECRAFT frame.

   The NOMAD UVIS Base frame is defined as follows (from [10]):

      -  +Y axis is along the nominal spectrometer fibre optic switch rotation
         axis, and it is nominally co-aligned with the spacecraft +Z axis;

      -  +Z axis is co-aligned with the -Y spacecraft axis and it is along
         the spectrometer boresight in "zero" scanning position;

      -  +X axis completes the right-handed frame, and it is parallel to the
         detector array lines and the wide side of the slit;

      -  the origin of this frame is located at the intersection of the
         spectrometer fibre optic switch rotation axis and mirror central
         axis.


   These diagrams illustrate the nominal TGO_NOMAD_UVIS_BASE frame with
   respect to the spacecraft frame.


   -X S/C side (HGA side) view:
   ----------------------------


                                    ^
                                    | toward Mars
                                    |
           +Znomad_uvis_base  ^     |
                              |
                              | Science deck
       +Ynomad_uvis_base     .|____________.
   .__  _____________<--------x            |     _______________  ___.
   |  \ \               \  +Xnomad_uvis_base    /               \ \  |
   |  / /                \   |     ___     |   /                / /  |
   |  \ \                 `. |    / _+Xsc || .'                 \ \  |
   |  / /             +Zsc <--------x) |  v|o |                 / /  |
   |  \ \                 .' |    \_|_+Yfrend`.                 \ \  |
   |  / /                /   |      |      |   \                / /  |
   .__\ \_______________/    |      |      |    \_______________\ \__.
       +Z Solar Array        ._____ v +Ysc .      -Z Solar Array
                               .'       `.
                              /           \
                             .   `.   .'   .             +Xsc and
                             |     `o'     |             +Xnomad_uvis_base
                             .      |      .              are into the page.
                              \     |     /
                               `.       .'
                            HGA  ` --- '


   -Y S/C side (Science Deck side) view:
   -------------------------------------

                                  _____
             +Xnomad_uvis_base ^ /     \ EDM
                               ||       |
                             ._|___________.
           +Ynomad_uvis_base | | |         |
                        <------o |         |
                             |___'+Znomad_uvis_base
   o==/ /==================o<|             |>o==================/ /==o
     +Z Solar Array          |             |        -Z Solar Array
                             |             |
                             |    +Xsc     |
                             |      ^      |
                             |      |      |
                             |      |      |
                        +Zsc .______|______.            +Ysc is into the
                            <-------x   `. ME            page;
                              /______+Ysc \             +Znomad_uvis_base is
                           HGA    `.|.'                  out of the page.


   Nominally, a single rotation of 90 degrees about +X spacecraft axis is
   required to align the TGO_SPACECRAFT to the TGO_NOMAD_UVIS_BASE frame.

   Since the SPICE frames subsystem calls for specifying the reverse
   transformation -- going from the instrument or structure frame to the
   base frame -- as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_NOMAD_UVIS_BASE        =  -143330
      FRAME_-143330_NAME               = 'TGO_NOMAD_UVIS_BASE'
      FRAME_-143330_CLASS              =   4
      FRAME_-143330_CLASS_ID           =  -143330
      FRAME_-143330_CENTER             =  -143
      TKFRAME_-143330_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143330_SPEC             = 'ANGLES'
      TKFRAME_-143330_UNITS            = 'DEGREES'
      TKFRAME_-143330_AXES             = ( 3,     2,      1   )
      TKFRAME_-143330_ANGLES           = ( 0.0,   0.0,  -90.0 )

   \begintext


NOMAD LNO flip mirror mechanism frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The NOMAD Limb Nadir and Occultation (LNO) spectrometer has a flip
   mirror that provides the possibility of observation in both nadir and
   solar occultation modes.

   Since this flip mirror rotates with respect to the LNO base, the
   TGO_NOMAD_LNO_FMM frame is defined as a CK frame with its orientation
   provided in a CK file relative to the TGO_NOMAD_LNO_BASE frames.

   The NOMAD LNO flip mirror mechanism frame -- TGO_NOMAD_LNO_FMM -- is defined
   as (from [10]):

      -  +Y axis is along the nominal spectrometer flip mirror rotation
         axis, and it is nominally co-aligned with the spectrometer base +Y
         axis;

      -  +Z axis is parallel to the flip mirror boresight that defines
         the spectrometer boresight; in 'nadir' scanner position is co-aligned
         with the -Y spacecraft axis;

      -  +X axis completes the right-handed frame; and it is parallel to the
         detector array lines and the wide side of the slit;

      -  the origin of this frame is located at the intersection of the
         spectrometer flip mirror rotation axis and mirror central axis.


   These diagrams illustrate the TGO_NOMAD_LNO_FMM frame for scanner
   positions 'nadir' ( 0.0 degrees) and solar 'occultation' (+67.07 deg from
   the spacecraft -Y axis to the -X in the XY plane). Both diagrams are +Z S/C
   side view:


    Scanner in 'nadir' position        Scanner in 'occultation' position
    ---------------------------        ---------------------------------

                +Zbase                             +Zbase
                ^ +Zfmm                            ^
                |                       +Xfmm  ^   |         +Zfmm
                |       Science                 \  |_~67deg .^
     +Xfmm      | NOMAD   Deck                   \ |  \  .'    Science Deck
   +Xbase. _____|___________.               NOMAD_\| .'________.
        <-------o           |              <-------o           |
         ============o========         +Xbase ==========o========
        SA+Z                |__..--,       SA+Z                |__..--,
          |            <-----o-..__|         |            <-----o-..__|
          |          +Xsc   ||     ME        |          +Xsc   ||     ME
          ._________________.|               ._________________.|
                          o-o|/|                             o-o|/|
                           \|V +Ysc                           \|V +Ysc
                            o  | :                             o  | :
                             \ |/                               \ |/
                              \|                                 \|
                                  HGA                                HGA

      +Zsc, +Ybase, and +Yfmm are out of the page;


   These sets of keywords define the NOMAD LNO flip mirror frames:

   \begindata

      FRAME_TGO_NOMAD_LNO_FMM        = -143305
      FRAME_-143305_NAME              = 'TGO_NOMAD_LNO_FMM'
      FRAME_-143305_CLASS             =  3
      FRAME_-143305_CLASS_ID          = -143305
      FRAME_-143305_CENTER            = -143
      CK_-143305_SCLK                 = -143
      CK_-143305_SPK                  = -143

   \begintext


NOMAD LNO Detector Frame
~~~~~~~~~~~~~~~~~~~~~~~~~

   Since the LNO detector receives radiation through the scanner, its frame
   TGO_NOMAD_LNO is defined to be nominally co-aligned with the LNO scanner
   frame TGO_NOMAD_LNO_FMM. This frame is introduced to allow incorporating
   into the LNO frame chain any misalignment between the scanner boresight
   direction and the detector view directions.

   Currently no misalignment data are available, and, therefore, the set of
   keywords below makes these frames co-aligned with their reference.

   \begindata

      FRAME_TGO_NOMAD_LNO             =  -143310
      FRAME_-143310_NAME              = 'TGO_NOMAD_LNO'
      FRAME_-143310_CLASS             =  4
      FRAME_-143310_CLASS_ID          =  -143310
      FRAME_-143310_CENTER            =  -143
      TKFRAME_-143310_RELATIVE        = 'TGO_NOMAD_LNO_FMM'
      TKFRAME_-143310_SPEC            = 'ANGLES'
      TKFRAME_-143310_UNITS           = 'DEGREES'
      TKFRAME_-143310_AXES            = ( 1,   2,   3   )
      TKFRAME_-143310_ANGLES          = ( 0.0, 0.0, 0.0 )

   \begintext


NOMAD LNO nadir and occultation science operations frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Because the NOMAD LNO flip mirrors can be rotated to only
   a limited number of positions for external observations -- 'nadir' and
   'occultation' -- a fixed frame co-aligned with the flip mirror
   frame in each of these positions is defined to allow computing mirror
   orientation without needing to use CK.

   IMPORTANT: Please note that using these frames will not reflect the
   behaviour of the LNO Detector and therefore these frames should only be
   used for science operations purposes.

   Each of these 'fixed-mirror-position science operations' frames is defined
   as a fixed offset frame with respect to the corresponding base frame for
   each of the spectrometers as follows (from [10]):

      -  +Y axis is along the nominal spectrometer flip mirror rotation
         axis, and it is nominally co-aligned with the spectrometer base +Y
         axis;

      -  +Z axis is parallel to the flip mirror boresight that defines
         the spectrometer boresight at a particular angle;

      -  +X axis completes the right-handed frame; and it is parallel to the
         detector array lines and the wide side of the slit;

      -  the origin of this frame is located at the intersection of the
         spectrometer flip mirror rotation axis and mirror central axis.


   These diagrams illustrate fixed mirror pointing directions co-aligned
   with the +Z axis of the corresponding 'fixed-mirror-position' frame:

   +Z S/C side view
   ----------------


                +Z*base                            +Z*base
                ^ +Z*nad                           ^
                |                       +X*occ ^   |         +Z*occ
                |       Science                 \  |_~67deg .^
     +X*nad     | NOMAD   Deck                   \ |  \  .'    Science Deck
   +X*base _____|___________.               NOMAD_\| .'________.
        <-------o           |              <-------o           |
         ============o========         +X*base =========o========
           SA+Z             |__..--,          SA+Z             |__..--,
          |            <-----o-..__|         |            <-----o-..__|
          |          +Xsc   ||     ME        |          +Xsc   ||     ME
          ._________________.|               ._________________.|
                          o-o|/|                             o-o|/|
                           \|V +Ysc                           \|V +Ysc
                            o  | :                             o  | :
                             \ |/                               \ |/
                              \|                                 \|
                                  HGA                                HGA


      +Zsc, +Y*base, +Y*nad and +Y*occ are out of the page;
      ``*base'' corresponds to ``nomad_lno_base'';
      ``*nad'' corresponds to ``nomad_lno_ops_nad''; and
      ``*occ'' corresponds to ``nomad_lno_ops_occ''.


   These 'fixed-mirror-position operations' frames are nominally rotated about
   the +X axis of the corresponding spectrometer base frames by the following
   angles:

      Frame name              Rotation Angle, deg
      ----------------------  -------------------
      TGO_NOMAD_LNO_OPS_NAD         0.00
      TGO_NOMAD_LNO_OPS_OCC       -67.07

   The following in-flight calibrated misalignment boresight was provided
   by Ian Thomas on July 26, 2016 [15] and on May 16, 2017 [19]:

      NOMAD_LNO_OPS_NAD Boresight: ( -0.001047198, -0.9999786, 0.006457718 )
      NOMAD_LNO_OPS_OCC Boresight: ( -0.92148,     -0.38838,   0.00628     )

   These boresights are relative to the TGO_SPACECRAFT frame. Given these
   boresights the rotation from the TGO_NOMAD_LNO_BASE frame to the
   TGO_NOMAD_LNO_NAD and TGO_NOMAD_LNO_NAD frames determined from the
   in-flight calibration data can be represented by the following rotation
   angles in degrees:

       nad
      M      |0.0| *  |-0.06000003670468607|  * |-0.37000276139181304|
       base       Z                         Y                         X

       occ
      M    = |0.0|  * |-67.1431540428917|  * |-0.9263765923679103|
       base       Z                      Y                        X


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_NOMAD_LNO_OPS_NAD      =  -143311
      FRAME_-143311_NAME               = 'TGO_NOMAD_LNO_OPS_NAD'
      FRAME_-143311_CLASS              =   4
      FRAME_-143311_CLASS_ID           =  -143311
      FRAME_-143311_CENTER             =  -143
      TKFRAME_-143311_RELATIVE         = 'TGO_NOMAD_LNO_BASE'
      TKFRAME_-143311_SPEC             = 'ANGLES'
      TKFRAME_-143311_UNITS            = 'DEGREES'
      TKFRAME_-143311_AXES             = ( 1,     2,      3   )
      TKFRAME_-143311_ANGLES           = ( +0.37000276139181304,
                                           +0.06000003670468607,
                                           +0.00000000000000000 )

      FRAME_TGO_NOMAD_LNO_OPS_OCC      =  -143312
      FRAME_-143312_NAME               = 'TGO_NOMAD_LNO_OPS_OCC'
      FRAME_-143312_CLASS              =   4
      FRAME_-143312_CLASS_ID           =  -143312
      FRAME_-143312_CENTER             =  -143
      TKFRAME_-143312_RELATIVE         = 'TGO_NOMAD_LNO_BASE'
      TKFRAME_-143312_SPEC             = 'ANGLES'
      TKFRAME_-143312_UNITS            = 'DEGREES'
      TKFRAME_-143312_AXES             = ( 1,     2,      3    )
      TKFRAME_-143312_ANGLES           = ( +0.9263765923679103,
                                           +67.1431540428917,
                                           +0.0000000000000000 )

   \begintext


NOMAD UVIS nadir and occultation frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   Because the NOMAD LNO UVIS fibre optic switch can only adopt
   a limited number of positions for external observations -- 'nadir' and
   'occultation' -- a fixed frame co-aligned with the fibre optic switch
   frame in each of these positions is defined to allow computing mirror
   orientation without needing to use CK.

   Each of these 'fixed-switch-position' frames is defined as a fixed
   offset frame with respect to the corresponding base frame for each of
   the spectrometers as follows (from [10]):

      -  +Y axis is along the nominal spectrometer fibre optic switch rotation
         axis, and it is nominally co-aligned with the spectrometer base +Y
         axis;

      -  +Z axis is parallel to the fibre optic switch boresight that defines
         the spectrometer boresight at a particular angle;

      -  +X axis completes the right-handed frame; and it is parallel to the
         detector array lines and the wide side of the slit;

      -  the origin of this frame is located at the intersection of the
         spectrometer optic fibre switch rotation axis and mirror central
         axis.


   These diagram illustrates fixed mirror pointing directions co-aligned
   with the +Z axis of the corresponding 'fixed-mirror-position' frame:

   +Z S/C side view
   ----------------


                +Z*base                            +Z*base
                ^ +Z*nad                           ^
                |                       +X*occ ^   |         +Z*occ
                |       Science                 \  |_~67deg .^
     +X*nad     | NOMAD   Deck                   \ |  \  .'    Science Deck
   +X*base _____|___________.               NOMAD_\| .'________.
        <-------o           |              <-------o           |
         ============o========         +X*base =========o========
           SA+Z             |__..--,          SA+Z             |__..--,
          |            <-----o-..__|         |            <-----o-..__|
          |          +Xsc   ||     ME        |          +Xsc   ||     ME
          ._________________.|               ._________________.|
                          o-o|/|                             o-o|/|
                           \|V +Ysc                           \|V +Ysc
                            o  | :                             o  | :
                             \ |/                               \ |/
                              \|                                 \|
                                  HGA                                HGA


      +Zsc, +Y*base, +Y*nad and +Y*occ are out of the page;
      ``*base'' corresponds to ``nomad_uvis_base'';
      ``*nad'' corresponds to ``nomad_uvis_nad''; and
      ``*occ'' corresponds to ``nomad_uvis_occ''.


   These 'fixed-mirror-position' frames are nominally rotated about the
   +X axis of the corresponding spectrometer base frames by the following
   angles:

      Frame name              Rotation Angle, deg
      ----------------------  -------------------
      TGO_NOMAD_UVIS_NAD         0.00
      TGO_NOMAD_UVIS_OCC       -67.07


   The following in-flight calibrated misalignment boresights were provided
   by Ian Thomas on July 26, 2016 [15]; September 16, 2016 [18]; May 16, 2017
   [19] and on June 13 2018 [22]:

   NOMAD_UVIS_NAD: ( -0.002312108759,   -0.999990516156,   0.003690765731    )
   NOMAD_UVIS_OCC: ( -0.922221097920913,-0.386613383297695,0.006207330031467 )


   These boresights are relative to the TGO_SPACECRAFT frame. Given these
   boresights the rotation from the TGO_NOMAD_UVIS_BASE frame to the
   TGO_NOMAD_UVIS_NAD and TGO_NOMAD_UVIS_NAD frames determined from the
   in-flight calibration data can be represented by the following rotation
   angles in degrees:

       nad
      M    = |0.0|  * |-0.13247419169719835|  * |-0.21146634488534072|
       base       Z                         Y                        X

       occ
      M    = |0.0|  * |-67.25296897005113|  * |-0.9198420756243424|
       base       Z                      Y                        X


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_NOMAD_UVIS_NAD         =  -143331
      FRAME_-143331_NAME               = 'TGO_NOMAD_UVIS_NAD'
      FRAME_-143331_CLASS              =   4
      FRAME_-143331_CLASS_ID           =  -143331
      FRAME_-143331_CENTER             =  -143
      TKFRAME_-143331_RELATIVE         = 'TGO_NOMAD_UVIS_BASE'
      TKFRAME_-143331_SPEC             = 'ANGLES'
      TKFRAME_-143331_UNITS            = 'DEGREES'
      TKFRAME_-143331_AXES             = ( 1,     2,      3      )
      TKFRAME_-143331_ANGLES           = ( +0.21146634488534072,
                                           +0.13247419169719835,
                                           +0.00000000000000000  )

      FRAME_TGO_NOMAD_UVIS_OCC         =  -143332
      FRAME_-143332_NAME               = 'TGO_NOMAD_UVIS_OCC'
      FRAME_-143332_CLASS              =   4
      FRAME_-143332_CLASS_ID           =  -143332
      FRAME_-143332_CENTER             =  -143
      TKFRAME_-143332_RELATIVE         = 'TGO_NOMAD_UVIS_BASE'
      TKFRAME_-143332_SPEC             = 'ANGLES'
      TKFRAME_-143332_UNITS            = 'DEGREES'
      TKFRAME_-143332_AXES             = ( 1,     2,      3     )
      TKFRAME_-143332_ANGLES           = ( +0.9198420756243424,
                                           +67.252968970051130,
                                           +0.0000000000000000  )

   \begintext


NOMAD Solar Occultation (SO) spectrometer frame:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The NOMAD Solar Occultation (SO) spectrometer is rigidly mounted on
   the spacecraft Science Deck. Therefore, the frame associated with
   it -- the NOMAD SO frame, TGO_NOMAD_SO -- is specified as a fixed
   offset frame with its orientation given relative to the TGO_SPACECRAFT
   frame.

   The NOMAD SO spectrometer operates in occultation mode only and its
   boresight is pointing approximately +67.07 deg from the
   spacecraft -Y axis to the -X in the XY plane.

   Therefore, the NOMAD SO frame is defined as follows (from [10]):

      -  +Z axis is parallel to the spectrometer boresight, pointing
         approximately at +67.07 degrees from the spacecraft -Y axis
         towards the -X axis in the XY plane;

      -  +X axis is parallel to the spectrometer detector array's lines,
         and it is approximately at +67.07 degrees from the spacecraft
         +X axis;

      -  +Y axis completes the right-handed frame and it is nominally
         aligned with the spacecraft +Z axis;

      -  the origin of this frame is located at geometrical centre of the
         first folding mirror at the entry optics of the spectrometer.


   This diagrams illustrates the nominal TGO_NOMAD_SO frame with respect
   to the spacecraft frame.

   +Z S/C side view
   ----------------

                'Nadir'  Towards Mars
                         ^
                +Xso ^   |         +Zso
                      \  |_~67deg .^
                       \ |  \  .'    Science Deck
                   NOMAD\| .'________.
                   |     o +Yso      |
                  ============o=======
                  SA+Z               |__..--,
                   |            <-----o-..__|
                   |          +Xsc   ||     ME
                   ._________________.|                +Zsc and +Ymir are
                                   o-o|/|               out of the page.
                                    \|V +Ysc
                                     o  | :
                                      \ |/
                                       \|
                                           HGA


   Nominally, a rotation of 90 degrees about +X spacecraft axis and then
   a rotation of -67.07 degrees about the +Y spacecraft axis are required to
   align the TGO_SPACECRAFT to the TGO_NOMAD_SO frames.

   The following in-flight calibrated misalignment boresight was provided
   by Ian Thomas on July 26, 2016 [15] and June 13, 2018 [22]:

      NOMAD_SO Boresight:  ( -0.9218973, -0.38738526, 0.00616719 )


   This boresight is relative to the TGO_SPACECRAFT frame. Given this
   boresight the rotation from the TGO_SPACECRAFT frame to the
   TGO_NOMAD_SO frame determined from the in-flight calibration
   data can be represented by the following rotation angles in degrees:

       so
      M    = |0.0|  * |-67.20504912816348|  * |89.0879257751404|
       sc         Z                       Y                     X


   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_NOMAD_SO               =  -143320
      FRAME_-143320_NAME               = 'TGO_NOMAD_SO'
      FRAME_-143320_CLASS              =   4
      FRAME_-143320_CLASS_ID           =  -143320
      FRAME_-143320_CENTER             =  -143
      TKFRAME_-143320_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143320_SPEC             = 'ANGLES'
      TKFRAME_-143320_UNITS            = 'DEGREES'
      TKFRAME_-143320_AXES             = ( 1,     2,      3    )
      TKFRAME_-143320_ANGLES           = ( -89.08792577514040
                                           +67.20504912816348,
                                           +0.000000000000000  )

   \begintext


CaSSIS Frames
------------------------------------------------------------------------

   This section of the file contains the definitions of the Colour and
   Stereo Surface Imaging System (CaSSIS) instrument frames.


CaSSIS Frame Tree
~~~~~~~~~~~~~~~~~

   The diagram below shows the CaSSIS frame hierarchy.


                               "J2000" INERTIAL
           +-----------------------------------------------------+
           |                          |                          |
           |<-pck                     |                          |<-pck
           |                          |                          |
           V                          |                          V
       "IAU_MARS"                     |                     "IAU_EARTH"
     MARS BODY-FIXED                  |<-ck               EARTH BODY-FIXED
     ---------------                  |                   ----------------
                                      V
                               "TGO_SPACECRAFT"
                               ----------------
                                      |
                                      |<-fixed
                                      |
                                      V
                               "TGO_CASSIS_CRU"
                               ----------------
                                      |
                                      |<-ck
                                      |
                                      V
                               "TGO_CASSIS_TEL"
                               ----------------
                                      |
                                      |<-fixed
                                      |
                               "TGO_CASSIS_FSA"
                               ----------------


CaSSIS Camera Rotation Unit frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The CaSSIS Camera Rotation Unit (CRU) is rigidly mounted on the spacecraft
   Science Deck. Therefore, the frame associated with it -- the CaSSIS CRU
   frame, TGO_CASSIS_CRU --  is specified as a fixed offset frame relative
   with its orientation given relative to the TGO_SPACECRAFT frame.

   The CaSSIS CRU frame is defined by the camera rotation unit design and
   its mounting on the spacecraft as follows (from [5]):

      -  +Y axis is along the nominal CaSSIS CRU rotation axis, and it is
         nominally co-aligned with the spacecraft -Y axis

      -  +Z axis is co-aligned with the +Z spacecraft axis;

      -  +X axis completes the right-handed frame;

      -  the origin of the frame is located at the center of the CaSSIS
         reference hole (RH) at the instrument's interface plane, i.e. the
         unit mounting plane to the spacecraft.

   Any misalignment between the nominal and actual CaSSIS CRU rotation axis
   measured pre-launch or during in-flight calibration should be incorporated
   into the definition of this frame.


   These diagrams illustrate the nominal TGO_CASSIS_CRU frame with respect to
   the spacecraft frame.


   -X S/C side (Main Engine side) view:
   ------------------------------------
                                    ^
                                    | toward Mars
                       +Ycru  ^
                              |
                              | Science deck
                    +Zcru    .|____________.
   .__  _____________<--------o            |     .______________  ___.
   |  \ \               \   +Xcru          |    /               \ \  |
   |  / /                \   |     ___     |   /                / /  |
   |  \ \                 `. |    / _+Xsc || .'                 \ \  |
   |  / /             +Zsc <--------x) |  v|o |                 / /  |
   |  \ \                 .' |    \_|_+Yfrend`.                 \ \  |
   |  / /                /   |      |      |   \                / /  |
   .__\ \_______________/    |      |      |    \_______________\ \__.
       +Z Solar Array        ._____ v +Ysc .      -Z Solar Array
                               .'       `.
                              /           \
                             .   `.   .'   .           +Xsc is into
                             |     `o'     |            the page and +Xcru
                             .      |      .            is out of the page.
                              \     |     /
                               `.       .'
                            HGA  ` --- '


   -Y S/C side (Science Deck side) view:
   -------------------------------------

                                  _____
                                 /     \  EDM
                                |       |
                             ._____________.
                             |             |
                             |             |
                             |             |
   o==/ /==================o<|             |>o==================/ /==o
     +Z Solar Array          |--.          |        -Z Solar Array
                      <--------o|+Ycru     |
                    +Zcru    |-|'          |
                             | |    ^ +Xsc |
                             | |    |      |
                         +Xcru v    |      |            +Ysc is into the
                        +Zsc .______|______.             page; +Ycru is out
                            <-------x   `. ME            of the page
                              /______+Ysc \
                           HGA    `.|.'


   +Z CaSSIS Rotation Unit side view (motor in "Launch" position: 180 deg):
   ----------------------------------------------------------------------------


        <-------x +Zsc      +Xcru
     +Ysc       |             ^
                |             |
                |             |
          +Xsc  v           .-|.                             .
                            |.||                         . ' /
                             |||      ______________ . '    /
                            /-|-.  .-|  ___    ____ ' \    ,
                          /'  | |  | | /   \  /    '. '.\  /
                         / |  | |  | ||     ||       '  ' .'
                  .-----.  |  | |  | ||     ||         '.  '.
                 /___,    ,|  | |  | ||     ||            '.  '.
                    /    / |  | |= | | \___/ \_____________/    '.    +Ycru
                   /     \ |  x--------------------------------------------->
                  /       '+Zcru|  |.|  /    | |  rotation axis   | | |
                 /         |    |  '-'_._____| |_______' '________' | |
                /          |    |     | |           .---  .------.  '-'
               /           '----|     | |     , -- .  '.          \  \_.-.
              /                 |     | |    |        '.  '.       \   | |
             /__________________|     '-'_   |            '. .      |  | |
            ||             |.--.|         \   \______________.         | |
            ||             ||  ||          \ .-------------------------'-'
      .-----''-------------''--''--------.  \|
      |                                  |
      '----------------------------------'


                                                           +Zsc and +Zcru are
                                                            into the page.


   Nominally, a single rotation of 180 degrees about +Z axis is required to
   co-align the TGO_CASSIS_CRU and the TGO_SPACECRAFT frames. The current
   rotation is derived form Mars Commissioning data and data from MTP000
   (MCO, bands misalignment is <2 pixel) 21].

   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_CASSIS_CRU             =  -143400
      FRAME_-143400_NAME               = 'TGO_CASSIS_CRU'
      FRAME_-143400_CLASS              =   4
      FRAME_-143400_CLASS_ID           =  -143400
      FRAME_-143400_CENTER             =  -143
      TKFRAME_-143400_RELATIVE         = 'TGO_SPACECRAFT'
      TKFRAME_-143400_SPEC             = 'ANGLES'
      TKFRAME_-143400_UNITS            = 'DEGREES'
      TKFRAME_-143400_AXES             = ( 1,   2,       3   )
      TKFRAME_-143400_ANGLES           = ( 0.021, 0.120, -179.881 )

   \begintext


CaSSIS Telescope frame
~~~~~~~~~~~~~~~~~~~~~~

   The CaSSIS telescope rotates counterclockwise around the +Y CaSSIS Camera
   Rotation Unit by an angle commanded from ground which ranges from 0
   degrees in the homing position to 360.0 degrees at the end of the range.

   The CaSSIS Telescope frame is defined by the camera rotation unit design
   as follows (from [5]):

      -  +Y axis is along the nominal CaSSIS CRU rotation axis, and it is
         'zero' position is co-aligned with the TGO_CASSIS_CRU +Y axis

      -  +Z axis is co-aligned with the TGO_CASSIS_CRU +Z axis, when the
         motor is in its 'zero' position;

      -  +X axis completes the right-handed frame;

      -  the origin of the frame is located at the focal point of the CaSSIS
         telescope.


   This diagram illustrates the TGO_CASSIS_TEL frame with respect to
   the TGO_CASSIS_CRU frame for different motor rotations.


   +Y CaSSIS Camera Rotation Unit view:
   ------------------------------------


                                  ^ +Xtel            ^ +Xcru
                            0 deg |                  |
                         position |                  |
                              ....|....              |
                           .'     |     `.           |         +Zcru
                         .'       |       `.         o--------->
                        '         |         '      +Ycru
                       .          |          .
                       .          |          .
           90 deg ---- .          o--------------> +Ztel
           position    .           +Ytel     .
                       .                     .
                     .-'.                   .'-.
                    '    .                 .    '
               120 deg    `.             .'     240 deg
               position     ` ......... '       position
                                  |
                                  |
                                180 deg
                                position
         +Xsc ^
              |
              |                                    +Ytel and +Ycru are out of
              |                                    the page; +Ysc is into
              |                                    the page.
     <--------x
   +Zsc        +Ysc


   For an arbitrary scanner angle, the TGO_CASSIS_TEL frame is rotated by
   this angle about the +Y axis with respect to the TGO_CASSIS_CRU frame.

   This set of keywords define the CaSSIS Telescope frame:

   \begindata

      FRAME_TGO_CASSIS_TEL             = -143410
      FRAME_-143410_NAME               = 'TGO_CASSIS_TEL'
      FRAME_-143410_CLASS              =  3
      FRAME_-143410_CLASS_ID           = -143410
      FRAME_-143410_CENTER             = -143
      CK_-143410_SCLK                  = -143
      CK_-143410_SPK                   = -143

   \begintext


CaSSIS Filter Strip Assembly (FSA) frame:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   The CaSSIS Camera Rotation Unit is designed to support quasi-simultaneous
   stereo imaging by aligning the image columns to the orbital track
   (see [6]). Above the detector surface, a filter strip is placed in order
   providing images in 4 different wavelength bands.

   The CaSSIS Filter Strip Assembly -- TGO_CASSIS_FSA frame is defined based
   on the telescope pointing axis and the detector surface orientation as
   follows:

      -  +Z axis points along the telescope optical boresight;

      -  +X axis is aligned to the detector rows; and it is nominally aligned
         to the +Ztel axis.

      -  +Y axis is aligned to the detector columns;

      -  the origin of the frame is located at the focal point of the CaSSIS
         telescope.

   The CaSSIS telescope optical boresight nominally points 10 degrees off
   from the CaSSIS rotation axis -- TGO_CASSIS_TEL +Y axis, towards the
   + X TGO_CASSIS_TEL axis.

   This diagram illustrates the TGO_CASSIS_FSA frame with respect to
   the TGO_CASSIS_TEL frame.


   +Z CaSSIS Rotation Unit side view:
   ----------------------------------


                           ^ +Ytel (rotation axis)
              ^            |    towards Mars
               \           |
          +Zfsa \          |    .---------.
                 \    .----|-.  .---------|   .> +Ysfa
                  \   '----|-'.'    .-.   | .'
           ..      \ /     |      .'   |  .'
           \ `.     \  ..  |-.  .'    .'.'|
            \  `.  / \/ |  | | |    .'.'  |
             \   `/  /\ |  | | |  .'.' .  |
              \  /  /  \|  | | |.'.' .' | |
               \/  /    \  | |  .' .'   | |
               /  /     |\ |.'.'  |     | |
              |  |      | \|.' .  |     | |
              |  |      |  o +Xfsa      | |
              |  |      |  |'   | |     | |
              |   \___.'   |    | |     | |
              |    ____    |    | |     | |
              |   /    \   |    | |     | |
              |  |      |  |----' '-----' '__
              |  |      |  |---.             .'         +Ztel and +Xfsa
              |  |      |  |   |          .'            are out of the page
              |  |      |  |   |.________'
              |   \____/   |---'|'------'
              '____________|____'
               '-----------|---'                   +Xtel
                           o-------------------------->
                         +Ztel

   Nominally, the off-pointing from the CaSSIS Filter Strip Assembly
   boresight and the CaSSIS Telescope rotation axis is 10 degrees towards
   the -X CaSSIS Telescope axis.

   During the geometrical measurements performed before the integration
   on the spacecraft, the angle between the rotation axis and the instrument
   optical boresight at 5 different positions (see p.34 of [6]) was:

       alpha = 9.89 +/- 0.1 degrees

   Therefore, in order to transform the CaSSIS Telescope frame into the
   CaSSIS Filter Strip Assembly frame, first an initial rotation of -90
   degrees about the +Y axis is required, followed by a -80.11 degrees
   rotation about the resulting +X axis.

   Later on as a result from the Mars Commissioning and MTP000 science data
   it the rotations have been adjusted [21].

   Since the SPICE frames subsystem calls for specifying the reverse
   transformation--going from the instrument or structure frame to the
   base frame--as compared to the description given above, the order of
   rotations assigned to the TKFRAME_*_AXES keyword is also reversed
   compared to the above text, and the signs associated with the
   rotation angles assigned to the TKFRAME_*_ANGLES keyword are the
   opposite from what is written in the above text.

   \begindata

      FRAME_TGO_CASSIS_FSA             =  -143420
      FRAME_-143420_NAME               = 'TGO_CASSIS_FSA'
      FRAME_-143420_CLASS              =  4
      FRAME_-143420_CLASS_ID           =  -143420
      FRAME_-143420_CENTER             =  -143
      TKFRAME_-143420_RELATIVE         = 'TGO_CASSIS_TEL'
      TKFRAME_-143420_SPEC             = 'ANGLES'
      TKFRAME_-143420_UNITS            = 'DEGREES'
      TKFRAME_-143420_AXES             = (  2,      1,      3     )
      TKFRAME_-143420_ANGLES           = ( 89.714, 80.005,  0.168 )

   \begintext


TGO NAIF ID Codes -- Definitions
===============================================================================

   This section contains name to NAIF ID mappings for the ExoMars 2016 mission.
   Once the contents of this file is loaded into the KERNEL POOL, these
   mappings become available within SPICE, making it possible to use names
   instead of ID code in the high level SPICE routine calls.

      Name                     ID       Synonyms
      ---------------------    -------  -----------------------

   Spacecraft:
   -----------
      TGO                      -143     TRACE GAS ORBITER
                                        EXOMARS 2016 TGO
                                        EXOMARS TGO
      TGO_SPACECRAFT           -143000
      TGO_HGA_STOWED           -143020
      TGO_HGA                  -143025
      TGO_LGA+Z                -143031
      TGO_LGA-Z                -143032
      TGO_LGA+X                -143033
      TGO_STR-1                -143041
      TGO_STR-2                -143042
      TGO_SA+Z_GIMBAL          -143060
      TGO_SA+Z_C1              -143061
      TGO_SA+Z_C2              -143062
      TGO_SA+Z_C3              -143063
      TGO_SA+Z_C4              -143064
      TGO_SA-Z_GIMBAL          -143070
      TGO_SA-Z_C1              -143071
      TGO_SA-Z_C2              -143072
      TGO_SA-Z_C3              -143073
      TGO_SA-Z_C4              -143074

   ACS:
   ----
      TGO_ACS                  -143100
      TGO_ACS_NIR_NAD          -143111
      TGO_ACS_NIR_OCC          -143112
      TGO_ACS_MIR              -143120
      TGO_ACS_TIRVIM           -143130
      TGO_ACS_TIRVIM_SCAN_BBY  -143131
      TGO_ACS_TIRVIM_SCAN_SPC  -143132
      TGO_ACS_TIRVIM_SCAN_NAD  -143133
      TGO_ACS_TIRVIM_SCAN_OCC  -143134
      TGO_ACS_TIRVIM_SUN       -143140
      TGO_ACS_TIRVIM_SUN_BSR   -143141

   FREND:
   ------
      TGO_FREND                -143200
      TGO_FREND_HE             -143210
      TGO_FREND_SC             -143220

   NOMAD:
   ------
      TGO_NOMAD                -143300
      TGO_NOMAD_LNO            -143310
      TGO_NOMAD_LNO_OPS_NAD    -143311
      TGO_NOMAD_LNO_OPS_OCC    -143312
      TGO_NOMAD_SO             -143320
      TGO_NOMAD_UVIS_NAD       -143331
      TGO_NOMAD_UVIS_OCC       -143332

   CaSSIS:
   -------
      TGO_CASSIS               -143400
      TGO_CASSIS_PAN           -143421
      TGO_CASSIS_RED           -143422
      TGO_CASSIS_NIR           -143423
      TGO_CASSIS_BLU           -143424


   The mappings summarized in this table are implemented by the keywords
   below.

   \begindata

      NAIF_BODY_NAME += ( 'TRACE GAS ORBITER'           )
      NAIF_BODY_CODE += ( -143                          )

      NAIF_BODY_NAME += ( 'TGO'                         )
      NAIF_BODY_CODE += ( -143                          )

      NAIF_BODY_NAME += ( 'EXOMARS TGO'                 )
      NAIF_BODY_CODE += ( -143                          )

      NAIF_BODY_NAME += ( 'EXOMARS 2016 TGO'            )
      NAIF_BODY_CODE += ( -143                          )

      NAIF_BODY_NAME += ( 'TGO_PLAN'                    )
      NAIF_BODY_CODE += ( -143999                       )

      NAIF_BODY_NAME += ( 'TGO PLAN'                    )
      NAIF_BODY_CODE += ( -143999                       )

      NAIF_BODY_NAME += ( 'TGO_SPACECRAFT'              )
      NAIF_BODY_CODE += ( -143000                       )

      NAIF_BODY_NAME += ( 'TGO_HGA_STOWED'              )
      NAIF_BODY_CODE += ( -143020                       )

      NAIF_BODY_NAME += ( 'TGO_HGA'                     )
      NAIF_BODY_CODE += ( -143025                       )

      NAIF_BODY_NAME += ( 'TGO_LGA+Z'                   )
      NAIF_BODY_CODE += ( -143031                       )

      NAIF_BODY_NAME += ( 'TGO_LGA-Z'                   )
      NAIF_BODY_CODE += ( -143032                       )

      NAIF_BODY_NAME += ( 'TGO_LGA+X'                   )
      NAIF_BODY_CODE += ( -143033                       )

      NAIF_BODY_NAME += ( 'TGO_STR-1'                   )
      NAIF_BODY_CODE += ( -143041                       )

      NAIF_BODY_NAME += ( 'TGO_STR-2'                   )
      NAIF_BODY_CODE += ( -143042                       )

      NAIF_BODY_NAME += ( 'TGO_SA+Z_GIMBAL'             )
      NAIF_BODY_CODE += ( -143060                       )

      NAIF_BODY_NAME += ( 'TGO_SA+Z_C1'                 )
      NAIF_BODY_CODE += ( -143061                       )

      NAIF_BODY_NAME += ( 'TGO_SA+Z_C2'                 )
      NAIF_BODY_CODE += ( -143062                       )

      NAIF_BODY_NAME += ( 'TGO_SA+Z_C3'                 )
      NAIF_BODY_CODE += ( -143063                       )

      NAIF_BODY_NAME += ( 'TGO_SA+Z_C4'                 )
      NAIF_BODY_CODE += ( -143064                       )

      NAIF_BODY_NAME += ( 'TGO_SA-Z_GIMBAL'             )
      NAIF_BODY_CODE += ( -143070                       )

      NAIF_BODY_NAME += ( 'TGO_SA-Z_C1'                 )
      NAIF_BODY_CODE += ( -143071                       )

      NAIF_BODY_NAME += ( 'TGO_SA-Z_C2'                 )
      NAIF_BODY_CODE += ( -143072                       )

      NAIF_BODY_NAME += ( 'TGO_SA-Z_C3'                 )
      NAIF_BODY_CODE += ( -143073                       )

      NAIF_BODY_NAME += ( 'TGO_SA-Z_C4'                 )
      NAIF_BODY_CODE += ( -143074                       )


      NAIF_BODY_NAME += ( 'TGO_ACS'                     )
      NAIF_BODY_CODE += ( -143100                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_NIR_NAD'             )
      NAIF_BODY_CODE += ( -143111                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_NIR_OCC'             )
      NAIF_BODY_CODE += ( -143112                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_MIR'                 )
      NAIF_BODY_CODE += ( -143120                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM'              )
      NAIF_BODY_CODE += ( -143130                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM_SCAN_BBY'     )
      NAIF_BODY_CODE += ( -143131                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM_SCAN_SPC'     )
      NAIF_BODY_CODE += ( -143132                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM_SCAN_NAD'     )
      NAIF_BODY_CODE += ( -143133                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM_SCAN_OCC'     )
      NAIF_BODY_CODE += ( -143134                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM_SCAN_OCC_BS'  )
      NAIF_BODY_CODE += ( -143135                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM_SUN'          )
      NAIF_BODY_CODE += ( -143140                       )

      NAIF_BODY_NAME += ( 'TGO_ACS_TIRVIM_SUN_BSR'      )
      NAIF_BODY_CODE += ( -143141                       )


      NAIF_BODY_NAME += ( 'TGO_FREND'                   )
      NAIF_BODY_CODE += ( -143200                       )

      NAIF_BODY_NAME += ( 'TGO_FREND_HE'                )
      NAIF_BODY_CODE += ( -143210                       )

      NAIF_BODY_NAME += ( 'TGO_FREND_SC'                )
      NAIF_BODY_CODE += ( -143220                       )


      NAIF_BODY_NAME += ( 'TGO_NOMAD'                   )
      NAIF_BODY_CODE += ( -143300                       )

      NAIF_BODY_NAME += ( 'TGO_NOMAD_LNO'               )
      NAIF_BODY_CODE += ( -143310                       )

      NAIF_BODY_NAME += ( 'TGO_NOMAD_LNO_OPS_NAD'       )
      NAIF_BODY_CODE += ( -143311                       )

      NAIF_BODY_NAME += ( 'TGO_NOMAD_LNO_OPS_OCC'       )
      NAIF_BODY_CODE += ( -143312                       )

      NAIF_BODY_NAME += ( 'TGO_NOMAD_SO'                )
      NAIF_BODY_CODE += ( -143320                       )

      NAIF_BODY_NAME += ( 'TGO_NOMAD_UVIS_NAD'          )
      NAIF_BODY_CODE += ( -143331                       )

      NAIF_BODY_NAME += ( 'TGO_NOMAD_UVIS_OCC'          )
      NAIF_BODY_CODE += ( -143332                       )


      NAIF_BODY_NAME += ( 'TGO_CASSIS'                  )
      NAIF_BODY_CODE += ( -143400                       )

      NAIF_BODY_NAME += ( 'TGO_CASSIS_PAN'              )
      NAIF_BODY_CODE += ( -143421                       )

      NAIF_BODY_NAME += ( 'TGO_CASSIS_RED'              )
      NAIF_BODY_CODE += ( -143422                       )

      NAIF_BODY_NAME += ( 'TGO_CASSIS_NIR'              )
      NAIF_BODY_CODE += ( -143423                       )

      NAIF_BODY_NAME += ( 'TGO_CASSIS_BLU'              )
      NAIF_BODY_CODE += ( -143424                       )

   \begintext


End of FK file.