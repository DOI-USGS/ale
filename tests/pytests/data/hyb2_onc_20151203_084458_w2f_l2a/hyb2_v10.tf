KPL/FK

Hayabusa2 Frames Definition Kernel
===========================================================================

   This frame kernel contains a set of frame definitions for the
   Hayabusa2 mission. It also contains NAIF name-ID mappings for
   Hayabusa2 instruments.


Version and Date
---------------------------------------------------------------------------

   Version 1.0 -- Mar. 29, 2018 -- Shin-ya MURAKAMI, ISAS/JAXA
        - Updated references
        - Corrected name corresponding to NAIF ID
        - Updated with some cosmetic changes
        - Fixed some typos

   Version 0.9 -- June 05, 2017 -- Yukio YAMAMOTO, ISAS/JAXA
        - Included the NAIF code definition of Asteroid Ryugu

   Version 0.8 -- May 08, 2017 -- Yukio YAMAMOTO, ISAS/JAXA
        - Updated the ONC alignment as a result of the ONC team report [1]

   Version 0.7 -- Mar. 09, 2017 -- Yukio YAMAMOTO, ISAS/JAXA
        - Updated TIR alignment by the TIR team

   Version 0.6 -- Jan. 16, 2017 -- Yukio YAMAMOTO, ISAS/JAXA
        - Fixed NIRS3 alignment as a result of the NIRS3 experiment

   Version 0.5 -- Nov. 18, 2016 -- Yukio YAMAMOTO, ISAS/JAXA
        - Improved LIDAR alignment as a result of the LIDAR experiment

   Version 0.4 -- Oct. 14, 2016 -- Yukio YAMAMOTO, ISAS/JAXA
        - Re-assigned version number
        - Renamed prefix from hay2 to hyb2
        - Fixed OME NAIF ID

   Version 0.3 -- Sep. 4, 2015 -- Yukio YAMAMOTO, ISAS/JAXA
        - Fixed typo

   Version 0.2 -- Jul. 28, 2015 -- Yukio YAMAMOTO, ISAS/JAXA
        - Added TIR definition
        - Fixed NAIF code in description

   Version 0.1 -- Nov. 18, 2014 -- Manabu YAMADA, PERC/Chitech
        - changed the NAIF ID (999 -> 37)
        - changed Instruments IDs
        - updated ASCII art of Spacecraft

   Version 0.0 -- Feb. 14, 2013 --


References
---------------------------------------------------------------------------

   [1] Suzuki, H. et al. (2018), Initial inflight calibration for
       Hayabusa2 optical navigation camera (ONC) for science
       observations of asteroid Ryugu, Icarus, 300, 341 -- 359.


Hayabusa2 NAIF IDs
---------------------------------------------------------------------------

   The following names and NAIF ID codes are assigned to the Hayabusa2
   spacecraft, its structures, and science instruments:

      NAME                                    NAIF ID
      -----------------------------------------------
      HAYABUSA2                                   -37
      HAYABUSA2_SC_BUS_PRIME                   -37000
      HAYABUSA2_ONC                            -37100
      HAYABUSA2_TIR-S                          -37200
      HAYABUSA2_NIRS3-S                        -37300
      HAYABUSA2_LIDAR                          -37400
      HAYABUSA2_DCAM3                          -37500
      HAYABUSA2_SCI                            -37600
      (reserved)                               -37700
      HAYABUSA2_OME                            -37800
      HAYABUSA2_MINERVA2                       -37810
      HAYABUSA2_TARGET_MARKER                  -37820
      HAYABUSA2_MASCOT                         -37830
      HAYABUSA2_CAPSULE                        -37840
      HAYABUSA2_LOCATION                       -37850

      RYUGU                                   2162173
      1999JU3                                 2162173
                               

   The keywords below implement the Hayabusa2 name-ID mappings.

      \begindata

         NAIF_BODY_NAME += ( 'HAYABUSA2' )
         NAIF_BODY_CODE += ( -37 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_SC_BUS_PRIME' )
         NAIF_BODY_CODE += ( -37000 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_ONC-T' )
         NAIF_BODY_CODE += ( -37100 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_ONC-W1' )
         NAIF_BODY_CODE += ( -37110 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_ONC-W2' )
         NAIF_BODY_CODE += ( -37120 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_TIR-S' )
         NAIF_BODY_CODE += ( -37200 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_NIRS3-S' )
         NAIF_BODY_CODE += ( -37300 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_LIDAR' )
         NAIF_BODY_CODE += ( -37400 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_DCAM3' )
         NAIF_BODY_CODE += ( -37500 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_SCI' )
         NAIF_BODY_CODE += ( -37600 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_OME' )
         NAIF_BODY_CODE += ( -37800 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_MINERVA2' )
         NAIF_BODY_CODE += ( -37810 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_TARGET_MARKER' )
         NAIF_BODY_CODE += ( -37820 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_MASCOT' )
         NAIF_BODY_CODE += ( -37830 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_CAPUSULE' )
         NAIF_BODY_CODE += ( -37840 )

         NAIF_BODY_NAME += ( 'HAYABUSA2_LOCATION' )
         NAIF_BODY_CODE += ( -37900 )

         NAIF_BODY_NAME += ( 'RYUGU' )
         NAIF_BODY_CODE += ( 2162173 )

         NAIF_BODY_NAME += ( '1999JU3' )
         NAIF_BODY_CODE += ( 2162173 )


      \begintext


Hayabusa2 Reference Frames
---------------------------------------------------------------------------

   The following reference frames for Hayabusa2 spacecraft, its
   structures, and science instruments are defined in this FK and other
   Hayabusa2 kernels (162173 1999 JU3 FK, PCKs, and IKs):


   162173 1999 JU3 body-fixed frame (defined in a separate FK and PCKs):
   ---------------------------------------------------------------------

      Name                    Relative to             Type           Frame ID
      ======================  ======================  ============   ========
      1999_JU3_FIXED          J2000                   PCK              162173


   Hayabusa2 spacecraft frame (duplicated in the IKs):
   ---------------------------------------------------

      Name                    Relative to             Type           Frame ID
      ======================  ======================  ============   ========
      HAYABUSA2_SC_BUS_PRIME  J2000                   CK               -37000


   Hayabusa2 science instrument frames (defined in the IK files):
   ------------------------------------------------------------------------

      Name                    Relative to             Type           Frame ID
      ======================  ======================  ============   ========
      HAYABUSA2_ONC-T         HAYABUSA2_SC_BUS_PRIME  FIXED            -37100

      HAYABUSA2_ONC-W1        HAYABUSA2_SC_BUS_PRIME  FIXED            -37110

      HAYABUSA2_ONC-W2        HAYABUSA2_SC_BUS_PRIME  FIXED            -37120

      HAYABUSA2_TIR-S         HAYABUSA2_SC_BUS_PRIME  FIXED            -37200

      HAYABUSA2_NIRS3-S       HAYABUSA2_SC_BUS_PRIME  FIXED            -37300

      HAYABUSA2_LIDAR         HAYABUSA2_SC_BUS_PRIME  FIXED            -37400

      HAYABUSA2_DCAM3         HAYABUSA2_SC_BUS_PRIME  FIXED            -37500

      HAYABUSA2_SCI           HAYABUSA2_SC_BUS_PRIME  FIXED            -37600

      HAYABUSA2_OME-A         HAYABUSA2_SC_BUS_PRIME  FIXED            -37800


Hayabusa2 Reference Frame Tree
---------------------------------------------------------------------------

   This diagram shows the Hayabusa2 frame tree:


                               "J2000" INERTIAL
              +--------------------------------------------+
              |                                            |
              |<-ck                                        |<-pck
              |                                            |
              |                                            |
              |                                            V
              |                                      "1999_JU3_FIXED"
              |                                      ----------------
              |
              |
              |
              |       "HAYABUSA_ONC-W2"  "HAYABUSA2_OME-A"  "HAYABUSA_TIR-S"
              |       -----------------  -----------------  ----------------
              |                ^                 ^                 ^
              |                |                 |                 |
              V                |<-fixed          |<-fixed          |<-fixed
    "HAYABUSA2_SC_BUS_PRIME"   |                 |                 |
              +----------------------------------------------------+
              |                |                 |                 |
              |<-fixed         |<-fixed          |<-fixed          |<-fixed
              |                |                 |                 |
              V                V                 V                 V
   "HAYABUSA2_LIDAR"  "HAYABUSA2_NIRS3-S"  "HAYABUSA2_ONC-T"  "HAYABUSA2_ONC-W1"
   -----------------  -------------------  -----------------  ------------------


1999_JU3 Body-Fixed Frame
---------------------------------------------------------------------------

   [TBD]

   The 1999_JU3 body-fixed frame, 1999_JU3_FIXED, is defined in a
   separate FK file (1999_JU3_fixed.tf). Its definition is also
   duplicated in some of the 1999_JU3 PCK files.


Hayabusa2 Spacecraft Frame
---------------------------------------------------------------------------

   The Hayabusa2 spacecraft frame, HAYABUSA2_SC_BUS_PRIME, is defined 
   as follows:

      - +Z axis is parallel to the nominal HGA boresight;

      - +X axis is parallel to the ion engines thrust vector;

      - +Y axis completes the right-handed frame;

   The Hayabusa2 spacecraft frame shown on this diagram:


                                     ^
                                     | HGA Boresight
                                     |

                                   X-band     X-band
                                    HGA        MGA
                                                 _
                                _____________  {@ }
                      _  .______Y___________Y___[]_.
              STT-B    \H|                         |
                      _/H|         ________        |
                         |        |        |       |_
                         |        |________|       | |
                         |                         | |E  Ion Engines
                        _|_                        | |
                       /|/_\ ONC-W2                | |     ------->
                      | ||                         | |E    Thrust Vector
                       \||                         |_|
                        -._________________________.
                          |####||___________|  W TT
                          |#|| |        ||
                           | LIDAR     OME-A  ONC-W1
                           | |                   ONC-T
                       SMP | |                   |
                           | |                   | +Zonc_w1,t
                          /___\                  V
            +Zsc
             ^
             |
             |
             x-----> +Xsc
               +Ysc is into the page



   This frame is defined as a CK-based frame below. The frame
   definition is also duplicated in Hayabusa2 science instrument IKs.

      \begindata


         FRAME_HAYABUSA2_SC_BUS_PRIME = -37000
         FRAME_-37000_NAME            = 'HAYABUSA2_SC_BUS_PRIME'
         FRAME_-37000_CLASS           = 3
         FRAME_-37000_CLASS_ID        = -37000
         FRAME_-37000_CENTER          = '-37'
         CK_-37000_SCLK               = '-37'
         CK_-37000_SPK                = '-37'


      \begintext


Hayabusa2 Science Instrument Frames
---------------------------------------------------------------------------

   This section contains frame definitions for Hayabusa2 science instruments.

   The instrument frames are defined as fixed offset frames with their
   orientation specified using Euler angles.

   Note that angles in the frame definitions are specified for "from 
   instrument to base (relative to ) frame" transformation.


  LIDAR Frame definitions
  ---------------------------------------------------------------------------

   -Y s/c side view:
   -----------------
                                     ^
                                     | HGA Boresight
                                     |

                                   X-band     X-band
                                    HGA        MGA
                                                 _
                                _____________  {@ }
                      _  .______Y___________Y___[]_.
              STT-B    \H|                         |
                      _/H|         ________        |
                         |        |        |       |_
                         |        |________|       | |
                         |                         | |E  Ion Engines
                        _|_                        | |
                       /|/_\ ONC-W2                | |     ------->
                      | ||                         | |E    Thrust Vector
                       \||                         |_|
                        -._________________________.
                          |####||___________|  W TT
                          |#|| |        ||
                           | LIDAR     OME-A
                           | |
                       SMP | | |
                           | | | +Zlid
                          /___\|
            +Zsc               |
             ^                 V
             |
             |
             x-----> +Xsc
               +Ysc is into the page

\begindata


   FRAME_HAYABUSA2_LIDAR   = -37400
   FRAME_-37400_NAME       = 'HAYABUSA2_LIDAR'
   FRAME_-37400_CLASS      = 4
   FRAME_-37400_CLASS_ID   = -37400
   FRAME_-37400_CENTER     = -37
   TKFRAME_-37400_SPEC     = 'ANGLES'
   TKFRAME_-37400_RELATIVE = 'HAYABUSA2_SC_BUS_PRIME'
   TKFRAME_-37400_ANGLES   = (    0.0000000000000000, 
                               -179.7908699380701023, 
                                 -0.0085943669591912 )
   TKFRAME_-37400_AXES     = ( 3, 2, 1 )
   TKFRAME_-37400_UNITS    = 'DEGREES'


\begintext

  ONC-T and ONC-W1 Frame definitions
  -------------------------------------------------------------------------

   -Y s/c side view:
   -----------------
                                     ^
                                     | HGA Boresight
                                     |

                                   X-band     X-band
                                    HGA        MGA
                                                 _
                                _____________  {@ }
                      _  .______Y___________Y___[]_.
              STT-B    \H|                         |
                      _/H|         ________        |
                         |        |        |       |_
                         |        |________|       | |
                         |                         | |E  Ion Engines
                        _|_                        | |
                       /|/_\ ONC-W2                | |     ------->
                      | ||                         | |E    Thrust Vector
                       \||                         |_|
                        -._________________________.
                          |####||___________|  W TT
                          |#|| |        ||
                           | LIDAR     OME-A  ONC-W1
                           | |                   ONC-T
                       SMP | |                   |
                           | |                   | +Zonc_w1,t
                          /___\                  V
            +Zsc
             ^
             |
             |
             x-----> +Xsc
               +Ysc is into the page

\begindata


   FRAME_HAYABUSA2_ONC-T   = -37100
   FRAME_-37100_NAME       = 'HAYABUSA2_ONC-T'
   FRAME_-37100_CLASS      = 4
   FRAME_-37100_CLASS_ID   = -37100
   FRAME_-37100_CENTER     = -37
   TKFRAME_-37100_SPEC     = 'ANGLES'
   TKFRAME_-37100_RELATIVE = 'HAYABUSA2_SC_BUS_PRIME'
   TKFRAME_-37100_ANGLES   = (    0.0000000, -179.9292340,    0.1365657 )
   TKFRAME_-37100_AXES     = ( 3, 2, 1 )
   TKFRAME_-37100_UNITS    = 'DEGREES'


   FRAME_HAYABUSA2_ONC-W1  = -37110
   FRAME_-37110_NAME       = 'HAYABUSA2_ONC-W1'
   FRAME_-37110_CLASS      = 4
   FRAME_-37110_CLASS_ID   = -37110
   FRAME_-37110_CENTER     = -37
   TKFRAME_-37110_SPEC     = 'ANGLES'
   TKFRAME_-37110_RELATIVE = 'HAYABUSA2_SC_BUS_PRIME'
   TKFRAME_-37110_ANGLES   = (    0.0000000, -180.2650038,    0.2291904 )
   TKFRAME_-37110_AXES     = ( 3, 2, 1 )
   TKFRAME_-37110_UNITS    = 'DEGREES'


\begintext

  NIRS3-S and ONC-W2 Frame definitions
  -------------------------------------------------------------------------

  The ONC-W2 is embedded in Hayabusa2 spacecraft and this boresight vector is
  rotated by -59 degrees about Science Instrument Boresights.

   -X s/c side view:
   -----------------
                                     ^
                                     | HGA Boresight
                                     |

                           Ka-band   X-band    X-band
                            HGA       MGA       HGA

                           ___________>@___________
                               Y______[=]______Y
                               |    / \ / \    |
 ===============>-------------o|    \_/ \_/    |o-------------<===============
                               | STT-A  STT-B  | /
                               |               |/
                               |               |
                               |      ___      |
                               |__   /   \     |.....................
                       NIRS3-S /  | |     |    |-.     .'
                               |  |  \___/     |  `-..'  31 deg
                               \__|________####.     `-.
                                 |___|   |_####         `-.
                                     |___|  || LIDAR       `-.
                                |     | |                     `-. +Z ONC-W2 
                                |     | | SMP                    `->
                    -Z NIRS3    V     | |
                                      | |
                                     /___\            +Xsc is into the page

           +Zsc
             ^
             |
             |
 +Ysc  <-----x


\begindata

   FRAME_HAYABUSA2_ONC-W2  = -37120
   FRAME_-37120_NAME       = 'HAYABUSA2_ONC-W2'
   FRAME_-37120_CLASS      = 4
   FRAME_-37120_CLASS_ID   = -37120
   FRAME_-37120_CENTER     = -37
   TKFRAME_-37120_SPEC     = 'ANGLES'
   TKFRAME_-37120_RELATIVE = 'HAYABUSA2_SC_BUS_PRIME'
   TKFRAME_-37120_ANGLES   = ( -270, -121,    0 )
   TKFRAME_-37120_AXES     = ( 3, 2, 1 )
   TKFRAME_-37120_UNITS    = 'DEGREES'
   

   FRAME_HAYABUSA2_NIRS3-S = -37300
   FRAME_-37300_NAME       = 'HAYABUSA2_NIRS3-S'
   FRAME_-37300_CLASS      = 4
   FRAME_-37300_CLASS_ID   = -37300
   FRAME_-37300_CENTER     = -37
   TKFRAME_-37300_SPEC     = 'ANGLES'
   TKFRAME_-37300_RELATIVE = 'HAYABUSA2_SC_BUS_PRIME'
   TKFRAME_-37300_ANGLES   = (    0.000, 0.225, 0.160 )
   TKFRAME_-37300_AXES     = ( 3, 2, 1 )
   TKFRAME_-37300_UNITS    = 'DEGREES'
   
   
\begintext

  OME-A nd TIR-S Frame definitions
  -------------------------------------------------------------------------

   +Y s/c side view:
   -----------------
                                     ^
                                     | HGA Boresight
                                     |

                         X-band   Ka-band
                          MGA       HGA
                           _
                          { @}  _____________
                         ._[]___Y___________Y______.  _
                         |                         |H/    STT-A
                         |                         |H\_
                        _|                         |
                       | |                         |
     Ion Engines      E| |                         |
                       | |                         |_
         <-------      | |                         ||\
    Thrust Vector     E| |                         |--.
                       |_|                         |  |
                         ._________________________.__| NIRS3-S
                          A     |___________||    |
                    TIR-S          ||        |____|  |
                          |      OME-A         | |   |
                          |                    | |   |
                     +Ztir|    +Zome|      SMP | |   V -Znirs3
                          V         |          | |
                                    V         /___\
            +Zsc
             ^
             |
             |
  +Xsc <-----o
             +Ysc is out of the page

\begindata

   FRAME_HAYABUSA2_TIR-S   = -37200
   FRAME_-37200_NAME       = 'HAYABUSA2_TIR-S'
   FRAME_-37200_CLASS      = 4
   FRAME_-37200_CLASS_ID   = -37200
   FRAME_-37200_CENTER     = -37
   TKFRAME_-37200_SPEC     = 'ANGLES'
   TKFRAME_-37200_RELATIVE = 'HAYABUSA2_SC_BUS_PRIME'
   TKFRAME_-37200_ANGLES   = ( -1.077008, -180.080079, 0.140688 )
   TKFRAME_-37200_AXES     = ( 3, 2, 1 )
   TKFRAME_-37200_UNITS    = 'DEGREES'

   FRAME_HAYABUSA2_OME-A   = -37800
   FRAME_-37800_NAME       = 'HAYABUSA2_OME-A'
   FRAME_-37800_CLASS      = 4
   FRAME_-37800_CLASS_ID   = -37800
   FRAME_-37800_CENTER     = -37
   TKFRAME_-37800_SPEC     = 'ANGLES'
   TKFRAME_-37800_RELATIVE = 'HAYABUSA2_SC_BUS_PRIME'
   TKFRAME_-37800_ANGLES   = (    0, -180,    0 )
   TKFRAME_-37800_AXES     = ( 3, 2, 1 )
   TKFRAME_-37800_UNITS    = 'DEGREES'

\begintext

End of FK file.
