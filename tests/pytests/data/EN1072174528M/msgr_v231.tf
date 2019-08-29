KPL/FK

MESSENGER Spacecraft Frame Definitions Kernel
===========================================================================

   This frame kernel contains the MESSENGER spacecraft, science instrument,
   and communication antennae frame definitions.


Version and Date
---------------------------------------------------------------

   The TEXT_KERNEL_ID stores version information of loaded project text
   kernels.  Each entry associated with the keyword is a string that
   consists of four parts: the kernel name, version, entry date, and type.
   For example, the frame kernel might have an entry as follows:

      TEXT_KERNEL_ID += 'MESSENGER_FRAMES V2.1.0 07-MAR-2012 FK'

   MESSENGER Frame Kernel Version:

      \begindata

      TEXT_KERNEL_ID += 'MESSENGER_FRAMES V2.3.1 15-OCT-2013 FK'

      \begintext

   Version 2.3.1 -- October 15, 2013 -- Scott Turner & Mike Reid
      Added commentary to clarify that this kernel identifies only the
      spacecraft frame and that two coordinate system origins have been used.

   Version 2.3.0 -- July 19, 2013 -- Scott Turner & Grant Stephens

      Introduced an offset to the NAC/WAC coalignment matrix that is the
      result of analysis of star calibration frames acquired to date since
      MOI.  This frame kernel version should be utilized with the 1.4.0 
      (or later) version of the MDIS IK.

      An additional CK based frame was added to the MDIS alignment chain
      between MSGR_MDIS_BASE and MSGR_MDIS_ART.  This frame, MSGR_MDIS_ART_CAL
      captures an empirical model of the temporal drift of the mounting 
      alignment measured from stellar calibration images.
      
   Version 2.2.0 -- May 21, 2012 -- Grant Stephens

      Updated the MDIS WAC base mounting to reflect a shift that took 
      place around Mercury orbit insertion.

   Version 2.1.0 -- October 13, 2010 -- Lillian Nguyen & Scott Turner

      Updated the MDIS NAC to WAC and the MDIS base to spacecraft alignment 
      matrices.
      Reserved a range of frame ID codes not to be assigned in this file
      (reserved for user applications).

   Version 2.0.0 -- Sept 11, 2008 -- Lillian Nguyen

      Added a C-kernel based frame for the nonlinear harmonic drive
      model of the MDIS pivot.

   Version 1.0.0 -- June 10, 2008 -- Lillian Nguyen

      Added frames for the phased array and fan beam antennas.

   Version 0.9.1 -- February 4, 2008 -- Lillian Nguyen

      Addition of text describing the principal axes of the solar panel frames.

   Version 0.9 -- August 9, 2007 -- Lillian Nguyen

      Corrected the MDIS Base, WAC, and NAC frames with updated alignment
      matrices.

   Version 0.8 -- July 24, 2007  -- Lillian Nguyen

      Updated the MASCS frame definitions with new boresights.
      Redefined the MASCS frames such that the MASCS coordinate systems
      approximately match the spacecraft coordinate system.

   Version 0.7 -- April 12, 2007  -- Lillian Nguyen

      Updated the MDIS frame definitions with in-flight alignment values.

   Version 0.6 -- April 10, 2007  -- Lillian Nguyen

      Updated the MLA frame definition with new boresight.
      Increased the FIPS ID code to accomodate EPPS SSD definitions.
      Made minor text changes and formatting edits.

   Version 0.5 -- March 6, 2007 -- Lillian Nguyen

      Updated the MAG frames with corrected entries.

   Version 0.4 -- July 22, 2005 -- Scott Turner

      Kernel updates in preparation for the Earth flyby.  This includes
      some documentation clean up and clarification, as well as the
      addition of the MDIS WAC filter frames.

   Version 0.3 -- June 17, 2005 -- Scott Turner

      Updated MDIS frame definitions to prepare for initial release
      of gimbal and attitude history C-kernel creation tools.  This
      includes all nominal alignments of the gimbal axis and detector
      boresights.

   Version 0.2 -- September 9, 2004 -- Scott Turner

      Added instrument frame content for the following instruments:
      GRNS, XRS, MAG, MLA, MASCS, EPPS, and MDIS.

   Version 0.1 -- September 2, 2004 -- Scott Turner

      Updated and released for preliminary operational usage for
      C-kernel creation and testing.

   Version 0.0 -- June 30, 2004 -- Scott Turner

      Initial prototype release to support C-kernel creation and
      utilization.


References
---------------------------------------------------------------

      1.   "C-kernel Required Reading"

      2.   "Kernel Pool Required Reading"

      3.   "Frames Required Reading"

      4.   "MESSENGER G&C Coordinate Systems, Component Alignments
           and Naming Conventions (Revision C)", R. M. Vaughan

      5.   MESSENGER project website, http://messenger.jhuapl.edu.

      6.   Email from Ed Rhodes regarding GRNS coordinate systems
           and field of view specifications.

      7.   Email from George Ho regarding XRS fields of view.

      8.   Email from Brian Anderson regarding MAG alignment
           information.

      9.   Email from Brian Anderson regarding updates to the nominal
           MAG alignment.

     10.   An email from Gene A. Heyler regarding MLA alignment forwarded
           by Gregory Neumann.

     11.   "MLA Coordinate System and Survey Results", Xiaoli Sun
           and Luis Ramos-Izquierdo (GSFC), 3/11/2004.

     12.   Email from William McClintock containing an update to the
           original MASCS boresight memo.

     13.   Email from Barry Mauk regarding EPPS fields of view and
           alignment.

     14.   Notes from a discussion with Ed Hawkins regarding MDIS
           articulation, alignment, and field of view orientation.

     15.   "The MESSENGER Mercury Dual Imaging System (MDIS):
           Design, Observing Plan, and Calibration", Ed Hawkins, et.al.

     16.   E-mail from Olivier Barnouin-Jha, received 9/14/2006.

     17.   E-mail from Haje Korth regarding conflicting transformation
           matrices for the MAG stowed orientation.

     18.   "MESSENGER Mercury Laser Altimeter (MLA) Calibration Report",
           JHU/APL drawing number 7384-9465.

     19.   E-mail from Gregory Neumann forwarded by Scott Turner on
           12/14/2006 regarding MLA receiver and transmitter calibration.

     20.   E-mail from Haje Korth with corrections to the transformation
           matrix for the stowed MAG position and Y-Z rotation angle,
           dated 3/5/2007.

     21.   Excerpt from an MDIS alignment report received from Scott Turner
           by e-mail on 4/10/2007. (full report and SD/SIS memo number
           pending).

     22.   E-mail from Robin Vaughan containing the MASCS boresights,
           received 7/4/2007, and e-mail exchange with Noam Izenberg,
           7/9/2007-7/18/2007.

     23.   E-mail received from Scott Turner on 8/9/2007 containing a
           correction to the MDIS alignment matrices.

     24.   E-mail from Scott Turner received on 1/15/2008.

     25.   Document excerpts containing G&C antenna coordinates, received
           by e-mail from Dan O'Shaughnessy on 5/14/2008.

     26.   Notes from discussions with Scott Turner concerning the
           MDIS pivot calibration, 8/2008.
           
     27.   Discussion with Scott Turner regarding reserved ID codes,
           4/27/2010
           
     28.   JHU/APL memo SIS-10-012 titled 'Updated Alignment Determination 
           for MESSENGER/MDIS'. 

     29.   JHU/APL memo SIS-12-007 titled 'Results from Initial Thermal 
           Calibration of MESSENGER/MDIS'

     30.   Pending JHU/APL memo containing updates to the MDIS alignment
           and optical distortion parameters.

Contact Information
---------------------------------------------------------------

   Direct questions, comments, or concerns about the contents of this kernel
   to:

      Lillian Nguyen, JHUAPL/SIS, (443)778-5477, Lillian.Nguyen@jhuapl.edu

      or

      Scott Turner, JHUAPL/SIS, (443)778-1693, Scott.Turner@jhuapl.edu


Implementation Notes
---------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make use
   of this frame kernel must "load" the kernel normally during program
   initialization.  Loading the kernel associates the data items with
   their names in a data structure called the "kernel pool".  The SPICELIB
   routine FURNSH loads a kernel into the pool as shown below:

      FORTRAN: (SPICELIB)

         CALL FURNSH ( frame_kernel_name )

      C: (CSPICE)

         furnsh_c ( frame_kernel_name );

      IDL: (ICY)

         cspice_furnsh, frame_kernel_name

   In order for a program or routine to extract data from the pool, the
   SPICELIB routines GDPOOL, GIPOOL, and GCPOOL are used.  See [2] for
   more details.

   This file was created and may be updated with a text editor or word
   processor.


MESSENGER Frames
---------------------------------------------------------------

   The ID codes -236050 to -236099 have been reserved to support user
   functions and are not utilized in this file or the MESSENGER dynamics 
   frames kernel [27].
   
   The following MESSENGER frames are defined in this kernel file:

      Frame Name                Relative To              Type      NAIF ID
      =======================   ===================      =======   =======

      Spacecraft Frames:
      ------------------
      MSGR_SPACECRAFT           J2000                     CK       -236000
      MSGR_SOLARPANEL_PLUS      MSGR_SPACECRAFT(*)        CK       -236001
      MSGR_SOLARPANEL_MINUS     MSGR_SPACECRAFT(*)        CK       -236002

      Antenna Frames:
      ---------------
      MSGR_RS_PAA_FRONT         MSGR_SPACECRAFT           FIXED    -236150
      MSGR_RS_PAA_REAR          MSGR_RS_PAA_FRONT         FIXED    -236160
      MSGR_RS_FB_FRONT          MSGR_RS_PAA_FRONT         FIXED    -236170
      MSGR_RS_FB_REAR           MSGR_RS_PAA_FRONT         FIXED    -236180

      GRNS Frames:
      ------------
      MSGR_GRNS_GRS             MSGR_SPACECRAFT           FIXED    -236200
      MSGR_GRNS_NS              MSGR_SPACECRAFT           FIXED    -236210

      XRS Frames:
      -----------
      MSGR_XRS_MXU              MSGR_SPACECRAFT           FIXED    -236300
      MSGR_XRS_SAX              MSGR_SPACECRAFT           FIXED    -236310

      MAG Frames:
      -----------
      MSGR_MAG                  MSGR_SPACECRAFT           FIXED    -236400
      MSGR_MAG_STOWED           MSGR_SPACECRAFT           FIXED    -236410
      MSGR_MAG_HALFWAY          MSGR_SPACECRAFT           FIXED    -236420

      MLA Frames:
      -----------
      MSGR_MLA                  MSGR_SPACECRAFT           FIXED    -236500
      MSGR_MLA_RECEIVER         MSGR_MLA                  FIXED    -236501

      MASCS Frames:
      -------------
      MSGR_MASCS_UVVS_ATMO      MSGR_SPACECRAFT           FIXED    -236600
      MSGR_MASCS_UVVS_SURF      MSGR_SPACECRAFT           FIXED    -236610
      MSGR_MASCS_VIRS           MSGR_SPACECRAFT           FIXED    -236620

      EPPS Frames:
      ------------
      MSGR_EPPS_EPS             MSGR_SPACECRAFT           FIXED    -236700
      MSGR_EPPS_FIPS            MSGR_SPACECRAFT           FIXED    -236720

      MDIS Frames:
      ------------
      MSGR_MDIS_BASE            MSGR_SPACECRAFT           FIXED    -236880
      MSGR_MDIS_ART             MGSR_MDIS_BASE            CK       -236890
                                or MSGR_MDIS_ART_RAW, for calibrated pivot
                                or MSGR_MDIS_ART_CAL, for drift model
      MSGR_MDIS_ART_RAW         MGSR_MDIS_BASE            CK       -236891
      MSGR_MDIS_ART_CAL         MSGR_MDIS_BASE            CK       -236892
      MSGR_MDIS_WAC             MSGR_MDIS_ART             FIXED    -236800
      MSGR_MDIS_WAC_FILTER1     MSGR_MDIS_WAC             FIXED    -236801
      MSGR_MDIS_WAC_FILTER2     MSGR_MDIS_WAC             FIXED    -236802
      MSGR_MDIS_WAC_FILTER3     MSGR_MDIS_WAC             FIXED    -236803
      MSGR_MDIS_WAC_FILTER4     MSGR_MDIS_WAC             FIXED    -236804
      MSGR_MDIS_WAC_FILTER5     MSGR_MDIS_WAC             FIXED    -236805
      MSGR_MDIS_WAC_FILTER6     MSGR_MDIS_WAC             FIXED    -236806
      MSGR_MDIS_WAC_FILTER7     MSGR_MDIS_WAC             FIXED    -236807
      MSGR_MDIS_WAC_FILTER8     MSGR_MDIS_WAC             FIXED    -236808
      MSGR_MDIS_WAC_FILTER9     MSGR_MDIS_WAC             FIXED    -236809
      MSGR_MDIS_WAC_FILTER10    MSGR_MDIS_WAC             FIXED    -236810
      MSGR_MDIS_WAC_FILTER11    MSGR_MDIS_WAC             FIXED    -236811
      MSGR_MDIS_WAC_FILTER12    MSGR_MDIS_WAC             FIXED    -236812
      MSGR_MDIS_NAC             MSGR_MDIS_WAC             FIXED    -236820

   Notes:

     (*) The solar panel frames are provided via C-kernels to account
         for their articulation.  Typically the spacecraft and solar
         panel attitude are provided in a single C-kernel, where the
         spacecraft frame is relative to J2000 and the solar panel
         frames are relative to the spacecraft body frame.  However,
         some analysis tools may require the solar panel frame related
         directly to the inertial, J2000, frame.  See the comments of
         the C-kernel in question for details.

     (+) No information regarding communication antennae has been provided
         for inclusion into the frame kernel as of the release date.


MESSENGER Frames Hierarchy
---------------------------------------------------------------

   This diagram is subject to major revisions as this kernel evolves
   to suit the needs of each instrument.

   The articulating frames referenced in this kernel are not required
   to follow the paths outlined in the hierarchy below, however; the
   presentation indicates the expected, or nominal, production path.

   The diagram below illustrates the MESSENGER frame hierarchy:

       'IAU_MERCURY' (MERCURY BODY FIXED)
          |
          |<--- pck
          |
       'J2000'
          |
          |<--- ck
          |
         'MSGR_SPACECRAFT'
              |
              |<--- ck
              |
             'MSGR_SOLARPANEL_PLUS'
              |
              |<--- ck
              |
             'MSGR_SOLARPANEL_MINUS'
              |
             'MSGR_GRNS_GRS'
              |
             'MSGR_GRNS_NS'
              |
             'MSGR_XRS_MXU'
              |
             'MSGR_XRS_SAX'
              |
             'MSGR_MAG'
              |
             'MSGR_MAG_STOWED'
              |
             'MSGR_MAG_HALFWAY'
              |
             'MSGR_MLA
              |   |
              |  'MGSR_MLA_RECEIVER'
              |
             'MSGR_MASCS_UVVS_ATMO'
              |
             'MSGR_MASCS_UVVS_SURF'
              |
             'MSGR_MASCS_VIRS'
              |
             'MSGR_EPPS_EPS'
              |
             'MSGR_EPPS_FIPS'
              |
             'MSGR_MDIS_BASE'
                  |
                  |<--- ck
                  |
                 'MGSR_MDIS_ART_RAW' (exists ONLY for calibrated pivot) 
                 'MSGR_MDIS_ART_CAL' (compensates for temporal drift)
                  |
                  |<--- ck
                  |
                 'MGSR_MDIS_ART'
                      |
                     'MSGR_MDIS_WAC'
                          |
                         'MSGR_MDIS_NAC'
                          |
                         'MSGR_MDIS_WAC_FILTER1'
                          |
                         'MSGR_MDIS_WAC_FILTER2'
                          |
                         'MSGR_MDIS_WAC_FILTER3'
                          |
                         'MSGR_MDIS_WAC_FILTER4'
                          |
                         'MSGR_MDIS_WAC_FILTER5'
                          |
                         'MSGR_MDIS_WAC_FILTER6'
                          |
                         'MSGR_MDIS_WAC_FILTER7'
                          |
                         'MSGR_MDIS_WAC_FILTER8'
                          |
                         'MSGR_MDIS_WAC_FILTER9'
                          |
                         'MSGR_MDIS_WAC_FILTER10'
                          |
                         'MSGR_MDIS_WAC_FILTER11'
                          |
                         'MSGR_MDIS_WAC_FILTER12'


Spacecraft Frames
---------------------------------------------------------------

   From [4] (with edits): (Note: The figures referenced below can not be
   easily reproduced here.  There is a diagram below that attempts, poorly,
   to illustrate the basic idea of what is contained there.  Refer
   to the aforementioned reference for specific details.)

   "The fundamental coordinate system used by MESSENGER G&C is called
   the spacecraft body frame.  The axes of this system are defined in
   [1] and shown in Figure 1.  The X axis is [parallel to] the nominal
   rotation axis for the solar panels.  The Z axis is the longitudinal
   axis of the central fuel tank (oxidizer tank) and passes through the
   center of the LVA nozzle on the top deck.  The Y axis completes a
   right-handed coordinate system, with -Y pointing out away from the
   sunshade."
   
   There are two coordinate system origins that have been used on the mission:
   One used during spacecraft development that has the orign at the adaptor
   ring and a second used by the G&C team in-flight that has the origin close
   to the center of the spacecraft body. A translation of 0.89662 m along the
   Z-axis accounts for the difference. The kernel contains only the alignment
   information and thus a single frame definition, since the axes between
   these coordinate systems are parallel.
   

   Spacecraft Body Frame (MSGR_SPACECRAFT):

                                                 Plus-X
                                                  Solar Panel
                                                     |
                      Sunshade__                     |  . ' |
                                \  .  '  \          . '     |      .>
                            .  '          \       '         |  . '
                          /               |     |           |'
                         /            .   |     |           |  X-Axis
         Minus-X         |       .  '  |  |     |           |
          Solar Panel    |    '        |  |  . '|           |
             |           |  |          |  |'    |         .
             | . ' |     |  |          |  |     |     . '
           . '     |     |  |     x    |  |     | . '
         '         |  . '|  |      \   |  |
       |           |'    |  |       \  |  |
       |           |     |  |        \ |  |
       |           |     |  |  Y-Axis \   |
       |           |     |  |    .  '  \| /
       |         .       |    '        - /
       |     . '         \          .  '
       | . '              \  .  ' |   \
                                  |    \_________ Bottom of
                                  |                 Adapter Ring
                                  |
                                  V  Z-Axis




      The orientation of the spacecraft body frame with respect to an
      inertial frame, typically J2000, is provided by a C-kernel (see
      [1] for details).

         \begindata

         FRAME_MSGR_SPACECRAFT       = -236000
         FRAME_-236000_NAME          = 'MSGR_SPACECRAFT'
         FRAME_-236000_CLASS         = 3
         FRAME_-236000_CLASS_ID      = -236000
         FRAME_-236000_CENTER        = -236
         CK_-236000_SCLK             = -236
         CK_-236000_SPK              = -236

         \begintext

   Solar Panel Frames (MSGR_SOLARPANEL_PLUS, MSGR_SOLARPANEL_MINUS):

      From [4]:

      "The G&C flight software assumes that each panel rotates about
      an axis that is parallel to the s/c X axis.  The orientation or
      position of each panel relative to the s/c body frame is
      represented by the angle measured in the YZ plane, relative to
      the s/c +Z axis, made by a vector normal to the panel as shown
      in Figure 14.  Specifically the normal vector pointing outward
      from the cell side of each panel is used as the position indicator.
      The zero angle is the +Z axis and angle increases with rotation
      about the +X axis (counterclockwise rotation when looking down
      the +X axis.)  While this definition allows the angle to take on
      the full range of values from 0 to 360 deg, the array drives only
      provide a 220-degree range of motion.  Hardware stops are located
      at the 70 deg and 290 degree locations.  The actual panel angles
      seen in flight will be restricted to this range.  The panels will
      be deployed such that both will be at the 90-deg location (panel
      cell-side normal vector aligned with s/c -Y axis)."

                                    o
                                +180

                                  |
                       _          |
                      | |       __|__
                      | |      \     /
                      | |       \ | /
                      | |    ____\ /____
                      | |   |     |     |
                      | |   |           |---Spacecraft Body
         o            | |   |     |     |
      +90             | |   |           |                      o
          <-----------------|-----+ - - - - - - - - - - -  +270
       -Y             | |   | _.-'|'-._ |
         sc           | | _.-'    |    '-._
                      _.-'  |     |     |  '-._
                  _.-'| |   |_____|_____|      '-._
            o  .-'    | |       / | \              '-.     o
         +70          | |      /__|__\                 +290
       Hardware       | |         |                  Hardware
         Stop         |_|         |                    Stop
                       |          |
                    Sunshade      |
                                  |
                                  V +Z
                                      sc
                               o     o
                              0 , 360

      The principal axes of both solar panel frames are

            +X-axis : Coincident with the spacecraft X-axis (the rotation axis
                      of the panel)

            +Z-axis : Outward directed normal to the solar cell side of the
                      solar panel

            +Y-axis : Completes the right-handed frame definition

      Since the solar panels are articulating structures, their
      orientation relative to the spacecraft body frame or the inertial
      frame are stored in C-kernels. (See [1] for details.)

         \begindata

         FRAME_MSGR_SOLARPANEL_PLUS  = -236001
         FRAME_-236001_NAME          = 'MSGR_SOLARPANEL_PLUS'
         FRAME_-236001_CLASS         = 3
         FRAME_-236001_CLASS_ID      = -236001
         FRAME_-236001_CENTER        = -236
         CK_-236001_SCLK             = -236
         CK_-236001_SPK              = -236

         FRAME_MSGR_SOLARPANEL_MINUS = -236002
         FRAME_-236002_NAME          = 'MSGR_SOLARPANEL_MINUS'
         FRAME_-236002_CLASS         = 3
         FRAME_-236002_CLASS_ID      = -236002
         FRAME_-236002_CENTER        = -236
         CK_-236002_SCLK             = -236
         CK_-236002_SPK              = -236

         \begintext


Antenna Frame Definitions
---------------------------------------------------------------

   From [25]:

   MESSENGER carries 2 phased-array antennas (high-gain) and 2 fanbeam
   antennas (medium-gain). A single reference frame is used for both antenna
   sets. The frame is defined by a rotation axis and a nominal position. The
   rotation axis is the axis about which the phased-array antenna boresight
   rotates and is normal to the plane in which the boresight moves. The
   nominal position is an axis in the scan plane that represents the 'zero'
   position; this axis is perpendicular to the rotation axis. The final axis
   of the antenna reference frame completes a right-handed system with the
   rotation axis as the Z axis and the nominal position as the -Y axis. The
   nominal alignment used for the antennas by Guidance and Control (G&C)
   flight software sets the rotation axis to the +Z axis and the nominal
   position to the -Y axis of the spacecraft body frame.

   The measured alignment made during spacecraft integration provide the
   following offsets from the nominal frame, given in spacecraft coordinate:

                                           [ -0.002285675 ]
           Antenna Rotation Axis (+Z)    = [ -0.002090396 ]
                                           [  0.999995202 ]

                                           [  0.006762866 ]
           Antenna Nominal Position (-Y) = [ -0.999974982 ]
                                           [ -0.002073187 ]

   The antenna X axis completes the right-handed frame.

   The following frame definition for the front phased array antenna captures
   these values. The remaining antenna frames are identical and are defined
   as identity offsets to this frame.

      \begindata

      FRAME_MSGR_RS_PAA_FRONT     = -236150
      FRAME_-236150_NAME          = 'MSGR_RS_PAA_FRONT'
      FRAME_-236150_CLASS         = 4
      FRAME_-236150_CLASS_ID      = -236150
      FRAME_-236150_CENTER        = -236
      TKFRAME_-236150_RELATIVE    = 'MSGR_SPACECRAFT'
      TKFRAME_-236150_SPEC        = 'MATRIX'
      TKFRAME_-236150_MATRIX      = ( 0.999974519315555
                                      0.006758094929627
                                      0.002299754888269
                                     -0.006762869909170
                                      0.999974978885710
                                      0.002074895928634
                                     -0.002285675000000
                                     -0.002090396000000
                                      0.999995202000000 )

      FRAME_MSGR_RS_PAA_REAR      = -236160
      FRAME_-236160_NAME          = 'MSGR_RS_PAA_REAR'
      FRAME_-236160_CLASS         = 4
      FRAME_-236160_CLASS_ID      = -236160
      FRAME_-236160_CENTER        = -236
      TKFRAME_-236160_RELATIVE    = 'MSGR_RS_PAA_FRONT'
      TKFRAME_-236160_SPEC        = 'MATRIX'
      TKFRAME_-236160_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0 )

      FRAME_MSGR_RS_FB_FRONT      = -236170
      FRAME_-236170_NAME          = 'MSGR_RS_FB_FRONT'
      FRAME_-236170_CLASS         = 4
      FRAME_-236170_CLASS_ID      = -236170
      FRAME_-236170_CENTER        = -236
      TKFRAME_-236170_RELATIVE    = 'MSGR_RS_PAA_FRONT'
      TKFRAME_-236170_SPEC        = 'MATRIX'
      TKFRAME_-236170_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0 )

      FRAME_MSGR_RS_FB_REAR       = -236180
      FRAME_-236180_NAME          = 'MSGR_RS_FB_REAR'
      FRAME_-236180_CLASS         = 4
      FRAME_-236180_CLASS_ID      = -236180
      FRAME_-236180_CENTER        = -236
      TKFRAME_-236180_RELATIVE    = 'MSGR_RS_PAA_FRONT'
      TKFRAME_-236180_SPEC        = 'MATRIX'
      TKFRAME_-236180_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0 )

      \begintext



Gamma-Ray and Neutron Spectrometer (GRNS) Frame Definitions
---------------------------------------------------------------

   From the diagram provided in [6], which has been recreated in
   part below:


                   |                      |
                   |      |         |     |
                   |      |         |     |
                   |      |    o------------------>
                   |      |  __|__  |     |        X
                   |  NS----|  |  | |     |         sc
                   |      | |__|__| |     |
                   |      \    |    /     |
                   |       \___|___/      |
                   |           |          |-----Spacecraft Body
                   |___________|__________|
                    | |   /    |    \
             GRS----| |  /     |     \
                    |_| /      |      \------Adapter Ring
                       /_______|_______\
                               |
                               |
                               V Z
                                  sc

   The Gamma-Ray Spectrometer (GRS) is mounted on the -X side of the
   spacecraft body, outside the adapter ring.  The Neutron Spectrometer
   is mounted outside the adapter ring as well, but further up the -Z
   axis of the spacecraft body and closer to the center.

   From [6]:

   "After conferring with other members of the MESSENGER Geochemistry
   Team, I have decided to use the S/C body coordinate system as the
   coordinate system for both the GRS and NS."

   Since GRS and NS are both mounted directly to the spacecraft body,
   all that is required is to supply the frame definition with the
   identity matrix:

      [     ]   [ 1  0  0 ]
      [ ROT ] = [ 0  1  0 ]
      [     ]   [ 0  0  1 ]

   where ROT specifies the transformation from the instrument frame
   to the spacecraft body frame.

   Gamma-Ray Spectrometer (MSGR_GRNS_GRS):

      \begindata

      FRAME_MSGR_GRNS_GRS         = -236200
      FRAME_-236200_NAME          = 'MSGR_GRNS_GRS'
      FRAME_-236200_CLASS         = 4
      FRAME_-236200_CLASS_ID      = -236200
      FRAME_-236200_CENTER        = -236
      TKFRAME_-236200_RELATIVE    = 'MSGR_SPACECRAFT'
      TKFRAME_-236200_SPEC        = 'MATRIX'
      TKFRAME_-236200_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0 )

      \begintext

   Neutron Spectrometer (MSGR_GRNS_NS):

      \begindata

      FRAME_MSGR_GRNS_NS          = -236210
      FRAME_-236210_NAME          = 'MSGR_GRNS_NS'
      FRAME_-236210_CLASS         = 4
      FRAME_-236210_CLASS_ID      = -236210
      FRAME_-236210_CENTER        = -236
      TKFRAME_-236210_RELATIVE    = 'MSGR_SPACECRAFT'
      TKFRAME_-236210_SPEC        = 'MATRIX'
      TKFRAME_-236210_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0 )

      \begintext


X-Ray Spectrometer (XRS) Frames:
---------------------------------------------------------------

   From [7]:

   "The MXU unit which consists of three gas proportional counters
   has a hexagon pattern collimator which restricts the FOV to 12
   degrees full angle.  And the MXU is mounted on the main instrument
   deck looking directly down (nadir)."

   MXU (MSGR_XRS_MXU):

      \begindata

      FRAME_MSGR_XRS_MXU          = -236300
      FRAME_-236300_NAME          = 'MSGR_XRS_MXU'
      FRAME_-236300_CLASS         = 4
      FRAME_-236300_CLASS_ID      = -236300
      FRAME_-236300_CENTER        = -236
      TKFRAME_-236300_RELATIVE    = 'MSGR_SPACECRAFT'
      TKFRAME_-236300_SPEC        = 'MATRIX'
      TKFRAME_-236300_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0 )

      \begintext

   From [7]:

   "The SAX unit which is our Silicon-PIN solar monitor is mounted
   directly behind the sun shade and looks at the Sun all the time
   while in orbit.  It has a FOV of 42 degrees defined by a circular
   collimator, and it is mounted along the center axis of the sun
   shade."

   The center axis of the sun shade points down the -Y axis of the
   spacecraft body frame.  If we make the Z-axis of the XRS_SAX frame
   the boresight of the detector, one such rotation matrix would be:

      [     ]   [ 1  0  0 ]
      [ ROT ] = [ 0  0 -1 ]
      [     ]   [ 0  1  0 ]

   which rotates vectors from the instrument frame into the spacecraft
   body frame.

   SAX (MSGR_XRS_SAX):

      \begindata

      FRAME_MSGR_XRS_SAX          = -236310
      FRAME_-236310_NAME          = 'MSGR_XRS_SAX'
      FRAME_-236310_CLASS         = 4
      FRAME_-236310_CLASS_ID      = -236310
      FRAME_-236310_CENTER        = -236
      TKFRAME_-236310_RELATIVE    = 'MSGR_SPACECRAFT'
      TKFRAME_-236310_SPEC        = 'MATRIX'
      TKFRAME_-236310_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                     -1.0
                                      0.0 )

      \begintext


Magnetometer (MAG) Frames:
---------------------------------------------------------------

   From [16]:

   "The transformation from sensor coordinates to spacecraft coordinates
   with the boom deployed as determined from the pre-launch calibrations
   is given by:

           [Bx-sc]   [  0.996345296191  0.0  0.085416923137 ] [Bx-mag]
           [By-sc] = [  0.0             1.0  0.0            ] [By-mag]   (1)
           [Bz-sc]   [ -0.085416923137  0.0  0.996345296191 ] [Bz-mag]

   The diagonal elements for X and Z are cos(4.9 degrees), the off-diagonal
   elements for X and Z are sin(4.9 degrees). The off-diagonal elements
   between X-Y and Y-Z will be determined using the in-flight roll maneuver
   results."

   This translates directly into the following frame definition:

   Deployed Magnetometer Frame (MSGR_MAG):

      \begindata

      FRAME_MSGR_MAG              = -236400
      FRAME_-236400_NAME          = 'MSGR_MAG'
      FRAME_-236400_CLASS         = 4
      FRAME_-236400_CLASS_ID      = -236400
      FRAME_-236400_CENTER        = -236
      TKFRAME_-236400_RELATIVE    = 'MSGR_SPACECRAFT'
      TKFRAME_-236400_SPEC        = 'MATRIX'
      TKFRAME_-236400_MATRIX      = ( 0.996345296191
                                      0.0
                                     -0.085416923137
                                      0.0
                                      1.0
                                      0.0
                                      0.085416923137
                                      0.0
                                      0.996345296191 )

      \begintext

   The nominal orientation, supplied in [8] is as follows:

           [Bx-sc]   [ 1  0  0 ] [Bx-mag]
           [By-sc] = [ 0  1  0 ] [By-mag]
           [Bz-sc]   [ 0  0  1 ] [Bz-mag]

   which translates into the following frame definition:

      FRAME_MSGR_MAG              = -236400
      FRAME_-236400_NAME          = 'MSGR_MAG'
      FRAME_-236400_CLASS         = 4
      FRAME_-236400_CLASS_ID      = -236400
      FRAME_-236400_CENTER        = -236
      TKFRAME_-236400_RELATIVE    = 'MSGR_SPACECRAFT'
      TKFRAME_-236400_SPEC        = 'MATRIX'
      TKFRAME_-236400_MATRIX      = ( 1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0
                                      0.0
                                      0.0
                                      0.0
                                      1.0 )

   Note: The above nominal frame definition is provided merely for
   reference purposes.  Since it is not surrounded by the "\begindata"
   and "\begintext" lines, it will not be loaded into the kernel pool
   with the rest of the frame data.

   From [16]:

   "In the stowed configuration the sensor is rotated 90 relative to its
   deployed orientation so that +Y sensor direction is aligned with the
   +Z spacecraft direction and the +Z sensor direction is aligned with
   the -Y spacecraft direction.  The precise stowed orientation remains
   to be determined after launch, but the pre-launch coordinate
   transformation matrix for the stowed configuration is given by modifying
   (1) appropriately:

           [Bx-sc]   [  0.996345296191  0.0  0.085416923137 ] [Bx-mag]
           [By-sc] = [ -0.085416923137  0.0  0.996345296191 ] [By-mag]   (2)"
           [Bz-sc]   [  0.0            -1.0  0.0            ] [Bz-mag]

   This translates directly into the following frame definition:

   Stowed Magnetometer Frame (MSGR_MAG_STOWED):

      \begindata

      FRAME_MSGR_MAG_STOWED         = -236410
      FRAME_-236410_NAME            = 'MSGR_MAG_STOWED'
      FRAME_-236410_CLASS           = 4
      FRAME_-236410_CLASS_ID        = -236410
      FRAME_-236410_CENTER          = -236
      TKFRAME_-236410_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236410_SPEC          = 'MATRIX'
      TKFRAME_-236410_MATRIX        = ( 0.996345296191
                                       -0.085416923137
                                        0.0
                                        0.0
                                        0.0
                                       -1.0
                                        0.085416923137
                                        0.996345296191
                                        0.0    )

      \begintext

   The nominal stowed orientation supplied in [8] is as follows:

           [Bx-sc]   [ 1  0  0 ] [Bx-mag]
           [By-sc] = [ 0  0  1 ] [By-mag]
           [Bz-sc]   [ 0 -1  0 ] [Bz-mag]

   which results in the nominal frame definition displayed below:

      FRAME_MSGR_MAG_STOWED         = -236410
      FRAME_-236410_NAME            = 'MSGR_MAG_STOWED'
      FRAME_-236410_CLASS           = 4
      FRAME_-236410_CLASS_ID        = -236410
      FRAME_-236410_CENTER          = -236
      TKFRAME_-236410_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236410_SPEC          = 'MATRIX'
      TKFRAME_-236410_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        0.0
                                       -1.0
                                        0.0
                                        1.0
                                        0.0 )

   Lastly, [16] provides an orientation for a contingency orientation,
   in which the magnetometer is halfway deployed:

   "In the halfway deployed configuration the sensor is rotated -90 relative
   to its deployed orientation so that +Y sensor direction is aligned with the
   -Z spacecraft direction and the +Z sensor direction is aligned with
   the +Y spacecraft direction.  The precise stowed orientation remains
   to be determined after launch, but the pre-launch coordinate
   transformation matrix for the stowed configuration is given by modifying
   (1) appropriately:

           [Bx-sc]   [  0.996345296191  0.0  0.085416923137 ] [Bx-mag]
           [By-sc] = [  0.085416923137  0.0 -0.996345296191 ] [By-mag]   (3)"
           [Bz-sc]   [  0.0             1.0  0.0            ] [Bz-mag]

   This translates directly into the following frame definition:

   Halfway Deployed Magnetometer Frame (MSGR_MAG_HALFWAY):

      \begindata

      FRAME_MSGR_MAG_HALFWAY        = -236420
      FRAME_-236420_NAME            = 'MSGR_MAG_HALFWAY'
      FRAME_-236420_CLASS           = 4
      FRAME_-236420_CLASS_ID        = -236420
      FRAME_-236420_CENTER          = -236
      TKFRAME_-236420_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236420_SPEC          = 'MATRIX'
      TKFRAME_-236420_MATRIX        = ( 0.996345296191
                                        0.085416923137
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.085416923137
                                       -0.996345296191
                                        0.0    )

      \begintext

   The nominal halfway deployed orientation supplied in [8] is as follows:

           [Bx-sc]   [ 1  0  0 ] [Bx-mag]
           [By-sc] = [ 0  0 -1 ] [By-mag]
           [Bz-sc]   [ 0  1  0 ] [Bz-mag]

   Halfway Deployed Magnetometer Frame (MSGR_MAG_HALFWAY):

      FRAME_MSGR_MAG_HALFWAY        = -236420
      FRAME_-236420_NAME            = 'MSGR_MAG_HALFWAY'
      FRAME_-236420_CLASS           = 4
      FRAME_-236420_CLASS_ID        = -236420
      FRAME_-236420_CENTER          = -236
      TKFRAME_-236420_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236420_SPEC          = 'MATRIX'
      TKFRAME_-236420_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                       -1.0
                                        0.0 )


Mercury Laser Altimeter (MLA) Frames:
---------------------------------------------------------------

   From [11]:

   "MLA Laser to MLA Cube Angles:


      Cube face defined as -Z

                    ^             .|
                    |              |
       ____________ |            ' |
      |____________||           .  |    +-------> +X
        /\    /\  |*|              |    |
       /__\  /__\  ^           '   |    |
                   |          .    |    |
       Alignment Cube              |    |
                             '     |    V
                            .      |   +Z
      5-10-03: 4'-17"              |
      6-25-03: 4'-18"      '       |
                          .
        delta = 1"     Laser Direction



      Cube face defined as -Z

                 ^        |.
                 |        |
       __________|_       | '
      |__________|_|      |  .           +Y <--------+
        /\    /|*|        |                          |
       /__\  /__\^        |   '                      |
                 |        |    .                     |
     Alignment Cube       |                          |
                          |     '                    V
                          |      .                  +Z
      5-10-03: 1'-4"      |
      6-25-03: 0'-58"     |       '
                                   .
        delta = 6"             Laser Direction


   Cube XZ Plane: MLA Laser at -(4' 18") or -1.251 mrad
   Cube YZ Plane: MLA Laser at -(0' 58") or -0.281 mrad"

   The above provides a clear definition of the MLA boresight vector
   expressed in the MLA cube frame.  A discussion of the MLA cube
   frame is presented in [10]:

                          [ 0.99999742474863 ]
   "(1) MLA cube x-axis = [ 0.00178578502884 ] in s/c body axes.
                          [-0.00140052416512 ]

                          [-0.00178584176639 ]
    (2) MLA cube y-axis = [ 0.99999840461256 ] in s/c body axes.
                          [-0.00003926213659 ]

    (3) Therefore, MLAcube_toSCBody matrix =

       [ 0.99999742474863  -0.00178584176639  0.00140045181700 ]
       [ 0.00178578502884   0.99999840461256  0.00004176315003 ]"
       [-0.00140052416512  -0.00003926213659  0.99999901849479 ]

   So, to properly support the MLA frame definitions two frames must
   be provided.  The first of these two frames is the aforementioned
   MLA alignment cube frame.  Simply place the above matrix into a
   frame definition, note the MATRIX specification utilizes column
   major ordering:

   MLA Cube Frame (MSGR_MLA_CUBE):

      FRAME_MSGR_MLA_CUBE           = -236510
      FRAME_-236510_NAME            = 'MSGR_MLA_CUBE'
      FRAME_-236510_CLASS           = 4
      FRAME_-236510_CLASS_ID        = -236510
      FRAME_-236510_CENTER          = -236
      TKFRAME_-236510_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236510_SPEC          = 'MATRIX'
      TKFRAME_-236510_MATRIX        = ( 0.99999742474863
                                        0.00178578502884
                                       -0.00140052416512
                                       -0.00178584176639
                                        0.99999840461256
                                       -0.00003926213659
                                        0.00140045181700
                                        0.00004176315003
                                        0.99999901849479 )

   Next, the MLA base frame itself is defined in as relative to the above
   MSGR_MLA_CUBE frame.  Since only boresight information has been provided,
   the frame transformation outlined below was constructed by computing the
   RA and DEC of the boresight vector relative to the MSGR_MLA_CUBE frame.
   These angles are then utilized in the following fashion to construct
   the frame definition:

      [     ]   [          ]  [           ]  [     ]
      [ ROT ] = [ -(RA+90) ]  [ -(90-DEC) ]  [ 0.0 ]
      [     ]   [          ]  [           ]  [     ]
                            Z              X        Z

   where [x]  represents the rotation matrix of a given x about the axis i.
            i

   The supplied boresight computed from the angles given in [11]:

                             [-0.001251    ]
      MLA Boresight Vector = [-0.000281    ]
                             [ 0.999999178 ]

   This places the supplied boresight vector in the cube frame as the
   Z-axis of the instrument frame.  The methodology outlined above
   results in the following:

      [     ]   [                  ]  [                    ]  [     ]
      [ ROT ] = [ -282.65967509403 ]  [ -0.073462996470092 ]  [ 0.0 ]
      [     ]   [                  ]  [                    ]  [     ]
                                    Z                       X        Z

   where [x]  represents the rotation matrix of a given angle x about the
            i
   axis i.

   MLA Frame (MSGR_MLA):

      FRAME_MSGR_MLA                = -236500
      FRAME_-236500_NAME            = 'MSGR_MLA'
      FRAME_-236500_CLASS           = 4
      FRAME_-236500_CLASS_ID        = -236500
      FRAME_-236500_CENTER          = -236
      TKFRAME_-236500_RELATIVE      = 'MSGR_MLA_CUBE'
      TKFRAME_-236500_SPEC          = 'ANGLES'
      TKFRAME_-236500_ANGLES        = ( -282.65967509403 -0.07346299647 0.0 )
      TKFRAME_-236500_AXES          = (  3                1             3   )
      TKFRAME_-236500_UNITS         = 'DEGREES'


   The frame definition given above for MLA has changed due to an updated
   boresight vector given in [16]. The new boresight vector in the spacecraft
   body frame is:

                             [ 0.0022105  ]
      MLA Boresight Vector = [ 0.0029215  ]
                             [ 0.99999329 ]

   This boresight pertains to the laser (transmitter), not the detector
   (receiver).

   The old MSGR_MLA frame defined above is used to determine the new
   MSGR_MLA frame based on the new boresight vector as follows:

   Set the instrument Z axis to the normalized boresight vector, and select
   the old MSGR_MLA frame's Y axis as the new MSGR_MLA Y axis. Set the X axis
   to the cross product of Y and Z, then readjust Y to form an orthogonal
   frame. This procedure gives us the new MSGR_MLA frame definition:

      \begindata

         FRAME_MSGR_MLA                = -236500
         FRAME_-236500_NAME            = 'MSGR_MLA'
         FRAME_-236500_CLASS           = 4
         FRAME_-236500_CLASS_ID        = -236500
         FRAME_-236500_CENTER          = -236
         TKFRAME_-236500_RELATIVE      = 'MSGR_SPACECRAFT'
         TKFRAME_-236500_SPEC          = 'MATRIX'
         TKFRAME_-236500_MATRIX        = (  2.2090105880850D-01
                                           -9.7529336496130D-01
                                            2.3610336177738D-03
                                            9.7529371776252D-01
                                            2.2089435732998D-01
                                           -2.8012484244305D-03
                                            2.2104999983228D-03
                                            2.9214999977833D-03
                                            9.9999328924124D-01 )

      \begintext

   The MLA instrument consists not only of a transmitter but also of four
   receiver telescopes. The signals from the four telescopes can be optically
   combined onto a single spot on the detector [18].

   According to [19], there does not appear to be a significant offset between
   the MLA laser boresight and any of the four MLA receiver telescopes and
   that for the time being it may be assumed that the laser and telescopes are
   perfectly boresighted. Thus, the receiver frame will be defined as the
   identity rotation from the transmitter frame.

      \begindata

         FRAME_MSGR_MLA_RECEIVER       = -236501
         FRAME_-236501_NAME            = 'MSGR_MLA_RECEIVER'
         FRAME_-236501_CLASS           = 4
         FRAME_-236501_CLASS_ID        = -236501
         FRAME_-236501_CENTER          = -236
         TKFRAME_-236501_RELATIVE      = 'MSGR_MLA'
         TKFRAME_-236501_SPEC          = 'MATRIX'
         TKFRAME_-236501_MATRIX        = ( 1.0
                                           0.0
                                           0.0
                                           0.0
                                           1.0
                                           0.0
                                           0.0
                                           0.0
                                           1.0 )

      \begintext


Mercury Atmospheric and Surface Composition Spectrometer (MASCS) Frames:
---------------------------------------------------------------

   Presented here first are the initial MASCS frames defined using pre-launch
   calibration data. These initial frames remain in the kernel for reference
   only and are not loaded into the SPICE kernel pool. The pre-launch
   frames are constructed using an arbitrarily chosen Euler rotation. The
   post-launch frames, defined below, are constructed such that the instrument
   axes closely match the spacecraft axes (the nominal instrument frame aligns
   with the spacecraft frame) [22].

   Pre-launch frames
   -----------------

   From [12]:

   "The telescope boresight was measured by using the 2 axis manipulator
   to position the image of a star source from the alignment mirror back
   onto itself.  The point source consisted of a halogen lamp illuminating
   a pinhole that was at the focus of the collimator.  The star source was
   reflected off of the alignment mirror and the manipulator was oriented
   such that the star image was visually placed on top of the pinhole.
   The star source was then scanned across the entrance slit in both the
   spectral and spatial directions.  In the spatial direction both the
   atmospheric and surface slits were measured.

   The telescope boresight was determined before and after shake.  The
   experiment consisted of determining the centroid of the entrance slit
   in both atmospheric and surface mode with respect to the alignment
   mirror.  The values are given in the table below.  Note that the +/-
   values are with respect to rotation about the respective spacecraft
   axis following the right hand rule:

                                   Pre-Shake             Post-Shake
                                   --------------        ---------------
      Atmospheric Slit                +0.076                +0.098
      (Spectral Direction)
      Rotation about y axis

      Atmospheric Slit                +0.026                -0.021
      (Spatial Direction)
      Rotation about x axis

      Surface Slit                    +0.076                +0.098
      (Spectral Direction)
      Rotation about y axis

      Surface Slit                    +0.016                -0.031
      (Spatial Direction)
      Rotation about x axis

      VIRS Slit                                             +0.478
      Rotation about y axis

      VIRS Slit                                             -0.002
      Rotation about x axis

   These rotation angles were then converted into direction cosines
   for the three boresights:

                               X            Y           Z
                           ----------  -----------  ----------
      UVVS Atmosphere       -0.00171    -0.00037     0.999998
      UVVS Surface          -0.00171    -0.00054     0.999998  "
      VIRS                  -0.00834    -0.00003     0.99997

   For the UVVS Atmosphere Boresight, [12] provides the following
   figures:

      Boresight Rotation relative to cube in the Y-Z plane:

                 _________________________
                |                         |
                |                         |  UVVS Atmosphere
                |                         |    Boresight
                |         Optical         |
                |         Alignment Cube  |   /
                |              \          |  /     o
                |               \         | / 0.021
                |                \        |/
                |                |*|-------------> +Z
                |                         |          sc
                |_________________________|
                             |
                             |
                             |
                             V
                            +Y
                              sc


      Boresight Rotation relative to cube in the X-Z plane:

                 _________________________
                |                         |
                |                         |  UVVS Atmosphere
                |                         |    Boresight
                |                         |
                |                         |   /
                |                         |  /
                |                         | /
                |                         |/
                |                         |       o
                |                        /|  0.098
                |_______________________/_|
                  |              |*|-------------> +Z
                  |               |                  sc
                  |               |
                  V          Optical Alignment Cube
                 +X
                   sc

   The UVVS Atmosphere frame only has boresight information provided, thus
   the frame transformation outlined below was constructed by computing the
   RA and DEC of the boresight vector relative to the MSGR_SPACECRAFT frame.
   These angles are then utilized in the following fashion to construct the
   frame definition:

      [     ]   [          ]  [           ]  [     ]
      [ ROT ] = [ -(RA+90) ]  [ -(90-DEC) ]  [ 0.0 ]
      [     ]   [          ]  [           ]  [     ]
                            Z              X        Z

   where [x]  represents the rotation matrix of a given angle x about the
            i
   axis i.

   The supplied boresight computed from the angles given in [12]:

                                         [-0.001710422 ]
      UVVS Atmosphere Boresight Vector = [-0.000366519 ]
                                         [ 0.99999847  ]

   The following matrix places the above boresight vector in the frame
   as the Z-axis of this instrument frame.  The methodology outlined above
   results in the following:

      [     ]   [                  ]  [                    ]  [     ]
      [ ROT ] = [ -282.09475707701 ]  [ -0.100224759480330 ]  [ 0.0 ]
      [     ]   [                  ]  [                    ]  [     ]
                                    Z                       X        Z

   where [x]  represents the rotation matrix of a given angle x about the
            i
   axis i.

   This yields the following frame definition:

   UVVS Atmosphere (MSGR_MASCS_UVVS_ATMO):

      FRAME_MSGR_MASCS_UVVS_ATMO    = -236600
      FRAME_-236600_NAME            = 'MSGR_MASCS_UVVS_ATMO'
      FRAME_-236600_CLASS           = 4
      FRAME_-236600_CLASS_ID        = -236600
      FRAME_-236600_CENTER          = -236
      TKFRAME_-236600_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236600_SPEC          = 'ANGLES'
      TKFRAME_-236600_ANGLES        = ( -282.09475707701
                                        -0.10022475948033
                                         0.0              )
      TKFRAME_-236600_AXES          = (  3
                                         1
                                         3  )
      TKFRAME_-236600_UNITS         = 'DEGREES'

   Applying precisely the same boresight to frame definition methodology
   as outlined above to the UVVS Surface boresight gives:

                                      [-0.001710422 ]
      UVVS Surface Boresight Vector = [-0.000541052 ]
                                      [ 0.999998391 ]

   The following matrix places the above boresight vector in the frame
   as the Z-axis of this instrument frame.  The methodology outlined for
   the atmospheric boresight yields the following:

      [     ]   [                  ]  [                   ]  [     ]
      [ ROT ] = [ -287.55354917649 ]  [ -0.10278620334886 ]  [ 0.0 ]
      [     ]   [                  ]  [                   ]  [     ]
                                    Z                      X        Z

   where [x]  represents the rotation matrix of a given angle x about the
            i
   axis i.

   This yields the following frame definition:

   UVVS Surface (MSGR_MASCS_UVVS_SURF):

      FRAME_MSGR_MASCS_UVVS_SURF    = -236610
      FRAME_-236610_NAME            = 'MSGR_MASCS_UVVS_SURF'
      FRAME_-236610_CLASS           = 4
      FRAME_-236610_CLASS_ID        = -236610
      FRAME_-236610_CENTER          = -236
      TKFRAME_-236610_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236610_SPEC          = 'ANGLES'
      TKFRAME_-236610_ANGLES        = ( -287.55354917649
                                        -0.10278620334886
                                         0.0 )
      TKFRAME_-236610_AXES          = (  3
                                         1
                                         3  )
      TKFRAME_-236610_UNITS         = 'DEGREES'

   Again, from [12], the following VIRS diagrams:

      Boresight Rotation relative to cube in the Y-Z plane:

                 _________________________
                |                         |
                |                         |
                |                         |    VIRS Boresight
                |         Optical         |
                |         Alignment Cube  |   /
                |              \          |  /     o
                |               \         | / 0.002
                |                \        |/
                |                |*|-------------> +Z
                |                         |          sc
                |_________________________|
                             |
                             |
                             |
                             V
                            +Y
                              sc


      Boresight Rotation relative to cube in the X-Z plane:

                 _________________________
                |                         |
                |                         |
                |                         |    VIRS Boresight
                |                         |
                |                         |   /
                |                         |  /
                |                         | /
                |                         |/
                |                         |       o
                |                        /|  0.478
                |_______________________/_|
                  |              |*|-------------> +Z
                  |               |                  sc
                  |               |
                  V          Optical Alignment Cube
                 +X
                   sc



   As with the UVVS frames, the report only supplied boresight alignment
   information.  Proceeding with the same methodology to build a frame:

                              [-0.008342577  ]
      VIRS Boresight Vector = [-0.0000349066 ]
                              [ 0.999965199  ]

   The following matrix places the above boresight vector in the frame
   as the Z-axis of this instrument frame.  The methodology outlined above
   results in the following:

      [     ]   [                  ]  [                   ]  [     ]
      [ ROT ] = [ -270.23973278111 ]  [ -0.47800418167392 ]  [ 0.0 ]
      [     ]   [                  ]  [                   ]  [     ]
                                    Z                      X        Z

   where [x]  represents the rotation matrix of a given x about the axis i.
            i

   This yields the following frame definition:

   VIRS (MSGR_MASCS_VIRS):

      FRAME_MSGR_MASCS_VIRS         = -236620
      FRAME_-236620_NAME            = 'MSGR_MASCS_VIRS'
      FRAME_-236620_CLASS           = 4
      FRAME_-236620_CLASS_ID        = -236620
      FRAME_-236620_CENTER          = -236
      TKFRAME_-236620_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236620_SPEC          = 'ANGLES'
      TKFRAME_-236620_ANGLES        = ( -270.23973278111
                                        -0.47800418167392
                                         0.0              )
      TKFRAME_-236620_AXES          = (  3
                                         1
                                         3  )
      TKFRAME_-236620_UNITS         = 'DEGREES'


   Post-launch frames
   ------------------

   The MASCS UVVS Atmosphere coordinate system is defined such that the +Z
   axis in instrument coordinates is the boresight, and the +X and +Y axes in
   instrument coordinates nominally align with the spacecraft +X and +Y axes,
   respectively [22]. The measured boresight vector in spacecraft coordinates
   is given in [22] as

                                                  [ 0.0010035067 ]
      UVVS Atmosphere Boresight Vector (+Z    ) = [ 0.0024908088 ]
                                          inst    [ 0.99998428   ]

   Taking the cross product of the boresight with the spacecraft +X axis, we
   arrive at the instrument +Y axis:

                                            [ 1.0 ]     [  0.0               ]
      UVVS Atmosphere +Y Vector = +Z     x  [ 0.0 ]  =  [  0.999996897852665 ]
                                    inst    [ 0.0 ]     [ -0.002490840229153 ]

   And we use that to adjust the +X vector to form an orthogonal frame:

                                          [  0.999999496474825 ]
      UVVS Atmosphere +X Vector = Y x Z = [ -0.000002499605140 ]
                                          [ -0.001003515743938 ]

   Using these three vectors, we define the rotation matrix that takes vectors
   from the instrument frame to the spacecraft frame as

      [     ]   [  0.999999496474825  0.000000000000000  0.0010035067 ]
      [ ROT ] = [ -0.000002499605140  0.999996897852665  0.0024908088 ]
      [     ]   [ -0.001003515743938 -0.002490840229153  0.99998428   ]

      \begindata

      FRAME_MSGR_MASCS_UVVS_ATMO    = -236600
      FRAME_-236600_NAME            = 'MSGR_MASCS_UVVS_ATMO'
      FRAME_-236600_CLASS           = 4
      FRAME_-236600_CLASS_ID        = -236600
      FRAME_-236600_CENTER          = -236
      TKFRAME_-236600_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236600_SPEC          = 'MATRIX'
      TKFRAME_-236600_MATRIX        = ( 0.999999496474825
                                       -0.000002499605140
                                       -0.001003515743938
                                        0.000000000000000
                                        0.999996897852665
                                       -0.002490840229153
                                        0.001003506700000
                                        0.002490808800000
                                        0.999984280000000 )

      \begintext


   The MASCS UVVS Surface coordinate system is defined such that the +Z axis
   in instrument coordinates is the boresight, and the +X and +Y axes in
   instrument coordinates nominally align with the spacecraft +X and +Y axes,
   respectively [22]. The measured boresight vector in spacecraft coordinates
   is given in [22] as

                                               [ 0.0010035193 ]
      UVVS Surface Boresight Vector (+Z    ) = [ 0.0023419775 ]
                                       inst    [ 0.99999684   ]

   Taking the cross product of the boresight with the spacecraft +X axis, we
   arrive at the instrument +Y axis:

                                         [ 1.0 ]     [  0.0               ]
      UVVS Surface +Y Vector = +Z     x  [ 0.0 ]  =  [  0.999997257564644 ]
                                 inst    [ 0.0 ]     [ -0.002341978477930 ]

   And we use that to adjust the +X vector to form an orthogonal frame:

                                       [  0.999999496474467 ]
      UVVS Surface +X Vector = Y x Z = [ -0.000002350220401 ]
                                       [ -0.001003516461651 ]

   Using these three vectors, we define the rotation matrix that takes vectors
   from the instrument frame to the spacecraft frame as

      [     ]   [  0.999999496474467  0.000000000000000  0.0010035193 ]
      [ ROT ] = [ -0.000002350220401  0.999997257564644  0.0023419775 ]
      [     ]   [ -0.001003516461651 -0.002341978477930  0.99999684   ]

      \begindata

      FRAME_MSGR_MASCS_UVVS_SURF    = -236610
      FRAME_-236610_NAME            = 'MSGR_MASCS_UVVS_SURF'
      FRAME_-236610_CLASS           = 4
      FRAME_-236610_CLASS_ID        = -236610
      FRAME_-236610_CENTER          = -236
      TKFRAME_-236610_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236610_SPEC          = 'MATRIX'
      TKFRAME_-236610_MATRIX        = ( 0.999999496474467
                                       -0.000002350220401
                                       -0.001003516461651
                                        0.000000000000000
                                        0.999997257564644
                                       -0.002341978477930
                                        0.001003519300000
                                        0.002341977500000
                                        0.999996840000000 )

      \begintext


   The MASCS VIRS coordinate system is defined such that the +Z axis in
   instrument coordinates is the boresight, and the +X and +Y axes in
   instrument coordinates nominally align with the spacecraft +X and +Y axes,
   respectively [22]. The measured boresight vector in spacecraft coordinates
   is given in [22] as

                                       [ -0.005523799  ]
      VIRS Boresight Vector (+Z    ) = [  0.0026272051 ]
                               inst    [  0.9999812841 ]

   Taking the cross product of the boresight with the spacecraft +X axis, we
   arrive at the instrument +Y axis:

                                 [ 1.0 ]     [  0.0               ]
      VIRS +Y Vector = +Z     x  [ 0.0 ]  =  [  0.999996548785363 ]
                         inst    [ 0.0 ]     [ -0.002627245204210 ]

   And we use that to adjust the +X vector to form an orthogonal frame:

                               [ 0.999984743705669 ]
      VIRS +X Vector = Y x Z = [ 0.000014512374554 ]
                               [ 0.005523779982826 ]

   Using these three vectors, we define the rotation matrix that takes vectors
   from the instrument frame to the spacecraft frame as

      [     ]   [ 0.999984743705669  0.000000000000000  -0.005523799  ]
      [ ROT ] = [ 0.000014512374554  0.999996548785363   0.0026272051 ]
      [     ]   [ 0.005523779982826 -0.002627245204210   0.9999812841 ]

      \begindata

      FRAME_MSGR_MASCS_VIRS         = -236620
      FRAME_-236620_NAME            = 'MSGR_MASCS_VIRS'
      FRAME_-236620_CLASS           = 4
      FRAME_-236620_CLASS_ID        = -236620
      FRAME_-236620_CENTER          = -236
      TKFRAME_-236620_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236620_SPEC          = 'MATRIX'
      TKFRAME_-236620_MATRIX        = ( 0.999984743705669
                                        0.000014512374554
                                        0.005523779982826
                                        0.000000000000000
                                        0.999996548785363
                                       -0.002627245204210
                                       -0.005523799
                                        0.0026272051
                                        0.9999812841 )

      \begintext


Energetic Particle and Plasma Spectrometer (EPPS) Frames:
---------------------------------------------------------------

   From [13]:

   "The EPS coordinate system (Xu[eps], Yu[eps], Zu[eps]) is exactly
   parallel to the spacecraft coordinate system (Xu[s/c], Yu[s/c], Zu[s/c]),
   and the rotational transformation between these two coordinate systems
   is just the Unit Matrix, I."

   This results in the following frame definition:

   Energetic Particle Spectrometer (MSGR_EPPS_EPS):

      \begindata

      FRAME_MSGR_EPPS_EPS           = -236700
      FRAME_-236700_NAME            = 'MSGR_EPPS_EPS'
      FRAME_-236700_CLASS           = 4
      FRAME_-236700_CLASS_ID        = -236700
      FRAME_-236700_CENTER          = -236
      TKFRAME_-236700_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236700_SPEC          = 'MATRIX'
      TKFRAME_-236700_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   Again from [13]:

   "4) As expressed in the spacecraft coordinate system, the FIPS
       coordinate-defining unit vectors are:

          a. Xu[fips] = ( 0.669025, -0.426106, 0.608965)
          b. Yu[fips] = ( 0.000000,  0.819339, 0.573309)
          c. Zu[fips] = (-0.74324,  -0.383558, 0.548158)

    5) The matrix for transforming the representation of any vector
       V between the FIPS coordinate system and the spacecraft coordinate
       system, as defined by the equation:

          V[s/c] = TM * V[fips]

       is given by:

          TM = [Xu[fips], Yu[fips], Zu[fips]].

       That is, the columns (not the rows) of the transformation matrix TM
       are made up of the unit vectors that define the FIPS coordinate
       system."

   All of this translates directly to the following frame definition:

   Fast Imaging Plasma Spectrometer:

      \begindata

      FRAME_MSGR_EPPS_FIPS          = -236720
      FRAME_-236720_NAME            = 'MSGR_EPPS_FIPS'
      FRAME_-236720_CLASS           = 4
      FRAME_-236720_CLASS_ID        = -236720
      FRAME_-236720_CENTER          = -236
      TKFRAME_-236720_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236720_SPEC          = 'MATRIX'
      TKFRAME_-236720_MATRIX        = ( 0.669025
                                       -0.426106
                                        0.608965
                                        0.000000
                                        0.819339
                                        0.573309
                                       -0.743240
                                       -0.383558
                                        0.548158 )


      \begintext


Mercury Dual Imaging System (MDIS) Frames:
---------------------------------------------------------------

   From [14]:

   MDIS articulation angles are given according to the following
   diagram:


                                    o
                                   0
                                                o
                                   ^ +Z      +50
         Sun Shade                 |   sc    /        o
            |                      |        /    . +55  (Hard Stop)
            |                      |       /    '
            |                      |      /   .'
            |                      |     /   '
            |                      |    /  .'
            |                      |   /  '
            |                      |  / .'
            |                      | / '
            |                      |/.'
            |  --------------------+------------------> +Y
            |                      |.                     sc
            |                      | '
            |                      |  .
            |                      |   '
            |                      |    .
            |                      |     '
            |                      |      .
            |                      |       '
            |                      |        .
                                   |         ' o
                                     o     -185 (Hard Stop)
                                 -180

                         (Stowed Configuration)

   The MDIS base frame is defined such that the Z-axis of the frame is
   the rotation axis of the articulation.  From the above diagram, this
   is clearly the -X axis of the spacecraft body frame.  For further
   convenience the 0 degree boresight of MDIS lies along the +X axis
   of this frame.  The rotation matrix that rotates vectors from this
   frame to the spacecraft body is defined:

      [     ]   [ 0  0 -1 ]
      [ ROT ] = [ 0  1  0 ]
      [     ]   [ 1  0  0 ]

   Which results in the following nominal frame definition:

   MDIS Base (0 degree) Articulation Frame (MSGR_MDIS_BASE):

      FRAME_MSGR_MDIS_BASE          = -236880
      FRAME_-236880_NAME            = 'MSGR_MDIS_BASE'
      FRAME_-236880_CLASS           = 4
      FRAME_-236880_CLASS_ID        = -236880
      FRAME_-236880_CENTER          = -236
      TKFRAME_-236880_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236880_SPEC          = 'MATRIX'
      TKFRAME_-236880_MATRIX        = ( 0.0
                                        0.0
                                        1.0
                                        0.0
                                        1.0
                                        0.0
                                       -1.0
                                        0.0
                                        0.0 )

   [21] gives us an estimated matrix for the MDIS base frame based on the
   analysis of images of star fields. The above nominal definition remains
   for reference but will not be loaded into the SPICE kernel pool. [22]
   gives a correction to the matrices in [21]. The corrected rotation matrix
   taking vectors from the MDIS base frame to the spacecraft frame is

      [     ]
      [ ROT ] =
      [     ]

      [  0.002002115198661447 -1.5279882434764006E-4 -0.9999979840915934    ]
      [ -9.873433017452157E-4  0.999999500598698     -1.5477583508433595E-4 ]
      [  0.9999975083408637    9.876511904033615E-4   0.0020019633339067866 ]

   Which results in the following frame definition:

      FRAME_MSGR_MDIS_BASE          = -236880
      FRAME_-236880_NAME            = 'MSGR_MDIS_BASE'
      FRAME_-236880_CLASS           = 4
      FRAME_-236880_CLASS_ID        = -236880
      FRAME_-236880_CENTER          = -236
      TKFRAME_-236880_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236880_SPEC          = 'MATRIX'
      TKFRAME_-236880_MATRIX        = ( 0.002002115198661447
                                       -9.873433017452157E-4
                                        0.9999975083408637
                                       -1.5279882434764006E-4
                                        0.999999500598698
                                        9.876511904033615E-4
                                       -0.9999979840915934
                                       -1.5477583508433595E-4
                                        0.0020019633339067866 )

   [28] gives further refinement to the matrix aligning the MDIS base to the
   spacecraft. The frame definition given above is retained in text only for
   historical purposes and is not loaded into the kernel pool. The update in
   [28] was estimated from the analysis of three pivot axis calibration
   sequences acquired with the WAC in 2007 and 2008. These sequences moved
   the pivot to seven different angles spanning the standard science imaging
   range of motion. The total change in the base mounting alignment of the
   camera as a result of this analysis was approximately 140 microradians in
   total. The update is given in the frame definition below.

      FRAME_MSGR_MDIS_BASE          = -236880
      FRAME_-236880_NAME            = 'MSGR_MDIS_BASE'
      FRAME_-236880_CLASS           = 4
      FRAME_-236880_CLASS_ID        = -236880
      FRAME_-236880_CENTER          = -236
      TKFRAME_-236880_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236880_SPEC          = 'MATRIX'
      TKFRAME_-236880_MATRIX        = ( 2.001278635903917e-003,
                                       -9.898172501274741e-004,
                                        9.999975075697103e-001,
                                       -1.354727542646206e-005,
                                        9.999995100102231e-001,
                                        9.898463441230367e-004,
                                       -9.999979973481410e-001,
                                       -1.552820000222388e-005,
                                        2.001264245970936e-003 )

   [29] describes an approximately 400 microradian shift in the alignment 
   of MDIS that happened at or around Mercury orbit insertion.  While 
   calibration images indicate the camera appears to continue drifting,
   the following matrix centers the base mounting alignment about the 
   average of the currently observed drift post orbit insertion:

      \begindata

      FRAME_MSGR_MDIS_BASE          = -236880
      FRAME_-236880_NAME            = 'MSGR_MDIS_BASE'
      FRAME_-236880_CLASS           = 4
      FRAME_-236880_CLASS_ID        = -236880
      FRAME_-236880_CENTER          = -236
      TKFRAME_-236880_RELATIVE      = 'MSGR_SPACECRAFT'
      TKFRAME_-236880_SPEC          = 'MATRIX'
      TKFRAME_-236880_MATRIX        = ( 1.915552505434404e-003
                                       -1.406138680413779e-003
                                        9.999971767123197e-001
                                       -1.008730395465398e-004
                                        9.999990060234849e-001
                                        1.406334480835401e-003
                                       -9.999981602399219e-001
                                       -1.035666622911730e-004
                                        1.915408759937454e-003 )
      \begintext

   The articulation frame, which rotates the base frame about its Z-axis
   is C-kernel based.  (See [1] for details):

   MDIS Articulation Frame (MSGR_MDIS_ART):

      \begindata

      FRAME_MSGR_MDIS_ART         = -236890
      FRAME_-236890_NAME          = 'MSGR_MDIS_ART'
      FRAME_-236890_CLASS         = 3
      FRAME_-236890_CLASS_ID      = -236890
      FRAME_-236890_CENTER        = -236
      CK_-236890_SCLK             = -236
      CK_-236890_SPK              = -236

      \begintext


   An additional C-kernel based frame, MSGR_MDIS_ART_RAW, is inserted between
   frames MSGR_MDIS_ART and MSGR_MDIS_BASE when the calibrated pivot angle is
   used [26]. The rotation from MSGR_MDIS_ART to MSGR_MDIS_ART_RAW contains the
   nonlinear element of the pivot motion. The rotation from MSGR_MDIS_ART_RAW
   to MSGR_MDIS_BASE contains the linear, or raw, element. The frame
   MSGR_MDIS_ART_RAW rotates the base frame about its Z-axis. (See [1] for
   details):

   MDIS Calibrated Articulation Frame (MSGR_MDIS_ART_RAW):

      \begindata

      FRAME_MSGR_MDIS_ART_RAW     = -236891
      FRAME_-236891_NAME          = 'MSGR_MDIS_ART_RAW'
      FRAME_-236891_CLASS         = 3
      FRAME_-236891_CLASS_ID      = -236891
      FRAME_-236891_CENTER        = -236
      CK_-236891_SCLK             = -236
      CK_-236891_SPK              = -236

      \begintext


   Another C-kernel based frame, MSGR_MDIS_ART_CAL, has been inserted between the
   frames MSGR_MDIS_ART and MSGR_MDIS_BASE.  This frame accounts for an observed,
   gradual temporal drift in the resolver output since MOI.   

   MDIS Temporal Drift Calibration (MSGR_MDIS_ART_CAL):

      \begindata

      FRAME_MSGR_MDIS_ART_CAL     = -236892
      FRAME_-236892_NAME          = 'MSGR_MDIS_ART_CAL'
      FRAME_-236892_CLASS         = 3
      FRAME_-236892_CLASS_ID      = -236892
      FRAME_-236892_CENTER        = -236
      CK_-236892_SCLK             = -236
      CK_-236892_SPK              = -236

      \begintext


   MDIS Camera Frames:

   MDIS consists of two distinct cameras, a wide angle camera (WAC) with
   12 filters and a narrow angle camera (NAC).  The following frame entries
   define a frame for each filter of the WAC and the NAC itself.
   Again, from [14]:

   The WAC and NAC detectors are mounted in a 180-degree rotation or
   "flipped" orientation about their respective boresights.  The diagram
   below illustrates the CCD relative orientation as viewed when looking
   out the boresight of the camera:


   (+Xwac) columns ^                     +---Active Region of CCD
                   |                     |
                   |         ________ ___V____
                   +--->    |        |        |
            (+Ywac) lines   | Memory |        |  WAC      * - denotes (0,0)
                            |  Zone  |        |               pixel location
                     _______|________|*_______|_______
              <---- |_______________ _________________|
        Articulation        |       *|        |
            Axis            |        | Memory |        lines (+Ynac)
                            |        |  Zone  |  NAC     <---+
                            |________|________|              |
                                 ^                           |
                                 |                           V columns (+Xnac)
          Active Region of CCD---+


   The following diagram illustrates the nominal alignment of the gimbal
   axes frame relative to the camera housing.

                                +X
                               ^  art
                               |
                               |
                               |
                          NAC  |  WAC
                          \  / | \   /
                         __\/__|__\ /_
                        |      |      |
                        |      |      |
                        |      +----------> +Y
                        |             |       art
                        |_____________|


   Then we have the following nominal rotation matrix that rotates vectors in
   the WAC base frame to the MDIS articulation frame:

      [     ]   [  0  0  1 ]
      [ ROT ] = [ -1  0  0 ]
      [     ]   [  0 -1  0 ]

   In the WAC coordinate system, lines are measured along the Y axis and
   columns along the X axis.

   MDIS Wide Angle Camera Frame (MGSR_MDIS_WAC):

   The WAC frame is the frame to which all other MDIS frames are to be
   tied.  The morphology filter (filter #1) will be used to image Mercury
   and as such defines the primary MDIS frame.  The MSGR_MDIS_WAC_FILTER1
   frame is simply a fixed offset, identity, alias to the MSGR_MDIS_WAC
   frame.

      FRAME_MSGR_MDIS_WAC           = -236800
      FRAME_-236800_NAME            = 'MSGR_MDIS_WAC'
      FRAME_-236800_CLASS           = 4
      FRAME_-236800_CLASS_ID        = -236800
      FRAME_-236800_CENTER          = -236
      TKFRAME_-236800_RELATIVE      = 'MSGR_MDIS_ART'
      TKFRAME_-236800_SPEC          = 'MATRIX'
      TKFRAME_-236800_MATRIX        = ( 0.0
                                       -1.0
                                        0.0
                                        0.0
                                        0.0
                                       -1.0
                                        1.0
                                        0.0
                                        0.0 )

   [21] gives us an estimated matrix for the MDIS WAC frame based on the
   analysis of images of star fields. [22] gives a correction to the matrices
   in [21]. The above nominal definition remains for reference but will not be
   loaded into the SPICE kernel pool. The estimated matrix for the WAC frame
   is effectively the clear filter (MSGR_MDIS_WAC_FILTER2) alignment. The
   estimated matrix from [22] taking vectors from the WAC frame to the
   articulation frame is

      [     ]
      [ ROT ] =
      [     ]

      [  6.3259799163338534E-6 -6.227457906745934E-4  0.9999998060738124   ]
      [ -0.9999484094198974    -0.010157681754071668  0.0                  ]
      [  0.010157679784231172  -0.9999482155037147   -6.227779201487688E-4 ]

   Which results in the following frame definition:

      \begindata

      FRAME_MSGR_MDIS_WAC           = -236800
      FRAME_-236800_NAME            = 'MSGR_MDIS_WAC'
      FRAME_-236800_CLASS           = 4
      FRAME_-236800_CLASS_ID        = -236800
      FRAME_-236800_CENTER          = -236
      TKFRAME_-236800_RELATIVE      = 'MSGR_MDIS_ART'
      TKFRAME_-236800_SPEC          = 'MATRIX'
      TKFRAME_-236800_MATRIX        = ( 6.3259799163338534E-6
                                       -0.9999484094198974
                                        0.010157679784231172
                                       -6.227457906745934E-4
                                       -0.010157681754071668
                                       -0.9999482155037147
                                        0.9999998060738124
                                        0.0
                                       -6.227779201487688E-4 )

      \begintext


   The NAC is aligned relative to the WAC.  It is nominally oriented
   "upside" down from the WAC, which results in the following matrix
   that rotates vectors from the NAC frame to the WAC:

      [     ]   [ -1  0  0 ]
      [ ROT ] = [  0 -1  0 ]
      [     ]   [  0  0  1 ]

   As with the WAC coordinate system, lines are measured along the Y axis
   and columns along the X axis.

   MDIS Narrow Angle Camera Frame (MGSR_MDIS_NAC):

      FRAME_MSGR_MDIS_NAC           = -236820
      FRAME_-236820_NAME            = 'MSGR_MDIS_NAC'
      FRAME_-236820_CLASS           = 4
      FRAME_-236820_CLASS_ID        = -236820
      FRAME_-236820_CENTER          = -236
      TKFRAME_-236820_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236820_SPEC          = 'MATRIX'
      TKFRAME_-236820_MATRIX        = (-1.0
                                        0.0
                                        0.0
                                        0.0
                                       -1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

   [21] gives us an estimated matrix for the MDIS NAC frame based on the
   analysis of images of star fields. [22] gives a correction to the matrices
   in [21]. The above nominal definition remains for reference but will not be
   loaded into the SPICE kernel pool. The estimated matrix taking vectors from
   [22] taking vectors from the NAC frame to the WAC frame is

      [     ]
      [ ROT ] =
      [     ]

      [ -0.9998325405251587     0.01821119615306077   0.0018009002419674593 ]
      [ -0.018211309182972833  -0.9998341592816491   -4.63831867058139E-5   ]
      [  0.0017997568860663496 -7.917217051588668E-5  0.9999983773021427    ]
      
   Which results in the following frame definition:

      FRAME_MSGR_MDIS_NAC           = -236820
      FRAME_-236820_NAME            = 'MSGR_MDIS_NAC'
      FRAME_-236820_CLASS           = 4
      FRAME_-236820_CLASS_ID        = -236820
      FRAME_-236820_CENTER          = -236
      TKFRAME_-236820_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236820_SPEC          = 'MATRIX'
      TKFRAME_-236820_MATRIX        = (-0.9998325405251587
                                       -0.018211309182972833
                                        0.0017997568860663496
                                        0.01821119615306077
                                       -0.9998341592816491
                                       -7.917217051588668E-5
                                        0.0018009002419674593
                                       -4.63831867058139E-5
                                        0.9999983773021427 )

   [28] gives further refinement to the matrix aligning the NAC to the WAC.
   The frame definition given above is retained in text only for historical
   purposes and is not loaded into the kernel pool. The update in [28] was
   determined from the analysis of two 5x5 NAC mosaics of the Pleiades with
   companion WAC images. The data were reduced to produce an updated
   alignment that shifts the NAC boresight a few NAC pixels relative to the
   WAC boresight and corrects for a few hundred microradian discrepancy in
   the twist angle between the two detectors. The update is given in the
   frame definition below.

      FRAME_MSGR_MDIS_NAC           = -236820
      FRAME_-236820_NAME            = 'MSGR_MDIS_NAC'
      FRAME_-236820_CLASS           = 4
      FRAME_-236820_CLASS_ID        = -236820
      FRAME_-236820_CENTER          = -236
      TKFRAME_-236820_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236820_SPEC          = 'MATRIX'
      TKFRAME_-236820_MATRIX        = (-9.998230400164143e-001,
                                       -1.872273552438176e-002,
                                        1.829706757803473e-003,
                                        1.872265195266730e-002,
                                       -9.998247138177479e-001,
                                       -6.279425093888615e-005,
                                        1.830561715644037e-003,
                                       -2.852617606747299e-005,
                                        9.999983241136270e-001 )

   [30] provides yet another refinement to the matrix aligning the NAC to
   the WAC.  As with previous versions, the frame definition above is 
   retained for historical/reference purposes alone and is not parsed into
   the kernel pool.  The following alignment matrix is the result of a
   detailed analysis of all NAC star calibration images collected to date.
   As [30] documents, there is a temporal drift, but this single matrix
   centers the alignment around an apparent shift that occurred sometime
   around MOI.

      \begindata

      FRAME_MSGR_MDIS_NAC           = -236820
      FRAME_-236820_NAME            = 'MSGR_MDIS_NAC'
      FRAME_-236820_CLASS           = 4
      FRAME_-236820_CLASS_ID        = -236820
      FRAME_-236820_CENTER          = -236
      TKFRAME_-236820_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236820_SPEC          = 'MATRIX'
      TKFRAME_-236820_MATRIX        = (-9.9982154874087930e-001,
                                       -1.8816063038872965e-002,
                                        1.6812034696471194e-003,
                                        1.8816101953593675e-002,
                                       -9.9982296145543590e-001,
                                        7.3316825722437500e-006,
                                        1.6807678784302770e-003,
                                        3.8964070113872890e-005,
                                        9.9999858674957140e-001 )

      \begintext

      
   The WAC has a total of 12 different filters that may be utilized.  Each
   filter has it's own frame definition that is offset from MSGR_MDIS_WAC.
   The nominal alignments follow:

   MDIS Wide Angle Camera Filter #1 Frame (MSGR_MDIS_WAC_FILTER1):

      The MSGR_MDIS_WAC frame is defined via filter #1.  So this frame
      definition is simply an identity offset providing an alias to the
      MSGR_MDIS_WAC frame root.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER1   = -236801
      FRAME_-236801_NAME            = 'MSGR_MDIS_WAC_FILTER1'
      FRAME_-236801_CLASS           = 4
      FRAME_-236801_CLASS_ID        = -236801
      FRAME_-236801_CENTER          = -236
      TKFRAME_-236801_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236801_SPEC          = 'MATRIX'
      TKFRAME_-236801_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #2 Frame (MSGR_MDIS_WAC_FILTER2):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER2   = -236802
      FRAME_-236802_NAME            = 'MSGR_MDIS_WAC_FILTER2'
      FRAME_-236802_CLASS           = 4
      FRAME_-236802_CLASS_ID        = -236802
      FRAME_-236802_CENTER          = -236
      TKFRAME_-236802_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236802_SPEC          = 'MATRIX'
      TKFRAME_-236802_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #3 Frame (MSGR_MDIS_WAC_FILTER3):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER3   = -236803
      FRAME_-236803_NAME            = 'MSGR_MDIS_WAC_FILTER3'
      FRAME_-236803_CLASS           = 4
      FRAME_-236803_CLASS_ID        = -236803
      FRAME_-236803_CENTER          = -236
      TKFRAME_-236803_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236803_SPEC          = 'MATRIX'
      TKFRAME_-236803_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #4 Frame (MSGR_MDIS_WAC_FILTER4):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER4   = -236804
      FRAME_-236804_NAME            = 'MSGR_MDIS_WAC_FILTER4'
      FRAME_-236804_CLASS           = 4
      FRAME_-236804_CLASS_ID        = -236804
      FRAME_-236804_CENTER          = -236
      TKFRAME_-236804_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236804_SPEC          = 'MATRIX'
      TKFRAME_-236804_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #5 Frame (MSGR_MDIS_WAC_FILTER5):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER5   = -236805
      FRAME_-236805_NAME            = 'MSGR_MDIS_WAC_FILTER5'
      FRAME_-236805_CLASS           = 4
      FRAME_-236805_CLASS_ID        = -236805
      FRAME_-236805_CENTER          = -236
      TKFRAME_-236805_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236805_SPEC          = 'MATRIX'
      TKFRAME_-236805_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #6 Frame (MSGR_MDIS_WAC_FILTER6):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER6   = -236806
      FRAME_-236806_NAME            = 'MSGR_MDIS_WAC_FILTER6'
      FRAME_-236806_CLASS           = 4
      FRAME_-236806_CLASS_ID        = -236806
      FRAME_-236806_CENTER          = -236
      TKFRAME_-236806_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236806_SPEC          = 'MATRIX'
      TKFRAME_-236806_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #7 Frame (MSGR_MDIS_WAC_FILTER7):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER7   = -236807
      FRAME_-236807_NAME            = 'MSGR_MDIS_WAC_FILTER7'
      FRAME_-236807_CLASS           = 4
      FRAME_-236807_CLASS_ID        = -236807
      FRAME_-236807_CENTER          = -236
      TKFRAME_-236807_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236807_SPEC          = 'MATRIX'
      TKFRAME_-236807_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #8 Frame (MSGR_MDIS_WAC_FILTER8):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER8   = -236808
      FRAME_-236808_NAME            = 'MSGR_MDIS_WAC_FILTER8'
      FRAME_-236808_CLASS           = 4
      FRAME_-236808_CLASS_ID        = -236808
      FRAME_-236808_CENTER          = -236
      TKFRAME_-236808_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236808_SPEC          = 'MATRIX'
      TKFRAME_-236808_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #9 Frame (MSGR_MDIS_WAC_FILTER9):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER9   = -236809
      FRAME_-236809_NAME            = 'MSGR_MDIS_WAC_FILTER9'
      FRAME_-236809_CLASS           = 4
      FRAME_-236809_CLASS_ID        = -236809
      FRAME_-236809_CENTER          = -236
      TKFRAME_-236809_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236809_SPEC          = 'MATRIX'
      TKFRAME_-236809_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #10 Frame (MSGR_MDIS_WAC_FILTER10):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER10  = -236810
      FRAME_-236810_NAME            = 'MSGR_MDIS_WAC_FILTER10'
      FRAME_-236810_CLASS           = 4
      FRAME_-236810_CLASS_ID        = -236810
      FRAME_-236810_CENTER          = -236
      TKFRAME_-236810_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236810_SPEC          = 'MATRIX'
      TKFRAME_-236810_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #11 Frame (MSGR_MDIS_WAC_FILTER11):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER11  = -236811
      FRAME_-236811_NAME            = 'MSGR_MDIS_WAC_FILTER11'
      FRAME_-236811_CLASS           = 4
      FRAME_-236811_CLASS_ID        = -236811
      FRAME_-236811_CENTER          = -236
      TKFRAME_-236811_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236811_SPEC          = 'MATRIX'
      TKFRAME_-236811_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext

   MDIS Wide Angle Camera Filter #12 Frame (MSGR_MDIS_WAC_FILTER12):

      The nominal alignment of this frame is defined as an alias to
      the MSGR_MDIS_WAC frame.  As the in-flight calibrations are
      performed, updates to this filter's relative alignment to the
      MSGR_MDIS_WAC frame may be provided.

      \begindata

      FRAME_MSGR_MDIS_WAC_FILTER12  = -236812
      FRAME_-236812_NAME            = 'MSGR_MDIS_WAC_FILTER12'
      FRAME_-236812_CLASS           = 4
      FRAME_-236812_CLASS_ID        = -236812
      FRAME_-236812_CENTER          = -236
      TKFRAME_-236812_RELATIVE      = 'MSGR_MDIS_WAC'
      TKFRAME_-236812_SPEC          = 'MATRIX'
      TKFRAME_-236812_MATRIX        = ( 1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0
                                        0.0
                                        0.0
                                        0.0
                                        1.0 )

      \begintext
