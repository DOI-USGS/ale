KPL/FK

Dawn Frames Kernel for Asteroid Ceres
========================================================================

   This frame kernel contains definition of the body-fixed frame for 
   asteroid Ceres.


Version and Date
========================================================================
 
   Version 0.0 -- May 9, 2005 -- Boris Semenov, NAIF

      Preliminary Version.


References
========================================================================

   1. ``Frames Required Reading''

   2. ``Kernel Pool Required Reading''

   3. ``PC-Kernel Required Reading''


Contact Information
========================================================================

   Boris V. Semenov, NAIF/JPL, (818)-354-8136, boris.semenov@jpl.nasa.gov


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


Dawn Frames
========================================================================

   The following Dawn frames are defined in this kernel file:

           Name                  Relative to           Type       NAIF ID
      ======================  ===================  ============   =======

   Dawn Target frames:
   -------------------
      CERES_FIXED             J2000                PCK            2000001


Dawn Target Frames
========================================================================

   This section of the file contains the body-fixed frame definition
   for one of the Dawn mission targets -- asteroids 1 Ceres.

   A body-fixed frame is defined for Ceres using standard body-fixed,
   PCK-based frame formation rules:
   
      -  +Z axis is toward the North pole;

      -  +X axis is toward the prime meridian;

      -  +Y axis completes the right hand frame;

      -  the origin of this frame is at the center of the body.

   The orientation of this frame is computed by evaluating
   corresponding rotation constants provided in the PCK file(s).

   \begindata

      FRAME_CERES_FIXED                =  2000001
      FRAME_2000001_NAME               = 'CERES_FIXED'
      FRAME_2000001_CLASS              =  2
      FRAME_2000001_CLASS_ID           =  2000001
      FRAME_2000001_CENTER             =  2000001
      OBJECT_2000001_FRAME             = 'CERES_FIXED'

   \begintext

