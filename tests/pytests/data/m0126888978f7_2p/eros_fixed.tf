KPL/FK

EROS Body-Fixed Frame Definition Kernel
===============================================================================

   This frame kernel contains asteroid EROS body-fixed frame definition.


Version and Date
--------------------------------------------------------

   Version 1.0 -- June 18, 2002 -- Boris Semenov

      Initial release.


References
--------------------------------------------------------

   1. ``Frames Required Reading''

   2. ``PCK Required Reading''


Contact Information
--------------------------------------------------------

   Boris V. Semenov, NAIF/JPL, (818)-354-8136, bsemenov@spice.jpl.nasa.gov


Implementation Notes
--------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must `load' the kernel, normally during program
   initialization. Loading the kernel associates data items with their
   names in a data structure called the `kernel pool'. The SPICELIB
   routine FURNSH loads a kernel file into the pool as shown below.

      CALL FURNSH ( frame_kernel_name )

   This file was created and may be updated with a text editor or word
   processor.


EROS Body-Fixed Frame
--------------------------------------------------------

   This kernel file defines asteroid EROS body-frame, EROS_FIXED, with the 
   frame ID code 2000433 in the same as any other PCK frame:
   
      *  +Z along asteroid's North pole;
      
      *  +X along asteroid's prime meridian;
      
      *  +Y complements to the right hand frame;
      
      *  the origin of this frame is at the center of the asteroid ellipsoid.
      
   As for any PCK frame orientation of this frame is computed by evaluating 
   corresponding rotation constants provided in a PCK file.

   \begindata

      FRAME_EROS_FIXED       =  2000433
      FRAME_2000433_NAME     = 'EROS_FIXED'
      FRAME_2000433_CLASS    =  2
      FRAME_2000433_CLASS_ID =  2000433
      FRAME_2000433_CENTER   =  2000433

      OBJECT_2000433_FRAME   = 'EROS_FIXED'

   \begintext

