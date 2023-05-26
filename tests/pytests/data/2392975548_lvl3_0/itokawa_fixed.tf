KPL/FK

\beginlabel
PDS_VERSION_ID               = PDS3
RECORD_TYPE                  = STREAM
RECORD_BYTES                 = "N/A"
^SPICE_KERNEL                = "itokawa_fixed.tf"
MISSION_NAME                 = HAYABUSA
SPACECRAFT_NAME              = HAYABUSA
DATA_SET_ID                  = "HAY-A-SPICE-6-V1.0"
KERNEL_TYPE_ID               = FK
PRODUCT_ID                   = "itokawa_fixed.tf"
PRODUCT_CREATION_TIME        = 2009-11-30T13:50:41
PRODUCER_ID                  = JAXA
MISSION_PHASE_NAME           = "N/A"
PRODUCT_VERSION_TYPE         = ACTUAL
PLATFORM_OR_MOUNTING_NAME    = "N/A"
START_TIME                   = "N/A"
STOP_TIME                    = "N/A"
SPACECRAFT_CLOCK_START_COUNT = "N/A"
SPACECRAFT_CLOCK_STOP_COUNT  = "N/A"
TARGET_NAME                  = "25143 ITOKAWA"
INSTRUMENT_NAME              = "N/A"
NAIF_INSTRUMENT_ID           = "N/A"
SOURCE_PRODUCT_ID            = "N/A"
NOTE                         = "See comments in the file for details"
OBJECT                       = SPICE_KERNEL
  INTERCHANGE_FORMAT         = ASCII
  KERNEL_TYPE                = FRAMES
  DESCRIPTION                = "SPICE Frames Kernel (FK) file containing
the definition of the asteroid Itokawa body-fixed frame, created by the
Hayabusa Joint Science Team. "
END_OBJECT                   = SPICE_KERNEL
\endlabel


ITOKAWA Body-Fixed Frame Definition Kernel
===========================================================================

   This frame kernel contains asteroid ITOKAWA body-fixed frame definition.

   
Version and Date
--------------------------------------------------------

   Version 1.0 -- May 25, 2004 -- Naru Hirata

      Initial release.


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


ITOKAWA Body-Fixed Frame
--------------------------------------------------------

   This kernel file defines asteroid ITOKAWA body-frame, ITOKAWA_FIXED, with
   the frame ID code 2025143 in the same as any other PCK frame:
   
      *  +Z along asteroid's North pole;
      
      *  +X along asteroid's prime meridian;
      
      *  +Y complements to the right hand frame;
      
      *  the origin of this frame is at the center of the asteroid ellipsoid.
      
   As for any PCK frame orientation of this frame is computed by evaluating 
   corresponding rotation constants provided in a PCK file.

   \begindata

      FRAME_ITOKAWA_FIXED    =  2025143
      FRAME_2025143_NAME     = 'ITOKAWA_FIXED'
      FRAME_2025143_CLASS    =  2
      FRAME_2025143_CLASS_ID =  2025143
      FRAME_2025143_CENTER   =  2025143

      OBJECT_2025143_FRAME   = 'ITOKAWA_FIXED'

   \begintext

