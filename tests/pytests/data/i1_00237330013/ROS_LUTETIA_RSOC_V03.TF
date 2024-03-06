KPL/FK

Asteroid 21 Lutetia Frames kernel
=================================

 
Version and Date
----------------

     Version 0.3 -- November 7, 2011 -- J. Vazquez, PSA, SRE-OS/ESA
         
         Added explanation as to how to use the kernel.

     Version 0.2 -- October 6, 2011 -- J. Vazquez, PSA, SRE-OS/ESA
     
         Corrected the rotation angle in the frame.

     Version 0.1 -- September 20, 2011 -- J. Vazquez, PSA, SRE-OS/ESA
     

References
----------

     1) Rotational Parameters of (21) Lutetia, L. Jorda & J.-B. Vincent

Description
-----------

     This kernel contains a frame for the Asteroid Lutetia, defined as:
     
       - The Z axis of the frame is the rotational axis of the asteroid.
       - The X axis contains the zero-longitude point, defined in [1].

     It implements the reference frame defined in [1], whose prime meridian
     contains the 'Lauriacum' crater.
     
Usage
-----

     This kernel must be used along with the following kernels:
     
       - ROS_Vxx.TF: generic Rosetta frames definition kernel.
       - ROS_LUTETIA_RSOC_Vxx.TPC: kernel that defines the Lutetia body-fixed
         frame.
     
     Once these frames are loaded, in order to compute positions and/or any
     other data relative to the frame defined above (i.e., the prime meridian
     of Lutetia crossing the X axis of the frame) the frame 'ROS_LUTETIA' must
     be specified as the frame of reference.
     
     


\begindata

      FRAME_ROS_LUTETIA                =  -2260021
      FRAME_-2260021_NAME               = 'ROS_LUTETIA'
      FRAME_-2260021_CLASS              =  4
      FRAME_-2260021_CLASS_ID           =  -2260021
      FRAME_-2260021_CENTER             =  2000021
      TKFRAME_-2260021_RELATIVE         = 'LUTETIA_FIXED'
      TKFRAME_-2260021_SPEC             = 'ANGLES'
      TKFRAME_-2260021_UNITS            = 'DEGREES'
      TKFRAME_-2260021_ANGLES           = (   0.000,  0.000,   164.700 )
      TKFRAME_-2260021_AXES             = (   1,      2,         3     )


\begintext
