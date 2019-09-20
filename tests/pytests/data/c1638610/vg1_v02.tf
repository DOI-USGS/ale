KPL/FK
 
Voyager-1 Frame Kernel
===========================================================================
 
     This file contains various frame definitions for Voyager-1, related
     to the spacecraft body and the scan platform.
 
 
Version and Date
--------------------------------------------------------

     Version 2.0 -- 27 November 2000 -- Jeff Bytof
     
        Changed instrument base frame to VG1_SCAN_PLATFORM. 


     Version 1.1 --  8 March 2000 -- Jeff Bytof
     
        Added the spacecraft high-gain antenna frame, FRAME_VG1_HGA.
        
        Added -100 to all scan platform instrument ID codes, to 
        associate them to the scan platform ID code of -31100.   


     Version 1.0 --  4 November 1999 -- Jeff Bytof
     
        Initial Release.
 

References
--------------------------------------------------------

     (1) Frames Required Reading (NAIF document).
 
     (2) C Kernel Required Reading (NAIF document).

 
Implementation Notes
--------------------------------------------------------
  
     This file is used by the SPICE system as follows: programs that make
     use of this frame kernel must `load' the kernel, normally during 
     program initialization. Loading the kernel associates data items with 
     their names in a data structure called the `kernel pool'. The SPICELIB
     routine LDPOOL loads a kernel file into the pool as shown below.
 
 
        CALL LDPOOL ( kernel_name )
 
 
     In order for a program or subroutine to extract data from the pool,
     the SPICELIB routines GDPOOL and GIPOOL are used. See [3] for more 
     details.
 
     This file was created and may be updated with a text editor or word
     processor.

 
Naming Conventions
--------------------------------------------------------
 
     All names referencing values in this frame kernel start with the
     characters `FRAME', 'CK' or `TKFRAME' followed by the Voyager-1 
     spacecraft bus ID number (-31000) added to the instrument or 
     alternate frame index number.  
 
     The remainder of the name is an underscore character followed by the
     unique name of the data item. For example, the Voyager-1 cone/clock 
     offsets relative to the spacecraft frame, given as three Euler angles, 
     are specified using two items:
 
        TKFRAME_-31200_ANGLES    = (  55.0, 0.0, 180.0 )
        TKFRAME_-31200_AXES      = (     3,   2,     1 )
 
     The upper bound on the length of the name of any data item is 32
     characters.
 
     If the same item is included in more then one file, or if the same
     item appears more than once within a single file, the latest value
     supersedes any earlier values.
 
 
TKFRAME Keyword Description
--------------------------------------------------------
 
     This section describes the TKFRAME keywords. The rotational offsets 
     can be given as three angles -- ROLL, PITCH and YAW, from which 
     a rotation matrix can be constructed that will transform the components of 
     a vector expressed in the spacecraft frame to components expressed in 
     the antenna fixed frame. For example, if x, y and z are the components of 
     a vector expressed in the spacecraft frame, X, Y and Z will be the components 
     of the same vector expressed in the antenna fixed frame: 
 
 
        [ X ]    [     ]  [ x ]
        | Y |  = | ROT |  | y |
        [ Z ]    [     ]  [ z ]
 
 
     where ROT is the rotation matrix constructed from the rotation angles
     as follows:
     
        [     ]   [     ]  [       ]  [      ]
        [ ROT ] = [ YAW ]  [ PITCH ]  [ ROLL ]
        [     ]   [     ]  [       ]  [      ]
                         Z          Y         X
                         
     where each of three matrixes on the right side represent a coordinate
     frame rotation by the given angle around the indicated axis. See the
     SPICELIB routine EUL2M for more information about constructing
     a rotation matrix from a set of rotation angles.
 
     Following are some examples of use of the TKFRAME keywords:
     
     The keyword that indicates which frame the axis rotations are
     referred to is:
     
        TKFRAME_-31200_RELATIVE  = 'VG1_SC_BUS'

            
     The keyword TKFRAME_-91000_ANGLES contain these values, in radians, 
     in the following order:
     
                                    ``ROLL''  ``PITCH''  ``YAW''       
        TKFRAME_-31200_ANGLES    = (  55.0,      0.0,     180.0 )

        
     The keyword TKFRAME_-31200_AXES contains integer codes of the 
     corresponding axes of rotations (1 -- X, 2 -- Y, 3 -- Z).
     
        TKFRAME_-31200_AXES      = (  3,   2,   1 )

     
     The keyword TKFRAME_-31200_UNITS gives the units of the angles.
     
        TKFRAME_-31202_UNITS     = 'DEGREES'
 

 


Spacecraft Bus
--------------------------------------------------------
 
     The following data represents the basic spacecraft bus, and
     the scan platform.  Spacecraft bus attitude with respect to an inertial 
     frame is provided by a C kernel (see [2] for more information).
     Scan platform orientation with respect to an inertial frame
     is also provided by a C-kernel.  Each instrument mounted
     on the scan platform may have measureable offsets from the 
     nominal scan platform orientation.  The Narrow Angle camera
     has been chosen as representative of the scan platform orientation.   

\begindata

   FRAME_VG1_SC_BUS         = -31000
   FRAME_-31000_NAME        = 'VG1_SC_BUS'
   FRAME_-31000_CLASS       = 3
   FRAME_-31000_CLASS_ID    = -31000
   FRAME_-31000_CENTER      = -31
   CK_-31000_SCLK           = -31
   CK_-31000_SPK            = -31

   FRAME_VG1_SCAN_PLATFORM  = -31100
   FRAME_-31100_NAME        = 'VG1_SCAN_PLATFORM'
   FRAME_-31100_CLASS       = 3
   FRAME_-31100_CLASS_ID    = -31100
   FRAME_-31100_CENTER      = -31
   CK_-31100_SCLK           = -31
   CK_-31100_SPK            = -31

\begintext 
 

Frame Definitions
--------------------------------------------------------

     Here are the frame definitions for Voyager 1.  These are 
     utilized by SPICE's FRAMES subsystem to provide automatic state 
     transformations to/from the various frames.  Note that SPICE toolkit 
     version N0047 or higher is required to use fixed-offset frames.
   
     Note that angles in the frame definitions are specified for the "from
     instrument to base (relative to) frame" transformation.

      
     Spacecraft body-fixed reference frames:
        

\begindata

        FRAME_VG1_CONE_CLOCK     = -31200
        FRAME_-31200_NAME        = 'VG1_CONE_CLOCK'
        FRAME_-31200_CLASS       = 4
        FRAME_-31200_CLASS_ID    = -31200
        FRAME_-31200_CENTER      = -31
        TKFRAME_-31200_SPEC      = 'ANGLES'
        TKFRAME_-31200_RELATIVE  = 'VG1_SC_BUS'
        TKFRAME_-31200_ANGLES    = (  55.0, 0.0, 180.0 )
        TKFRAME_-31200_AXES      = (     3,   2,     1 )
        TKFRAME_-31200_UNITS     = 'DEGREES'
        
        FRAME_VG1_AZ_EL          = -31300
        FRAME_-31300_NAME        = 'VG1_AZ_EL'
        FRAME_-31300_CLASS       = 4
        FRAME_-31300_CLASS_ID    = -31300
        FRAME_-31300_CENTER      = -31
        TKFRAME_-31300_SPEC      = 'ANGLES'
        TKFRAME_-31300_RELATIVE  = 'VG1_SC_BUS'
        TKFRAME_-31300_ANGLES    = ( -173.0,      0.0,   180.0 )
        TKFRAME_-31300_AXES      = (      1,        2,       3 )
        TKFRAME_-31300_UNITS     = 'DEGREES'
        
\begintext

        The boresight of the antenna is the +Z axis.
        
\begindata
              
        FRAME_VG1_HGA            = -31400
        FRAME_-31400_NAME        = 'VG1_HGA'
        FRAME_-31400_CLASS       = 4
        FRAME_-31400_CLASS_ID    = -31400
        FRAME_-31400_CENTER      = -31
        TKFRAME_-31400_SPEC      = 'ANGLES'
        TKFRAME_-31400_RELATIVE  = 'VG1_SC_BUS'
        TKFRAME_-31400_ANGLES    = ( 180.0,      0.0,     0.0 )
        TKFRAME_-31400_AXES      = (     1,        2,       3 )
        TKFRAME_-31400_UNITS     = 'DEGREES'

\begintext

        Voyager-1 scan platform instrument frame definitions.
  
        The Euler angles of rotation given below are 
        referred to the ISSNA frame of reference which
        in this case is considered the equivalent of the
        scan platform's frame.
        
        Boresights are +Z in each instrument frame.
        
 
\begindata


        FRAME_VG1_ISSNA          = -31101
        FRAME_-31101_NAME        = 'VG1_ISSNA'
        FRAME_-31101_CLASS       = 4
        FRAME_-31101_CLASS_ID    = -31101
        FRAME_-31101_CENTER      = -31
        TKFRAME_-31101_SPEC      = 'ANGLES'
        TKFRAME_-31101_RELATIVE  = 'VG1_SCAN_PLATFORM'
        TKFRAME_-31101_ANGLES    = ( 0.0,  0.0,  0.0 )
        TKFRAME_-31101_AXES      = (   1,    2,    3 )
        TKFRAME_-31101_UNITS     = 'DEGREES'


        FRAME_VG1_ISSWA          = -31102
        FRAME_-31102_NAME        = 'VG1_ISSWA'
        FRAME_-31102_CLASS       = 4
        FRAME_-31102_CLASS_ID    = -31102
        FRAME_-31102_CENTER      = -31
        TKFRAME_-31102_SPEC      = 'ANGLES'
        TKFRAME_-31102_RELATIVE  = 'VG1_SCAN_PLATFORM'
        TKFRAME_-31102_ANGLES    = ( 0.275, -0.0247, 0.0315 )
        TKFRAME_-31102_AXES      = (     3,       1,      2 )
        TKFRAME_-31102_UNITS     = 'DEGREES'
         

        FRAME_VG1_PPS            = -31103
        FRAME_-31103_NAME        = 'VG1_PPS'
        FRAME_-31103_CLASS       = 4
        FRAME_-31103_CLASS_ID    = -31103
        FRAME_-31103_CENTER      = -31
        TKFRAME_-31103_SPEC      = 'ANGLES'
        TKFRAME_-31103_RELATIVE  = 'VG1_SCAN_PLATFORM'
        TKFRAME_-31103_ANGLES    = ( 0.0,  0.0,  -0.034 )
        TKFRAME_-31103_AXES      = (   3,    1,       2 )
        TKFRAME_-31103_UNITS     = 'DEGREES'
        

        FRAME_VG1_UVS            = -31104
        FRAME_-31104_NAME        = 'VG1_UVS'
        FRAME_-31104_CLASS       = 4
        FRAME_-31104_CLASS_ID    = -31104
        FRAME_-31104_CENTER      = -31
        TKFRAME_-31104_SPEC      = 'ANGLES'
        TKFRAME_-31104_RELATIVE  = 'VG1_SCAN_PLATFORM'
        TKFRAME_-31104_ANGLES    = ( 0.0,  0.030,  0.010 )
        TKFRAME_-31104_AXES      = (   3,    1,        2 )
        TKFRAME_-31104_UNITS     = 'DEGREES'


        FRAME_VG1_UVSOCC         = -31105
        FRAME_-31105_NAME        = 'VG1_UVSOCC'
        FRAME_-31105_CLASS       = 4
        FRAME_-31105_CLASS_ID    = -31105
        FRAME_-31105_CENTER      = -31
        TKFRAME_-31105_SPEC      = 'ANGLES'
        TKFRAME_-31105_RELATIVE  = 'VG1_SCAN_PLATFORM'
        TKFRAME_-31105_ANGLES    = ( 0.0, -0.00368, -19.5 )
        TKFRAME_-31105_AXES      = (   3,        1,     2 )
        TKFRAME_-31105_UNITS     = 'DEGREES'
        

        FRAME_VG1_IRIS           = -31106
        FRAME_-31106_NAME        = 'VG1_IRIS'
        FRAME_-31106_CLASS       = 4
        FRAME_-31106_CLASS_ID    = -31106
        FRAME_-31106_CENTER      = -31
        TKFRAME_-31106_SPEC      = 'ANGLES'
        TKFRAME_-31106_RELATIVE  = 'VG1_SCAN_PLATFORM'
        TKFRAME_-31106_ANGLES    = ( 0.0, -0.020, 0.024 )
        TKFRAME_-31106_AXES      = (   3,      1,     2 )
        TKFRAME_-31106_UNITS     = 'DEGREES'
        

        FRAME_VG1_IRISOCC        = -31107
        FRAME_-31107_NAME        = 'VG1_IRISOCC'
        FRAME_-31107_CLASS       = 4
        FRAME_-31107_CLASS_ID    = -31107
        FRAME_-31107_CENTER      = -31
        TKFRAME_-31107_SPEC      = 'ANGLES'
        TKFRAME_-31107_RELATIVE  = 'VG1_SCAN_PLATFORM'
        TKFRAME_-31107_ANGLES    = ( 0.0, -0.000591, -20.8 )
        TKFRAME_-31107_AXES      = (   3,         1,     2 )
        TKFRAME_-31107_UNITS     = 'DEGREES'

\begintext        
 
