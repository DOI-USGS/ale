KPL/FK
 
Voyager-2 Frame Kernel
===========================================================================
 
     This file contains various frame definitions for Voyager-2, 
     related to the spacecraft body and the scan platform.
 
  
Version and Date
--------------------------------------------------------

     Version 2   -- 27 Nov. 2000 -- Jeff Bytof

        Changed base frame of instruments to VG2_SCAN_PLATFORM. 

        Eliminated multiple frames for PPS.  That can 
        be handled in the PPS instrument kernel.

 
     Version 1.2 -- 16 Aug. 2000  -- Jeff Bytof
    
        Added additional frame definitions for the PPS instrument 
        because it has multiple fields of view.


     Version 1.1 --  8 March 2000 -- Jeff Bytof
     
        Added the spacecraft high-gain antenna frame, FRAME_VG2_HGA.
        
        Added -100 to all scan platform instrument ID codes, to 
        associate them to the scan platform ID code of -32100.   


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
     characters `FRAME', 'CK' or `TKFRAME' followed by the Voyager-2 
     spacecraft bus ID number (-32000) added to the instrument or 
     alternate frame index number.  
 
     The remainder of the name is an underscore character followed by the
     unique name of the data item. For example, the Voyager-2 cone/clock 
     offsets relative to the spacecraft frame, given as three Euler angles, 
     are specified using two items:
 
        TKFRAME_-32200_ANGLES    = (  55.0, 0.0, 180.0 )
        TKFRAME_-32200_AXES      = (     3,   2,     1 )
 
     The upper bound on the length of the name of any data item is 32
     characters.
 
     If the same item is included in more then one file, or if the same
     item appears more than once within a single file, the latest value
     supersedes any earlier values.
     
 
TKFRAME Keyword Description
--------------------------------------------------------
 
     This section describes the TKFRAME keywords. The rotational offsets 
     can be given as three angles -- ROLL, PITCH and YAW, from which 
     a rotation matrix can be constructed that will transform the components 
     of a vector expressed in the spacecraft frame to components expressed 
     in the antenna fixed frame. For example, if x, y and z are the 
     components of a vector expressed in the spacecraft frame, X, Y and Z 
     will be the components of the same vector expressed in the antenna 
     fixed frame: 
 
 
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
     
        TKFRAME_-32200_RELATIVE  = 'VG2_SC_BUS'

            
     The keyword TKFRAME_-91000_ANGLES contain these values, in radians, 
     in the following order:
     
                                    ``ROLL''  ``PITCH''  ``YAW''       
        TKFRAME_-32200_ANGLES    = (  55.0,      0.0,     180.0 )

        
     The keyword TKFRAME_-32200_AXES contains integer codes of the 
     corresponding axes of rotations (1 -- X, 2 -- Y, 3 -- Z).
     
        TKFRAME_-32200_AXES      = (  3,   2,   1 )

     
     The keyword TKFRAME_-32200_UNITS gives the units of the angles.
     
        TKFRAME_-32202_UNITS     = 'DEGREES'
 


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

   FRAME_VG2_SC_BUS         = -32000
   FRAME_-32000_NAME        = 'VG2_SC_BUS'
   FRAME_-32000_CLASS       = 3
   FRAME_-32000_CLASS_ID    = -32000
   FRAME_-32000_CENTER      = -32
   CK_-32000_SCLK           = -32
   CK_-32000_SPK            = -32

   FRAME_VG2_SCAN_PLATFORM  = -32100
   FRAME_-32100_NAME        = 'VG2_SCAN_PLATFORM'
   FRAME_-32100_CLASS       = 3
   FRAME_-32100_CLASS_ID    = -32100
   FRAME_-32100_CENTER      = -32
   CK_-32100_SCLK           = -32
   CK_-32100_SPK            = -32

\begintext 
 

Frame Definitions
--------------------------------------------------------

     Here are the antenna frame definitions for Voyager 2.  These will be 
     utilized by SPICE's FRAMES subsystem to provide automatic state 
     transformations to/from the Voyager-2 frame.  Note that SPICE toolkit 
     version N0047 or higher is required to use fixed-offset frames.
   
     Note that angles in the frame definitions are specified for the "from
     instrument to base (relative to) frame" transformation.
      

     Spacecraft body-fixed frames:
        

\begindata

        FRAME_VG2_CONE_CLOCK     = -32200
        FRAME_-32200_NAME        = 'VG2_CONE_CLOCK'
        FRAME_-32200_CLASS       = 4
        FRAME_-32200_CLASS_ID    = -32200
        FRAME_-32200_CENTER      = -32
        TKFRAME_-32200_SPEC      = 'ANGLES'
        TKFRAME_-32200_RELATIVE  = 'VG2_SC_BUS'
        TKFRAME_-32200_ANGLES    = (  55.0, 0.0, 180.0 )
        TKFRAME_-32200_AXES      = (     3,   2,     1 )
        TKFRAME_-32200_UNITS     = 'DEGREES'
        
        FRAME_VG2_AZ_EL          = -32300
        FRAME_-32300_NAME        = 'VG2_AZ_EL'
        FRAME_-32300_CLASS       = 4
        FRAME_-32300_CLASS_ID    = -32300
        FRAME_-32300_CENTER      = -32
        TKFRAME_-32300_SPEC      = 'ANGLES'
        TKFRAME_-32300_RELATIVE  = 'VG2_SC_BUS'
        TKFRAME_-32300_ANGLES    = ( -173.0,      0.0,   180.0 )
        TKFRAME_-32300_AXES      = (      1,        2,       3 )
        TKFRAME_-32300_UNITS     = 'DEGREES'
        
        
\begintext

        The boresight of the antenna is the +Z axis.
        
\begindata
              
        FRAME_VG2_HGA            = -32400
        FRAME_-32400_NAME        = 'VG2_HGA'
        FRAME_-32400_CLASS       = 4
        FRAME_-32400_CLASS_ID    = -32400
        FRAME_-32400_CENTER      = -32
        TKFRAME_-32400_SPEC      = 'ANGLES'
        TKFRAME_-32400_RELATIVE  = 'VG2_SC_BUS'
        TKFRAME_-32400_ANGLES    = ( 180.0,      0.0,     0.0 )
        TKFRAME_-32400_AXES      = (     1,        2,       3 )
        TKFRAME_-32400_UNITS     = 'DEGREES'

\begintext

        Voyager-2 scan platform instrument frame definitions.
  
        The Euler angles of rotation given below are 
        referred to the ISSNA frame of reference which
        in this case is considered the equivalent of the
        scan platform's frame.
 
        Boresights are +Z in each instrument frame.


\begindata


        FRAME_VG2_ISSNA          = -32101
        FRAME_-32101_NAME        = 'VG2_ISSNA'
        FRAME_-32101_CLASS       = 4
        FRAME_-32101_CLASS_ID    = -32101
        FRAME_-32101_CENTER      = -32
        TKFRAME_-32101_SPEC      = 'ANGLES'
        TKFRAME_-32101_RELATIVE  = 'VG2_SCAN_PLATFORM'
        TKFRAME_-32101_ANGLES    = ( 0.0,  0.0,  0.0 )
        TKFRAME_-32101_AXES      = (   1,    2,    3 )
        TKFRAME_-32101_UNITS     = 'DEGREES'


        FRAME_VG2_ISSWA          = -32102
        FRAME_-32102_NAME        = 'VG2_ISSWA'
        FRAME_-32102_CLASS       = 4
        FRAME_-32102_CLASS_ID    = -32102
        FRAME_-32102_CENTER      = -32
        TKFRAME_-32102_SPEC      = 'ANGLES'
        TKFRAME_-32102_RELATIVE  = 'VG2_SCAN_PLATFORM'
        TKFRAME_-32102_ANGLES    = ( 0.171102, 0.0068, -0.0308 )
        TKFRAME_-32102_AXES      = (        3,      1,       2 )
        TKFRAME_-32102_UNITS     = 'DEGREES'
         

        FRAME_VG2_PPS            = -32103
        FRAME_-32103_NAME        = 'VG2_PPS'
        FRAME_-32103_CLASS       = 4
        FRAME_-32103_CLASS_ID    = -32103
        FRAME_-32103_CENTER      = -32
        TKFRAME_-32103_SPEC      = 'ANGLES'
        TKFRAME_-32103_RELATIVE  = 'VG2_SCAN_PLATFORM'
        TKFRAME_-32103_ANGLES    = ( 0.0,  -0.003,  -0.060 )
        TKFRAME_-32103_AXES      = (   3,       1,       2 )
        TKFRAME_-32103_UNITS     = 'DEGREES'
        

        FRAME_VG2_UVS            = -32104
        FRAME_-32104_NAME        = 'VG2_UVS'
        FRAME_-32104_CLASS       = 4
        FRAME_-32104_CLASS_ID    = -32104
        FRAME_-32104_CENTER      = -32
        TKFRAME_-32104_SPEC      = 'ANGLES'
        TKFRAME_-32104_RELATIVE  = 'VG2_SCAN_PLATFORM'
        TKFRAME_-32104_ANGLES    = ( 0.0,  -0.08,  0.0 )
        TKFRAME_-32104_AXES      = (   3,      1,    2 )
        TKFRAME_-32104_UNITS     = 'DEGREES'


        FRAME_VG2_UVSOCC         = -32105
        FRAME_-32105_NAME        = 'VG2_UVSOCC'
        FRAME_-32105_CLASS       = 4
        FRAME_-32105_CLASS_ID    = -32105
        FRAME_-32105_CENTER      = -32
        TKFRAME_-32105_SPEC      = 'ANGLES'
        TKFRAME_-32105_RELATIVE  = 'VG2_SCAN_PLATFORM'
        TKFRAME_-32105_ANGLES    = ( 0.0,  0.0, -19.26 )
        TKFRAME_-32105_AXES      = (   3,    1,      2 )
        TKFRAME_-32105_UNITS     = 'DEGREES'
        

        FRAME_VG2_IRIS           = -32106
        FRAME_-32106_NAME        = 'VG2_IRIS'
        FRAME_-32106_CLASS       = 4
        FRAME_-32106_CLASS_ID    = -32106
        FRAME_-32106_CENTER      = -32
        TKFRAME_-32106_SPEC      = 'ANGLES'
        TKFRAME_-32106_RELATIVE  = 'VG2_SCAN_PLATFORM'
        TKFRAME_-32106_ANGLES    = ( 0.0, 0.009 , 0.016 )
        TKFRAME_-32106_AXES      = (   3,     1,      2 )
        TKFRAME_-32106_UNITS     = 'DEGREES'
        

        FRAME_VG2_IRISOCC        = -32107
        FRAME_-32107_NAME        = 'VG2_IRISOCC'
        FRAME_-32107_CLASS       = 4
        FRAME_-32107_CLASS_ID    = -32107
        FRAME_-32107_CENTER      = -32
        TKFRAME_-32107_SPEC      = 'ANGLES'
        TKFRAME_-32107_RELATIVE  = 'VG2_SCAN_PLATFORM'
        TKFRAME_-32107_ANGLES    = ( 0.0, 0.0, -20.8 )
        TKFRAME_-32107_AXES      = (   3,   1,     2 )
        TKFRAME_-32107_UNITS     = 'DEGREES'

\begintext
