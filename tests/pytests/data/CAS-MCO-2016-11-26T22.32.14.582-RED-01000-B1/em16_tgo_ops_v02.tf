KPL/FK

Frame (FK) SPICE kernel file for TGO science operations frames
===============================================================================

   This frames kernel defines a number of frames used by the TGO
   science operations centre to perform mission analysis and attitude
   dependent science opportunity identification.
   
   These frames can be used stand-alone, i.e. referring directly to them and
   assuming they correspond to the TGO spacecraft reference frame, or
   in combination with the TGO spacecraft frames. The latter will allow the
   user to use the existing alignments and instrument frame definitions to
   perform instrument specific mission analysis and attitude dependent
   science opportunity identification. Please refer to the section ``Using
   these frames'' for further details.


Version and Date
-------------------------------------------------------------------------------

   Version 0.2 -- August 2, 2016 -- Marc Costa Sitja, ESAC/ESA
   
      Removed ``TGO Mars Nadir orbit-aligned pointing'' frame definition for 
      there is no use case in ExoMars 2016 for it.
      Corrected minor typos.

   Version 0.1 -- June 6, 2016 -- Jorge Diaz del Rio, ODC Space
   
      Update comments to reflect new file naming conventions for ExoMars 2016
      frame kernels.

   Version 0.0 -- May 22, 2016 -- Jorge Diaz del Rio, ODC Space
   
      Initial version.


References
-------------------------------------------------------------------------------

   [1]   "Frames Required Reading"
   
   [2]   "Kernel Pool Required Reading"
   
   [3]   ``Science Operations Centre - Flight Dynamics - Pointing
         Timeline-ICD'' EXM-GS-ICD-ESC-50003 Issue 1.4, 15-12-2015
   

Contact Information
-------------------------------------------------------------------------------

   If you have any questions regarding this file contact SPICE support at
   ESAC:

           Marc Costa Sitja
           (+34) 91-8131-457
           mcosta@sciops.esa.int, esa_spice@sciops.esa.int
           
   or SPICE support at IKI:
   
           Anton Ledkov
           +7 (495) 333-12-66
           aledkov@rssi.ru
           
   or NAIF at JPL:
   
           Boris Semenov
           (818) 354-8136
           Boris.Semenov@jpl.nasa.gov
      
     
Implementation Notes
-------------------------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make
   use of this frame kernel must "load" the kernel normally during 
   program initialization. Loading the kernel associates the data items 
   with their names in a data structure called the "kernel pool". The 
   routine that loads a kernel into the pool is shown below:
                                                                               
      FORTRAN: (SPICELIB)

         CALL FURNSH ( frame_kernel_name )

      C: (CSPICE)

         furnsh_c ( frame_kernel_name );

      IDL: (ICY)

         cspice_furnsh, frame_kernel_name
         
      MATLAB: (MICE)
      
         cspice_furnsh ( 'frame_kernel_name' )

   This file was created and may be updated with a text editor or word
   processor.


TGO Science Operations frame names and NAIF ID Codes
-------------------------------------------------------------------------------
 
   The following frame is defined in this kernel file:

      SPICE Frame Name          Long-name
      ------------------------  ---------------------------------------------
      TGO_MARS_NPO              TGO Mars Nadir power-optimized pointing


   These frame has the following centers, frame class and NAIF
   ID:
   
      SPICE Frame Name          Center                 Class     NAIF ID
      ------------------------  ---------------------  -------  ---------
      TGO_MARS_NPO              TGO                    DYNAMIC   -143910


   The keywords implementing that frame definitions is located in the 
   "TGO Science Operations Frame Definitions" section.
               

General Notes About This File
-------------------------------------------------------------------------------

   About Required Data:
   --------------------
   All the dynamic frames defined in this file require at least one
   of the following kernel types to be loaded prior to their evaluation, 
   normally during program initialization:

     - Planetary and Satellite ephemeris data (SPK), i.e. de432, de405, etc;
     - Spacecraft ephemeris data (SPK);

   Note that loading different kernels will lead to different
   orientations of the same frame at a given epoch, providing different
   results from each other, in terms of state vectors referred to these 
   frames.


   Using these frames
   ------------------
   These frames have been implemented to define the different pointing
   profiles for the TGO spacecraft. These pointing profiles can be
   used in two different ways:

      [1] ``As is'' for analysis of offsets between the spacecraft
          attitude defined in the corresponding CK and a given pointing
          profile. Loading this kernel in combination with any TGO CK
          will allow the user to perform this comparison between the
          TGO_SPACECRAFT frame and any of the different frames defined
          within this kernel.
   
      [2] In combination with the TGO Frames kernel, to define
          a default pointing profile for the whole duration of the mission
          together with the spacecraft and instrument frames defined in the
          TGO FK. In this way, instrument-specific mission analysis
          activities, for which a particular pointing profile and knowledge
          of the instruments is required, can be conducted without the need
          for a spacecraft CK.
      
          In order to define such default pointing profile, the latest
          TGO frames kernel and this file shall be loaded before the
          selected ``TGO spacecraft frame overwrite'' frame kernel. As
          an example, imagine that the desired default pointing profile is
          "Nadir power optimized with respect to Mars", then the furnish
          (metakernel) file should contain the following sequence of frames
          kernels, in the following order:
      
              ...
         
              $DATA/fk/em16_tgo_v00.tf
              $DATA/fk/em16_tgo_ops_v00.tf
              $DATA/fk/em16_tgo_sc_mars_npo_v01.tf
         
              ...
         
            (*) the example presents version 0.0 of the ExoMars-2016 frames
            and TGO Science Operations frames kernels. Newer versions of
            these files will produce the same results. 
   
          By loading the ``em16_tgo_sc_mars_npo_vNN.tf'' frames kernel last,
          the spacecraft frame TGO_SPACECRAFT, which is defined as a CK-based
          frame in the ``TGO frames kernel'', will be overwritten as a
          type-4 fixed offset frame, mapping the TGO_SPACECRAFT frame to
          the TGO_MARS_NPO frame defined in the ``TGO Science
          Operations Frames Kernel'' (this) file.
      

TGO Science Operations Frame Definitions
-------------------------------------------------------------------------------

   This section contains the definition of the TGO science operations
   frames.

   
TGO Mars Nadir power-optimized pointing frame (TGO_MARS_NPO)
------------------------------------------------------------------------

   Definition:
   -----------
   The TGO Mars Nadir power-optimized pointing frame is defined as follows
   (from [3]):

      -  -Y axis is the primary vector and points from TGO to the
         center of Mars (Nadir direction);
         
      -  -X axis is the secondary vector and is the orthogonal component
         to the -Y axis of the Sun position relative to TGO;
         
      -  +Z axis completes the right-handed system;
      
      -  the original of this frame is the spacecraft's center of mass.
      
   All vectors are geometric: no corrections are used.
   
   
   Required Data:
   --------------
   This frame is defined as a two-vector frame.
   
   Both the primary and the secondary vector are defined as an 
   'observer-target position' vectors, therefore, the ephemeris data
   required to compute both the TGO-Mars position and the TGO-Sun
   position in J2000 frame have to be loaded before using this frame.


   Remarks:
   --------
   Since the primary and secondary vectors of this frame are defined
   based on the TGO-Mars position and TGO-Sun position vectors, the usage
   of different ephemerides to compute these vectors may lead to different
   frame orientation at given time.
   
  \begindata
      
      FRAME_TGO_MARS_NPO            = -143910
      FRAME_-143910_NAME            = 'TGO_MARS_NPO'
      FRAME_-143910_CLASS           =  5
      FRAME_-143910_CLASS_ID        = -143910
      FRAME_-143910_CENTER          = -143
      FRAME_-143910_RELATIVE        = 'J2000'
      FRAME_-143910_DEF_STYLE       = 'PARAMETERIZED'
      FRAME_-143910_FAMILY          = 'TWO-VECTOR'
      FRAME_-143910_PRI_AXIS        = '-Y'
      FRAME_-143910_PRI_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
      FRAME_-143910_PRI_OBSERVER    = 'TGO'
      FRAME_-143910_PRI_TARGET      = 'MARS'
      FRAME_-143910_PRI_ABCORR      = 'NONE'
      FRAME_-143910_SEC_AXIS        = '-X'
      FRAME_-143910_SEC_VECTOR_DEF  = 'OBSERVER_TARGET_POSITION'
      FRAME_-143910_SEC_OBSERVER    = 'TGO'
      FRAME_-143910_SEC_TARGET      = 'SUN'
      FRAME_-143910_SEC_ABCORR      = 'NONE'
      FRAME_-143910_SEC_FRAME       = 'J2000'
  
  \begintext

