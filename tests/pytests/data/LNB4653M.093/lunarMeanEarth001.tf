
KPL/FK


   SPICE Generic Lunar Reference Frame Specification Kernel
   =====================================================================

   Original file name:                   lunar_060616.tf
   Creation date:                        2006 June 16 18:54
   Created by:                           Nat Bachman  (NAIF/JPL)

   Modified:				 2007 July 10 15:02
   Modified By:				 Jeff Anderson
   Note:				 Define lunar frame to default
                                         to mean-earth


   Introduction
   =====================================================================

   This kernel specifies lunar body-fixed reference frames for use by
   SPICE-based application software.  These reference frames are
   associated with high-accuracy lunar orientation data provided by the
   JPL Solar System Dynamics Group's planetary ephemerides (both
   trajectory and lunar orientation data are stored in these ephemeris
   files).  These ephemerides have names of the form DE-nnn (DE stands
   for "developmental ephemeris").

   The frames specified by this kernel are realizations of two different
   lunar reference systems:

      Principal axes (PA) system 
      --------------------------
      The axes of this system are defined by the principal axes of the
      Moon.  Note that, due to the nature of the Moon's orbit and
      rotation, the Z axis of this system does not coincide with the
      Moon's mean spin axis, nor does the X axis coincide with the mean
      direction to the center of the Earth.
 
      Lunar principal axes frames realizing the lunar PA system and
      specified by this kernel are associated with JPL planetary
      ephemerides.  Each new JPL planetary ephemeris can (but does not
      necessarily) define a new realization of the lunar principal axes
      system.  Coordinates of lunar surface features expressed in lunar
      PA frames can change slightly from one lunar ephemeris version to
      the next.
 

      Mean Earth/polar axis (ME) system
      ---------------------------------
      The Lunar mean Earth/axis system is a lunar body-fixed reference
      system used in the IAU/IAG Working Group Report [2] to describe
      the orientation of the Moon relative to the ICRF frame. The +Z
      axis of this system is aligned with the mean lunar north pole,
      while the prime meridian contains the the mean Earth direction.
 
      The mean directions used to define the axes of a mean Earth/polar
      axis reference frame realizing the lunar ME system and specified
      by this kernel are associated with a given JPL planetary
      ephemeris version.  The rotation between the mean Earth frame for
      a given ephemeris version and the associated principal axes frame
      is given by a constant matrix (see [1]).


   For each JPL planetary ephemeris (DE), this kernel includes
   specifications of the corresponding principal axes and mean Earth/
   polar axis frames.  The names of these frames have the form

      LUNAR_DE-nnn_PR_AXES
     
   and

      LUNAR_DE-nn_MEAN_EARTH

   respectively.  This set of DE-dependent frame specifications will
   grow over time; frame specifications pertaining to older DEs will
   be retained in later versions of this frame kernel.

   For each type of reference frame, there are two "generic"
   frame specifications:  these frames are simply aliases for the
   two lunar body-fixed frames associated with the latest DE.  The
   generic frame names are

      LUNAR_PR_AXES
      LUNAR_MEAN_EARTH

   SPICE users may, if they wish, modify this kernel to assign these
   frame aliases to older DE-based frames.  NAIF recommends that, if
   this file is modified, the name of this file also be changed to
   avoid confusion.


   Comparison of PA and ME frames
   ------------------------------

   The rotation between the mean Earth frame for a given DE and the
   associated principal axes frame for the same DE is given by a constant
   matrix (see [1]).  For DE-403, the rotation angle of this matrix is
   approximately 0.028241 degrees; this is equivalent to approximately 860 m
   when expressed as a displacement along a great circle on the Moon's
   surface.


   Comparison of DE-based and IAU/IAG report rotation data
   -------------------------------------------------------

   Within the SPICE system, the lunar ME frame specified by the
   rotational elements from the IAU/IAG Working Group report [2] is
   given the name IAU_MOON; the data defining this frame are provided
   in a generic text PCK.

   The orientation of the lunar ME frame obtained from the DE-based
   lunar libration data and application of the DE-based PA-to-ME
   rotation described above does not agree closely with the lunar ME
   frame orientation given by the rotational elements from the IAU/IAG
   Working Group report (that is, the IAU_MOON frame). The difference
   is due to truncation of the libration series used in the report's
   formula for lunar orientation (see [1]).
 
   In the case of DE-403, for the time period ~2000-2020, this
   time-dependent difference has an amplitude of approximately 0.005
   degrees, which is equivalent to approximately 150 m, measured along
   a great circle on the Moon's surface, while the average value is
   approximately 0.0025 degrees, or 75 m.


   Using this Kernel
   =====================================================================

   In order for a SPICE-based application to use reference frames
   specified by this kernel, the application must load both this kernel
   and a binary lunar PCK containing lunar orientation data for the
   time of interest.  Normally the kernels need be loaded only once
   during program initialization.
 
   SPICE users may find it convenient to use a meta-kernel (also called
   a "FURNSH kernel") to name the kernels to be loaded. Below, we show
   an example of such a meta-kernel, as well as the source code of a
   small Fortran program that uses lunar body fixed frames.  The
   program's output is included as well.
 
   The kernel names shown here are simply used as examples; users must
   select the kernels appropriate for their applications.
 
   Numeric results shown below may differ from those obtained on users'
   computer systems.


   Meta-kernel
   -----------

      Example meta-kernel showing use of 

        - binary lunar PCK
        - generic lunar frame kernel (FK)
        - leapseconds kernel (LSK)
        - planetary SPK

       16-JUN-2006 (NJB)

       Note:  to actually use this kernel, replace the @ 
       characters below with backslashes (\).  The
       backslash character cannot be used here because these
       comments would be interpreted as actual load commands.

          @begindata

            KERNELS_TO_LOAD = ( 'de403_2000-2020_pa.bpc'
                                'lunar_060616.tf'
                                '/kernels/gen/lsk/leapseconds.ker'
                                '/kernels/gen/spk/de405.bsp'       )

          @begintext


   Example program
   ---------------

            PROGRAM EX1
            IMPLICIT NONE

            INTEGER               FILSIZ
            PARAMETER           ( FILSIZ = 255 )

            CHARACTER*(FILSIZ)    META

            DOUBLE PRECISION      ET
            DOUBLE PRECISION      LT
            DOUBLE PRECISION      STME  ( 6 )
            DOUBLE PRECISION      STPA  ( 6 )

      C
      C     Prompt user for meta-kernel name.
      C
            CALL PROMPT ( 'Enter name of meta-kernel > ', META )

      C
      C     Load lunar PCK, generic lunar frame kernel,
      C     leapseconds kernel, and planetary ephemeris
      C     via metakernel.
      C
            CALL FURNSH ( META )

      C
      C     Convert a time of interest from UTC to ET.
      C
            CALL STR2ET ( '2006 jun 8 06:50:00', ET )

            WRITE (*,*) 'ET (sec past J2000 TDB): ', ET
            WRITE (*,*) '   State of Earth relative to Moon'

      C
      C     Find the geometric state of the Earth relative to the
      C     Moon at ET, expressed relative to the generic ME frame.
      C    
            CALL SPKEZR ( 'Earth',  ET,      'LUNAR_MEAN_EARTH', 
           .              'NONE',   'Moon',  STME,               LT )

            WRITE (*,*) '      In ME frame:'
            WRITE (*,*) STME

      C
      C     Find the geometric state of the Earth relative to the
      C     Moon at ET, expressed relative to the generic PA frame.
      C    
            CALL SPKEZR ( 'Earth',  ET,      'LUNAR_PR_AXES', 
           .              'NONE',   'Moon',  STPA,               LT )

            WRITE (*,*) '      In PA frame:'
            WRITE (*,*) STPA

            END


   Program output
   --------------

   Enter name of meta-kernel > meta
    ET (sec past J2000 TDB):   203021465.
       State of Earth relative to Moon
          In ME frame:
     391739.183 -33210.254  25299.0887 -0.0592286405 -0.048721834  0.0917188552
          In PA frame:
     391719.148 -33331.588  25449.2934 -0.0592788895 -0.0487034073  0.0916961762



   References
   =====================================================================

   [1]  A.S. Konopliv, S.W. Asmar, E. Carranza, W.L. Sjogren, and D.N.
        Yuan (2001). "Recent Gravity Models as a Result of the Lunar
        Prospector Mission," Icarus 150, pp. 1-18.

   [2]  Seidelmann, P.K., Abalakin, V.K., Bursa, M., Davies, M.E.,
        Bergh, C. de, Lieske, J.H., Oberst, J., Simon, J.L., Standish,
        E.M., Stooke, P., and Thomas, P.C. (2002). "Report of the
        IAU/IAG Working Group on Cartographic Coordinates and Rotational
        Elements of the Planets and Satellites: 2000," Celestial
        Mechanics and Dynamical Astronomy, v.82, Issue 1, pp. 83-111.

   [3]  Roncoli, R. (2005).  "Lunar Constants and Models Document," 
        JPL D-32296.


   Frame Specifications
   =====================================================================

   LUNAR_PR_AXES is the name of the generic lunar principal axes
   reference frame.  This frame is an alias for the principal axes
   frame defined by the latest version of the JPL Solar System Dynamics
   Group's planetary ephemeris.
 
   Currently LUNAR_PR_AXES is an alias for the lunar principal axes
   frame associated with the planetary ephemeris DE-403.

   \begindata
 
      FRAME_LUNAR_PR_AXES            = 310000
      FRAME_310000_NAME              = 'LUNAR_PR_AXES'
      FRAME_310000_CLASS             = 4
      FRAME_310000_CLASS_ID          = 310000
      FRAME_310000_CENTER            = 301

      TKFRAME_310000_SPEC            = 'MATRIX'
      TKFRAME_310000_RELATIVE        = 'LUNAR_DE-403_PR_AXES'
      TKFRAME_310000_MATRIX          = ( 1 0 0
                                         0 1 0
                                         0 0 1 )

   \begintext

   LUNAR_MEAN_EARTH is the name of the generic lunar mean Earth/ polar
   axis reference frame.  This frame is an alias for the mean
   Earth/polar axis frame defined by the latest version of the JPL
   Solar System Dynamics Group's planetary ephemeris.
 
   Currently LUNAR_MEAN_EARTH is an alias for the lunar mean Earth/
   polar axis frame associated with the planetary ephemeris DE-403.

   \begindata

      FRAME_LUNAR_MEAN_EARTH         = 310001
      FRAME_310001_NAME              = 'LUNAR_MEAN_EARTH'
      FRAME_310001_CLASS             = 4
      FRAME_310001_CLASS_ID          = 310001
      FRAME_310001_CENTER            = 301
 
      TKFRAME_310001_SPEC            = 'MATRIX'
      TKFRAME_310001_RELATIVE        = 'LUNAR_DE-403_MEAN_EARTH'
      TKFRAME_310001_MATRIX          = ( 1 0 0
                                         0 1 0
                                         0 0 1 )

   \begintext


   LUNAR_DE-403_PR_AXES is the name of the lunar principal axes
   reference frame defined by JPL's DE-403 planetary ephemeris.

   \begindata

      FRAME_LUNAR_DE-403_PR_AXES     = 310002
      FRAME_310002_NAME              = 'LUNAR_DE-403_PR_AXES'
      FRAME_310002_CLASS             = 2
      FRAME_310002_CLASS_ID          = 31000
      FRAME_310002_CENTER            = 301

   \begintext



   LUNAR_DE-403_MEAN_EARTH is the name of the lunar mean Earth/polar
   axis reference frame defined by JPL's DE-403 planetary ephemeris.

   Rotation angles are from reference [1].

   \begindata
 
      FRAME_LUNAR_DE-403_MEAN_EARTH  = 310003
      FRAME_310003_NAME              = 'LUNAR_DE-403_MEAN_EARTH'
      FRAME_310003_CLASS             = 4
      FRAME_310003_CLASS_ID          = 310003
      FRAME_310003_CENTER            = 301

      TKFRAME_310003_SPEC            = 'ANGLES'
      TKFRAME_310003_RELATIVE        = 'LUNAR_PR_AXES'
      TKFRAME_310003_ANGLES          = (   63.8986   79.0768   0.1462  )
      TKFRAME_310003_AXES            = (   3,        2,        1       )
      TKFRAME_310003_UNITS           = 'ARCSECONDS'

   \begintext

   Set to Mean Earth Lunar frame, dcook July 2007

   \begindata
   OBJECT_MOON_FRAME = 'LUNAR_MEAN_EARTH'
   OBJECT_301_FRAME = 'LUNAR_MEAN_EARTH'

   \begintext
   End of kernel









