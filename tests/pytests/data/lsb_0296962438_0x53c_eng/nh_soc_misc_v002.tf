KPL/FK

\beginlabel
PDS_VERSION_ID               = PDS3
RECORD_TYPE                  = STREAM
RECORD_BYTES                 = "N/A"
^SPICE_KERNEL                = "nh_soc_misc_v002.tf"
MISSION_NAME                 = "NEW HORIZONS"
SPACECRAFT_NAME              = "NEW HORIZONS"
DATA_SET_ID                  = "NH-J/P/SS-SPICE-6-V1.0"
KERNEL_TYPE_ID               = FK
PRODUCT_ID                   = "nh_soc_misc_v002.tf"
PRODUCT_CREATION_TIME        = 2016-04-30T00:00:00
PRODUCER_ID                  = "SWRI"
MISSION_PHASE_NAME           = "N/A"
PRODUCT_VERSION_TYPE         = ACTUAL
PLATFORM_OR_MOUNTING_NAME    = "N/A"
START_TIME                   = "N/A"
STOP_TIME                    = "N/A"
SPACECRAFT_CLOCK_START_COUNT = "N/A"
SPACECRAFT_CLOCK_STOP_COUNT  = "N/A"
TARGET_NAME                  = {
                               JUPITER,
                               PLUTO,
                               "SOLAR SYSTEM"
                               }
INSTRUMENT_NAME              = "N/A"
NAIF_INSTRUMENT_ID           = "N/A"
SOURCE_PRODUCT_ID            = "N/A"
NOTE                         = "See comments in the file for details"
OBJECT                       = SPICE_KERNEL
  INTERCHANGE_FORMAT         = ASCII
  KERNEL_TYPE                = FRAMES
  DESCRIPTION                = "NH frames kernel.  Provides supplemental
name-to-ID mapping for misc bodies (Asteroid APL; target KBO 2014 MU69 
of proposed NH extended mission; etc.) "
END_OBJECT                   = SPICE_KERNEL
\endlabel

KPL/FK

New Horizons Science Operations Center-specific Target Frames Kernel
===============================================================================

   This frame kernel contains the NAIF body name/code translation
   for miscellaneous targets that will show up in some NH FITS file
   headers due to targets added that are not in the project Frames 
   Kernel (nh_vXXX.tf) or other project SPICE kernels.


Version and Date
-------------------------------------------------------------------------------

   The TEXT_KERNEL_ID stores version information of loaded project text
   kernels. Each entry associated with the keyword is a string that comprises
   four parts:  kernel name; kernel version; entry date; kernel type. For 
   example, the NH I-kernel might have an entry as follows:

           TEXT_KERNEL_ID += 'NEWHORIZONS V2.0.0 30-APRIL-2016    IK'
                                  |          |         |          |
                                  |          |         |          |
              KERNEL NAME <-------+          |         |          |
                                             |         |          V
                             VERSION <-------+         |     KERNEL TYPE
                                                       |
                                                       V
                                                  ENTRY DATE

   New Horizons Science Operations Center-specific Frames Kernel Version:

           \begindata

           TEXT_KERNEL_ID += 'NEWHORIZONS_SOC_FRAMES V0.0.1 21-AUG-2008 FK'

           \begintext

   Version 0.0.0 -- August 21, 2008 -- Brian Carcich, SWRI Consultant

            --   Initial version:  Provide NAIF body name/code translation
                                   for asteroid 2002 JF56, aka (132524) APL

   Version 0.0.1 -- August 21, 2008 -- Brian Carcich, SWRI Consultant

            --   Initial version:  Moved name 2002_JF56 last so it is
                                   preferred output of BODC2N().

   Version 0.0.2 -- April, 30 2016   - Brian Carcich, SWRI Consultant

            --   Added 2014 MU69/3713011 name/ID mapping
            --   Fixed email

References
-------------------------------------------------------------------------------

            1.   ``SPICE NAIF_IDS Required Reading''

            2.   ``SPICE PCK Required Reading''

            3.   ``SPICE Kernel Pool Required Reading''

            4.   ``SPICE FRAMES Required Reading''

            5.   ``SPK Required Reading''

            6.   ``BRIEF User's Guide''

            7.   Minor Planet Center - Minor Planet Names:
                   http://cfa-www.harvard.edu/iau/lists/MPNames.html
                   - dated 2008-Jul-17

            8.   http://en.wikipedia.org/wiki/132524_APL
                 - as of 2008-Jul-21



Contact Information
-------------------------------------------------------------------------------

   Brian Carcich, SWRI Consultant, BrianTCarcich<AT>gmail.com

   - replace <AT> with ASCII 64 character


Implementation Notes
-------------------------------------------------------------------------------

   This file is used by the SPICE system as follows: programs that make use of
   this instrument kernel must ``load'' the kernel, normally during program
   initialization. Loading the kernel associates data items with their names
   in a data structure called the ``kernel pool''. The SPICELIB routine FURNSH,
   CSPICE routine furnsh_c, and IDL routine cspice_furnsh load SPICE kernels
   as shown below:

   FORTRAN (SPICELIB)

           CALL FURNSH ( 'kernel_name' )

   C (CSPICE)

           furnsh_c ( "kernel_name" )

   ICY (IDL)

           cspice_furnsh, 'kernel_name'

   In order for a program or subroutine to extract data from the pool, the
   SPICELIB routines GDPOOL, GCPOOL, and GIPOOL are used. See [3] for details.

   This file was created and may be updated with a text editor or word
   processor.


NAIF Body Code for asteroid APL (formerly 2002 JF56)
-------------------------------------------------------------------------------

  This file provides the SPICE toolkit with the NAIF Body Code for asteroid 
  2002 JF56 with various forms of its name.  See [1] for more details.

  The NAIF Body Code of 3127862 is included in the SP-Kernel provided 
  by the New Horizons project to the Science Operation Center (SOC).
  in SP-Kernel originally named 'sb-2002jf56-2.bsp'

  The output of the NAIF diagnostic program BRIEF on that SP-Kernel was 
  used to determine the NAIF Body Code of the asteroid; that output is
  included here:

    Brief.  Version: 2.2.0        (SPICE Toolkit N0058)


    Summary for: /home/soc/spice/kernels/recon/spk/sb-2002jf56-2.bsp

    Body: 3127862
          Start of Interval (ET)              End of Interval (ET)
          --------------------------------    --------------------------------
          2006 JAN 19 00:00:00.000            2006 OCT 02 00:00:00.000

  See [5] and [6] for details.

  Since no pole solution exists for 2002 JF56, no frame information will be 
  provided (e.g. BODY3127862_POLE_RA, &c; see [2] and [4] for details).

  Note that, according to the NAIF_IDS Required Reading [1], the ID of this
  asteroid in the JPL Asteroid and Comet Catalog is probably 1127862, while
  this asteroid also has an ID of 132524 assigned by the Minor Planet Center 
  published in various places (e.g. see [7] and [8]).

  Several names will be provided, with and without spaces, all referring
  to the same object.

\begindata

NAIF_BODY_NAME += ( 'APL'
                  , '132524_APL'
                  , '(132524) APL'
                  , '(132524) 2002 JF56'
                  , '2002 JF56'
                  , '2002_JF56'
                  , '132524 APL'
                  )
NAIF_BODY_CODE += ( 2132524
                  , 2132524
                  , 2132524
                  , 2132524
                  , 2132524
                  , 2132524
                  , 2132524
                  )

NAIF_BODY_NAME += ( '2014 MU69'
                  , '2014_MU69'
                  )
NAIF_BODY_CODE += ( 3713011
                  , 3713011
                  )

\begintext

