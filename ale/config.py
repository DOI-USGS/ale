"""
Config File
"""

# Directory with metakernals
spice_root = "/data/spice/"
cassini = '/scratch/localhome/kberry/dev/ale/ale/mk/cas_2004_v25.tm'
mdis = '/data/spice/mess-e_v_h-spice-6-v1.0/messsp_1000/extras/mk' # Messenger
mro = '/data/spice/mro-m-spice-6-v1.0/mrosp_1000/extras/mk' # Mars Reconnaissance Orbiter
kaguya = '/data/spice/SELENE/kernels/mk/'
dawn = '/data/spice/dawn-m_a-spice-6-v1.0/dawnsp_1000/extras/mk'


# Group = Kernels
#   NaifFrameCode             = -82360
#   LeapSecond                = $base/kernels/lsk/naif0012.tls
#   TargetAttitudeShape       = ($base/kernels/pck/pck00009.tpc,
#                                $cassini/kernels/pck/cpck15Dec2017.tpc)
#   TargetPosition            = (Table, $base/kernels/spk/de430.bsp)
#   InstrumentPointing        = (Table, $cassini/kernels/ck/08222_08227ra.bc,
#                                $cassini/kernels/fk/cas_v40.tf)
#   Instrument                = Null
#   SpacecraftClock           = $cassini/kernels/sclk/cas00172.tsc
#   InstrumentPosition        = (Table,
#                                $cassini/kernels/spk/180628RU_SCPSE_08220_082-
#                                72.bsp)
#   InstrumentAddendum        = $cassini/kernels/iak/IssNAAddendum004.ti
#
