import pytest
from unittest.mock import patch
import pvl
import numpy as np
from ale.base.data_isis import IsisSpice

from conftest import get_image_label

testlabel = """
Object = IsisCube
  Object = Core
    StartByte   = 65537
    Format      = Tile
    TileSamples = 128
    TileLines   = 128

    Group = Dimensions
      Samples = 256
      Lines   = 5000
      Bands   = 1
    End_Group

    Group = Pixels
      Type       = SignedWord
      ByteOrder  = Lsb
      Base       = 0.0
      Multiplier = 1.0
    End_Group
  End_Object

  Group = Instrument
    SpacecraftName              = "MARS RECONNAISSANCE ORBITER"
    InstrumentId                = HIRISE
    TargetName                  = Mars
    StartTime                   = 2006-11-17T03:27:53.118
    StopTime                    = 2006-11-17T03:27:54.792
    ObservationStartCount       = 848201291:54379
    SpacecraftClockStartCount   = 848201291:62546
    SpacecraftClockStopCount    = 848201293:41165
    ReadoutStartCount           = 848201300:53057
    CalibrationStartTime        = 2006-11-17T03:27:53.104
    CalibrationStartCount       = 848201291:61647
    AnalogPowerStartTime        = -9999
    AnalogPowerStartCount       = -9999
    MissionPhaseName            = "PRIMARY SCIENCE PHASE"
    LineExposureDuration        = 334.7500 <MICROSECONDS>
    ScanExposureDuration        = 83.6875 <MICROSECONDS>
    DeltaLineTimerCount         = 155
    Summing                     = 4
    Tdi                         = 64
    FocusPositionCount          = 2020
    PoweredCpmmFlag             = (On, On, On, On, On, On, On, On, On, On, On,
                                   On, On, On)
    CpmmNumber                  = 4
    CcdId                       = BG12
    ChannelNumber               = 0
    LookupTableType             = Stored
    LookupTableNumber           = 10
    LookupTableMinimum          = -9998
    LookupTableMaximum          = -9998
    LookupTableMedian           = -9998
    LookupTableKValue           = -9998
    StimulationLampFlag         = (Off, Off, Off)
    HeaterControlFlag           = (On, On, On, On, On, On, On, On, On, On, On,
                                   On, On, On)
    OptBnchFlexureTemperature   = 20.455 <C>
    OptBnchMirrorTemperature    = 20.1949 <C>
    OptBnchFoldFlatTemperature  = 20.5417 <C>
    OptBnchFpaTemperature       = 19.8482 <C>
    OptBnchFpeTemperature       = 19.5881 <C>
    OptBnchLivingRmTemperature  = 20.1949 <C>
    OptBnchBoxBeamTemperature   = 20.455 <C>
    OptBnchCoverTemperature     = 20.1082 <C>
    FieldStopTemperature        = 18.375 <C>
    FpaPositiveYTemperature     = 19.1548 <C>
    FpaNegativeYTemperature     = 19.0681 <C>
    FpeTemperature              = 17.9418 <C>
    PrimaryMirrorMntTemperature = 20.0215 <C>
    PrimaryMirrorTemperature    = 20.3683 <C>
    PrimaryMirrorBafTemperature = 0.414005 <C>
    MsTrussLeg0ATemperature     = 20.3683 <C>
    MsTrussLeg0BTemperature     = 20.5417 <C>
    MsTrussLeg120ATemperature   = 19.5881 <C>
    MsTrussLeg120BTemperature   = 20.2816 <C>
    MsTrussLeg240ATemperature   = 19.6748 <C>
    MsTrussLeg240BTemperature   = 19.9348 <C>
    BarrelBaffleTemperature     = -21.006 <C>
    SunShadeTemperature         = -28.7562 <C>
    SpiderLeg30Temperature      = 17.7686 <C>
    SpiderLeg120Temperature     = -9999
    SpiderLeg240Temperature     = -9999
    SecMirrorMtrRngTemperature  = 19.5881 <C>
    SecMirrorTemperature        = 20.7151 <C>
    SecMirrorBaffleTemperature  = -18.7871 <C>
    IeaTemperature              = 25.8353 <C>
    FocusMotorTemperature       = 21.4088 <C>
    IePwsBoardTemperature       = 17.7363 <C>
    CpmmPwsBoardTemperature     = 18.078 <C>
    MechTlmBoardTemperature     = 35.0546 <C>
    InstContBoardTemperature    = 34.6875 <C>
    DllLockedFlag               = (YES, YES)
    DllResetCount               = 0
    DllLockedOnceFlag           = (YES, YES)
    DllFrequenceCorrectCount    = 4
    ADCTimingSetting            = -9999
    Unlutted                    = TRUE
  End_Group

  Group = Archive
    DataSetId              = MRO-M-HIRISE-2-EDR-V1.0
    ProducerId             = UA
    ObservationId          = PSP_001446_1790
    ProductId              = PSP_001446_1790_BG12_0
    ProductVersionId       = 1.0
    EdrProductCreationTime = 2006-11-17T05:03:31
    RationaleDescription   = Null
    OrbitNumber            = 1446
    SoftwareName           = "HiRISE_Observation v2.9 (2.43 2006/10/01
                              05:41:12)"
    ObservationStartTime   = 2006-11-17T03:27:52.993
    ReadoutStartTime       = 2006-11-17T03:28:01.973
    TrimLines              = 607
    FelicsCompressionFlag  = YES
    IdFlightSoftwareName   = IE_FSW_V4
  End_Group

  Group = BandBin
    Name   = BlueGreen
    Center = 500 <NANOMETERS>
    Width  = 200 <NANOMETERS>
  End_Group

  Group = Kernels
    NaifIkCode                = -74699
    LeapSecond                = $base/kernels/lsk/naif0009.tls
    TargetAttitudeShape       = $base/kernels/pck/pck00009.tpc
    TargetPosition            = (Table, $base/kernels/spk/de405.bsp)
    InstrumentPointing        = (Table,
                                 $mro/kernels/ck/mro_sc_psp_061114_061120.bc,
                                 $mro/kernels/fk/mro_v14.tf)
    Instrument                = $mro/kernels/ik/mro_hirise_v11.ti
    SpacecraftClock           = $mro/kernels/sclk/MRO_SCLKSCET.00042.65536.tsc
    InstrumentPosition        = (Table, $mro/kernels/spk/mro_psp1.bsp)
    InstrumentAddendum        = $mro/kernels/iak/hiriseAddendum006.ti
    ShapeModel                = Null
    InstrumentPositionQuality = Reconstructed
    InstrumentPointingQuality = Reconstructed
    CameraVersion             = 1
  End_Group
End_Object

Object = Label
  Bytes = 65536
End_Object

Object = Table
  Name      = "HiRISE Calibration Ancillary"
  StartByte = 2687543
  Bytes     = 4920
  Records   = 41
  ByteOrder = Lsb

  Group = Field
    Name = GapFlag
    Type = Integer
    Size = 1
  End_Group

  Group = Field
    Name = LineNumber
    Type = Integer
    Size = 1
  End_Group

  Group = Field
    Name = BufferPixels
    Type = Integer
    Size = 12
  End_Group

  Group = Field
    Name = DarkPixels
    Type = Integer
    Size = 16
  End_Group
End_Object

Object = Table
  Name      = "HiRISE Calibration Image"
  StartByte = 2692463
  Bytes     = 41984
  Records   = 41
  ByteOrder = Lsb

  Group = Field
    Name = Calibration
    Type = Integer
    Size = 256
  End_Group
End_Object

Object = Table
  Name        = "HiRISE Ancillary"
  StartByte   = 2734447
  Bytes       = 600000
  Records     = 5000
  ByteOrder   = Lsb
  Association = Lines

  Group = Field
    Name = GapFlag
    Type = Integer
    Size = 1
  End_Group

  Group = Field
    Name = LineNumber
    Type = Integer
    Size = 1
  End_Group

  Group = Field
    Name = BufferPixels
    Type = Integer
    Size = 12
  End_Group

  Group = Field
    Name = DarkPixels
    Type = Integer
    Size = 16
  End_Group
End_Object

Object = Table
  Name                = InstrumentPointing
  StartByte           = 3364439
  Bytes               = 1152
  Records             = 18
  ByteOrder           = Lsb
  TimeDependentFrames = (-74000, -74900, 1)
  ConstantFrames      = (-74690, -74000)
  ConstantRotation    = (0.9999703083413, 0.0, -0.0077059999872177,
                         8.81584889031119e-06, 0.99999934560434,
                         0.0011439900269605, 0.0077059949444447,
                         -0.0011440239949305, 0.99996965396507)
  CkTableStartTime    = 217006138.29611
  CkTableEndTime      = 217006139.96986
  CkTableOriginalSize = 5001
  Description         = "Created by spiceinit"
  Kernels             = ($mro/kernels/ck/mro_sc_psp_061114_061120.bc,
                         $mro/kernels/fk/mro_v14.tf)

  Group = Field
    Name = J2000Q0
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Q1
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Q2
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Q3
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = AV1
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = AV2
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = AV3
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = ET
    Type = Double
    Size = 1
  End_Group
End_Object

Object = Table
  Name                 = InstrumentPosition
  StartByte            = 3564479
  Bytes                = 168
  Records              = 3
  ByteOrder            = Lsb
  CacheType            = HermiteSpline
  SpkTableStartTime    = 217006138.29611
  SpkTableEndTime      = 217006139.96986
  SpkTableOriginalSize = 5001.0
  Description          = "Created by spiceinit"
  Kernels              = $mro/kernels/spk/mro_psp1.bsp

  Group = Field
    Name = J2000X
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Y
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Z
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000XV
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000YV
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000ZV
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = ET
    Type = Double
    Size = 1
  End_Group
End_Object

Object = Table
  Name                = BodyRotation
  StartByte           = 3724511
  Bytes               = 128
  Records             = 2
  ByteOrder           = Lsb
  TimeDependentFrames = (10014, 1)
  CkTableStartTime    = 217006138.29611
  CkTableEndTime      = 217006139.96986
  CkTableOriginalSize = 2
  Description         = "Created by spiceinit"
  Kernels             = ($base/kernels/spk/de405.bsp,
                         $base/kernels/pck/pck00009.tpc)
  SolarLongitude      = 136.56543136097

  Group = Field
    Name = J2000Q0
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Q1
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Q2
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Q3
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = AV1
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = AV2
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = AV3
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = ET
    Type = Double
    Size = 1
  End_Group
End_Object

Object = Table
  Name                 = SunPosition
  StartByte            = 3924551
  Bytes                = 112
  Records              = 2
  ByteOrder            = Lsb
  CacheType            = Linear
  SpkTableStartTime    = 217006138.29611
  SpkTableEndTime      = 217006139.96986
  SpkTableOriginalSize = 2.0
  Description          = "Created by spiceinit"
  Kernels              = $base/kernels/spk/de405.bsp

  Group = Field
    Name = J2000X
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Y
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000Z
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000XV
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000YV
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = J2000ZV
    Type = Double
    Size = 1
  End_Group

  Group = Field
    Name = ET
    Type = Double
    Size = 1
  End_Group
End_Object

Object = History
  Name      = IsisCube
  StartByte = 4084583
  Bytes     = 3483
End_Object

Object = OriginalLabel
  Name      = IsisCube
  StartByte = 3334447
  Bytes     = 29992
End_Object

Object = NaifKeywords
  BODY499_RADII                            = (3396.19, 3396.19, 3376.2)
  INS-74699_FOCAL_LENGTH                   = 11994.9988
  INS-74699_PIXEL_PITCH                    = 0.012
  CLOCK_ET_-74999_848201291:62546_COMPUTED = 0000000000000040
  INS-74699_TRANSX                         = (-82.2033, 2.0e-06, 0.012)
  INS-74699_TRANSY                         = (11.9993, -0.012, 2.0e-06)
  INS-74699_ITRANSS                        = (1001.28, 0.0163, -83.3333)
  INS-74699_ITRANSL                        = (6850.08, 83.3333, 0.0163)
  INS-74699_OD_K                           = (-0.0048509, 2.41312e-07,
                                              -1.62369e-13)
End_Object
End
"""

@pytest.fixture
def testdata():
    isis_spice = IsisSpice()
    isis_spice.label = pvl.loads(testlabel)
    return isis_spice


def test_sensor_position(testdata):
    pass


def test_sun_position(testdata):
    pass


def test_detector_center_sample(testdata):
    assert testdata.detector_center_sample == None


def test_detector_center_line(testdata):
    assert testdata.detector_center_line == None


def test_focal_length(testdata):
    assert testdata.focal_length == 11994.9988


def test_focal2pixel_lines(testdata):
    assert testdata.focal2pixel_lines == [6850.08, 83.3333, 0.0163]


def test_focal2pixel_sample(testdata):
    assert testdata.focal2pixel_samples == [1001.28, 0.0163, -83.3333]


def test_pixel2focal_x(testdata):
    assert testdata.pixel2focal_x == [-82.2033, 2.0e-06, 0.012]


def test_pixel2focal_y(testdata):
    assert testdata.pixel2focal_y == [11.9993, -0.012, 2.0e-06]


def test_target_body_radii(testdata):
    assert testdata.target_body_radii == [3396.19, 3396.19, 3376.2]


def test_ephemeris_start_time(testdata):
   assert testdata.ephemeris_start_time == 2


def test_naif_keywords(testdata):

    naif_keywords =   pvl.loads("""Object = NaifKeywords
                        BODY499_RADII                            = (3396.19, 3396.19, 3376.2)
                        INS-74699_FOCAL_LENGTH                   = 11994.9988
                        INS-74699_PIXEL_PITCH                    = 0.012
                        CLOCK_ET_-74999_848201291:62546_COMPUTED = 0000000000000040
                        INS-74699_TRANSX                         = (-82.2033, 2.0e-06, 0.012)
                        INS-74699_TRANSY                         = (11.9993, -0.012, 2.0e-06)
                        INS-74699_ITRANSS                        = (1001.28, 0.0163, -83.3333)
                        INS-74699_ITRANSL                        = (6850.08, 83.3333, 0.0163)
                        INS-74699_OD_K                           = (-0.0048509, 2.41312e-07, -1.62369e-13)
                        END_OBJECT
                        END""")['NaifKeywords']
    assert testdata.naif_keywords == naif_keywords


def test_odtk(testdata):
    assert testdata.odtk == [-0.0048509, 2.41312e-07, -1.62369e-13]


def test_frame_chain(testdata):
    testdata._inst_pointing_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'ET' : [0, 1],
        'TimeDependentFrames' : [-1000, -100, 1],
        'ConstantRotation' : [0, 0, 1, 1, 0 , 0, 0, 1, 0],
        'ConstantFrames' : [-1020, -1000]}
    testdata._body_orientation_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'ET' : [0, 1],
        'TimeDependentFrames' : [80, 1],
        'ConstantRotation' : [0, 0, 1, 1, 0 , 0, 0, 1, 0],
        'ConstantFrames' : [81, 80]}
    frame_chain = testdata.frame_chain
    assert len(frame_chain.nodes) == 5
    assert frame_chain.has_node(1)
    assert frame_chain.has_node(80)
    assert frame_chain.has_node(81)
    assert frame_chain.has_node(-1000)
    assert frame_chain.has_node(-1020)
    assert len(frame_chain.edges) == 8
    assert frame_chain.has_edge(1, 80)
    assert frame_chain.has_edge(80, 1)
    assert frame_chain.has_edge(81, 80)
    assert frame_chain.has_edge(80, 81)
    assert frame_chain.has_edge(1, -1000)
    assert frame_chain.has_edge(-1000, 1)
    assert frame_chain.has_edge(-1020, -1000)
    assert frame_chain.has_edge(-1000, -1020)

def test_sun_position_cache(testdata):
    testdata._inst_pointing_table = {
        'J2000Q0' : [1, 1],
        'J2000Q1' : [0, 0],
        'J2000Q2' : [0, 0],
        'J2000Q3' : [0, 0],
        'ET' : [0, 1],
        'TimeDependentFrames' : [-1000, -100, 1]}
    testdata._body_orientation_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'ET' : [0, 1],
        'TimeDependentFrames' : [80, 1]}
    testdata._sun_position_table = {
        'J2000X' : [1, 0],
        'J2000Y' : [0, 1],
        'J2000Z' : [0, 0],
        'J2000XV' : [-1, 0],
        'J2000YV' : [0, -1],
        'J2000ZV' : [0, 0],
        'ET' : [0, 1]}
    sun_pos, sun_vel, sun_times = testdata.sun_position
    np.testing.assert_almost_equal(sun_pos, [[1000, 0, 0], [0, 0, 1000]])
    np.testing.assert_almost_equal(sun_vel,
                                   [[-1000, -2000*np.pi*np.sqrt(3)/9, 2000*np.pi*np.sqrt(3)/9],
                                    [-2000*np.pi*np.sqrt(3)/9, 2000*np.pi*np.sqrt(3)/9, -1000]])
    np.testing.assert_equal(sun_times, [0, 1])

def test_sun_position_polynomial(testdata):
    testdata._inst_pointing_table = {
        'J2000Q0' : [1, 1],
        'J2000Q1' : [0, 0],
        'J2000Q2' : [0, 0],
        'J2000Q3' : [0, 0],
        'ET' : [2, 4],
        'TimeDependentFrames' : [-1000, -100, 1]}
    testdata._body_orientation_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'ET' : [2, 4],
        'TimeDependentFrames' : [80, 1]}
    testdata._sun_position_table = {
        'SpkTableOriginalSize' : 2,
        'SpkTableStartTime' : 2,
        'SpkTableEndTime' : 4,
        'J2000SVX' : [1, -1, 2],
        'J2000SVY' : [0, 1, 2],
        'J2000SVZ' : [0, -1, 1]}

    sun_pos, sun_vel, sun_times = testdata.sun_position
    np.testing.assert_almost_equal(sun_pos, [[1000, 0, 0], [-1000, 0, 1000]])
    np.testing.assert_almost_equal(sun_vel,
                                   [[-500, 500 - 1000*np.pi*np.sqrt(3)/9, -500 + 1000*np.pi*np.sqrt(3)/9],
                                    [-500 - 1000*np.pi*np.sqrt(3)/9, -500 + 2000*np.pi*np.sqrt(3)/9, 500 - 1000*np.pi*np.sqrt(3)/9]])
    np.testing.assert_equal(sun_times, [2, 4])

def test_inst_position_cache(testdata):
    testdata._inst_pointing_table = {
        'J2000Q0' : [1, 1],
        'J2000Q1' : [0, 0],
        'J2000Q2' : [0, 0],
        'J2000Q3' : [0, 0],
        'ET' : [0, 1],
        'TimeDependentFrames' : [-1000, -100, 1]}
    testdata._body_orientation_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'ET' : [0, 1],
        'TimeDependentFrames' : [80, 1]}
    testdata._inst_position_table = {
        'J2000X' : [1, 0],
        'J2000Y' : [0, 1],
        'J2000Z' : [0, 0],
        'J2000XV' : [-1, 0],
        'J2000YV' : [0, -1],
        'J2000ZV' : [0, 0],
        'ET' : [0, 1]}
    sensor_pos, sensor_vel, sensor_times = testdata.sensor_position
    np.testing.assert_almost_equal(sensor_pos, [[1000, 0, 0], [0, 0, 1000]])
    np.testing.assert_almost_equal(sensor_vel,
                                   [[-1000, -2000*np.pi*np.sqrt(3)/9, 2000*np.pi*np.sqrt(3)/9],
                                    [-2000*np.pi*np.sqrt(3)/9, 2000*np.pi*np.sqrt(3)/9, -1000]])
    np.testing.assert_equal(sensor_times, [0, 1])

def test_no_tables():
    test_file = get_image_label('B10_013341_1010_XN_79S172W')
    test_mix_in = IsisSpice()
    test_mix_in._file = test_file
    test_mix_in.label = pvl.load(test_file)
    with pytest.raises(ValueError):
        test_mix_in.inst_pointing_table
    with pytest.raises(ValueError):
        test_mix_in.body_orientation_table
    with pytest.raises(ValueError):
        test_mix_in.inst_position_table
    with pytest.raises(ValueError):
        test_mix_in.sun_position_table
