import pytest
import pvl

import ale
from ale import base
from ale.base import Driver
from ale.base import label_isis
from ale.base.label_isis import IsisLabel


@pytest.fixture
def test_label():
    return """
Object = IsisCube
Object = Core
StartByte   = 65537
Format      = Tile
TileSamples = 512
TileLines   = 512

Group = Dimensions
  Samples = 1024
  Lines   = 1024
  Bands   = 1
End_Group

Group = Pixels
  Type       = Real
  ByteOrder  = Lsb
  Base       = 0.0
  Multiplier = 1.0
End_Group
End_Object

Group = Instrument
SpacecraftName        = Messenger
InstrumentName        = "MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA"
InstrumentId          = MDIS-NAC
TargetName            = Venus
OriginalTargetName    = VENUS
StartTime             = 2007-06-06T00:22:10.751814
StopTime              = 2007-06-06T00:22:10.768814
SpacecraftClockCount  = 1/0089576657:973000
MissionPhaseName      = "VENUS 2 FLYBY"
ExposureDuration      = 17 <MS>
ExposureType          = AUTO
DetectorTemperature   = -43.65 <DEGC>
FocalPlaneTemperature = -23.63 <DEGC>
FilterTemperature     = N/A
OpticsTemperature     = -24.72 <DEGC>
AttitudeQuality       = Ok
FilterWheelPosition   = 28320
PivotPosition         = -6847
FpuBinningMode        = 0
PixelBinningMode      = 0
SubFrameMode          = 0
JailBars              = 0
DpuId                 = DPU-A
PivotAngle            = -18.805847167969 <Degrees>
Unlutted              = 1
LutInversionTable     = $messenger/calibration/LUT_INVERT/MDISLUTINV_0.TAB
End_Group

Group = Archive
DataSetId                 = MESS-E/V/H-MDIS-2-EDR-RAWDATA-V1.0
DataQualityId             = 0000000000000000
ProducerId                = "APPLIED COHERENT TECHNOLOGY CORPORATION"
EdrSourceProductId        = 0089576657_IM4WV
ProductId                 = EN0089576657M
SequenceName              = 07157_DEP_NAC_MOSAIC_1
ObservationId             = 3276
ObservationType           = N/A
SiteId                    = N/A
MissionElapsedTime        = 89576657
EdrProductCreationTime    = 2011-11-21T22:38:34
ObservationStartTime      = 2007-06-06T00:22:10.751814
SpacecraftClockStartCount = 1/0089576657:973000
SpacecraftClockStopCount  = 1/0089576657:990000
Exposure                  = 17
CCDTemperature            = 1022
OriginalFilterNumber      = 0
OrbitNumber               = 0
YearDoy                   = 2007157
SourceProductId           = (EN0089576657M, MDISLUTINV_0)
End_Group

Group = BandBin
Name   = "748 BP 53"
Number = 2
Center = 747.7 <NM>
Width  = 52.6 <NM>
End_Group

End_Object
"""

class testclass(Driver, IsisLabel):

    @property
    def detector_center_line(self):
        return 1
    @property
    def detector_center_sample(self): 
        return 1
    @property
    def focal2pixel_lines(self):
        return 1 
    @property
    def focal2pixel_samples(self):
        return 1
    @property
    def focal_length(self):
        return 1

    @property
    def image_lines(self):
        return 1

    @property
    def image_samples(self):
        return 1

    @property
    def isis_naif_keywords(self):
        return {"key" : "value"}

    @property
    def line_scan_rate(self):
        return ([0.0],[0],[0.0])

    @property
    def line_summing(self):
        return 1 

    @property
    def name_platform(self):
        return "test"

    @property
    def name_sensor(self):
        return "testsensor"

    @property
    def usgscsm_distortion_model(self): 
        return {'test': { 'coefficients' : [0.0, 0.0, 0.0] } }

    @property
    def rotation_chain(self):
        return "rotation chain object"

    @property
    def sample_summing(self):
        return 1

    @property
    def sensor_frame_id(self):
        return "test id"

    @property
    def sensor_model_version(self):
        return 1

    @property
    def sensor_position(self):
        return ([1],[1],[1])

    @property
    def sensor_type(self):
        return "test"

    @property
    def start_time(self):
        return 0

    @property
    def starting_detector_line(self):
        return 0

    @property
    def starting_detector_sample(self):
        return 0

    @property
    def stop_time(self):
        return 0

    @property
    def sun_position(self):
        return ([1], [1])

    @property
    def target_body_id(self):
        return 1

def test_me(test_label):
    me = testclass(test_label)
    assert me.image_lines == 1


