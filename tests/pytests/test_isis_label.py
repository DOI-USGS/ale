import pytest
import pvl

import ale
from ale import base
from ale.base import label_isis


@pytest.fixture
def test_cube(monkeypatch):
    label = """
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

    def test_label(file):
        return pvl.loads(label)
    monkeypatch.setattr(pvl, 'load', test_label)

    test_image = type('TestCubeDriver', (base.Driver, base.label_isis.IsisLabel), {})(label)
    return test_image

def test_spacecraft_clock_start_count(test_cube):
    assert test_cube.spacecraft_clock_start_count == "1/0089576657:973000"

def test_target_name(test_cube):
    assert test_cube.target_name.lower() == "venus"

def test_exposure_duration(test_cube):
    assert test_cube.exposure_duration == 0.017

def test_image_samples(test_cube):
    assert test_cube.image_samples == 1024

def test_image_lines(test_cube):
    assert test_cube.image_lines == 1024

def test_sample_summing(test_cube):
    assert test_cube.sample_summing == 1

def test_line_summing(test_cube):
    assert test_cube.line_summing == 1

