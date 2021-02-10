# Disabling all tests for this mix-in until it is re-written to align with 1.0 architecture
# - Jesse Mapel 5/23/19

# from collections import namedtuple
#
# import pytest
#
# import ale
# from ale.base import data_isis
# from ale import base
# from ale import util
#
# import pvl
# import numpy as np
# import struct
# from scipy.spatial.transform import Rotation
#
#
# @pytest.fixture
# def test_cube(monkeypatch):
#     label = """
# Object = IsisCube
# Object = Core
# StartByte   = 65537
# Format      = Tile
# TileSamples = 512
# TileLines   = 512
#
# Group = Dimensions
#   Samples = 1024
#   Lines   = 1024
#   Bands   = 1
# End_Group
#
# Group = Pixels
#   Type       = Real
#   ByteOrder  = Lsb
#   Base       = 0.0
#   Multiplier = 1.0
# End_Group
# End_Object
#
# Group = Instrument
# SpacecraftName        = Messenger
# InstrumentName        = "MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA"
# InstrumentId          = MDIS-NAC
# TargetName            = Venus
# OriginalTargetName    = VENUS
# StartTime             = 2007-06-06T00:22:10.751814
# StopTime              = 2007-06-06T00:22:10.768814
# SpacecraftClockCount  = 1/0089576657:973000
# MissionPhaseName      = "VENUS 2 FLYBY"
# ExposureDuration      = 17 <MS>
# ExposureType          = AUTO
# DetectorTemperature   = -43.65 <DEGC>
# FocalPlaneTemperature = -23.63 <DEGC>
# FilterTemperature     = N/A
# OpticsTemperature     = -24.72 <DEGC>
# AttitudeQuality       = Ok
# FilterWheelPosition   = 28320
# PivotPosition         = -6847
# FpuBinningMode        = 0
# PixelBinningMode      = 0
# SubFrameMode          = 0
# JailBars              = 0
# DpuId                 = DPU-A
# PivotAngle            = -18.805847167969 <Degrees>
# Unlutted              = 1
# LutInversionTable     = $messenger/calibration/LUT_INVERT/MDISLUTINV_0.TAB
# End_Group
#
# Group = Archive
# DataSetId                 = MESS-E/V/H-MDIS-2-EDR-RAWDATA-V1.0
# DataQualityId             = 0000000000000000
# ProducerId                = "APPLIED COHERENT TECHNOLOGY CORPORATION"
# EdrSourceProductId        = 0089576657_IM4WV
# ProductId                 = EN0089576657M
# SequenceName              = 07157_DEP_NAC_MOSAIC_1
# ObservationId             = 3276
# ObservationType           = N/A
# SiteId                    = N/A
# MissionElapsedTime        = 89576657
# EdrProductCreationTime    = 2011-11-21T22:38:34
# ObservationStartTime      = 2007-06-06T00:22:10.751814
# SpacecraftClockStartCount = 1/0089576657:973000
# SpacecraftClockStopCount  = 1/0089576657:990000
# Exposure                  = 17
# CCDTemperature            = 1022
# OriginalFilterNumber      = 0
# OrbitNumber               = 0
# YearDoy                   = 2007157
# SourceProductId           = (EN0089576657M, MDISLUTINV_0)
# End_Group
#
# Group = BandBin
# Name   = "748 BP 53"
# Number = 2
# Center = 747.7 <NM>
# Width  = 52.6 <NM>
# End_Group
#
# Group = Kernels
# NaifIkCode                = -236820
# LeapSecond                = $base/kernels/lsk/naif0012.tls
# TargetAttitudeShape       = $base/kernels/pck/pck00009.tpc
# TargetPosition            = (Table, $base/kernels/spk/de405.bsp)
# InstrumentPointing        = (Table,
#                              $messenger/kernels/ck/msgr_0706_v04.bc,
#                              $messenger/kernels/ck/msgr_mdis_sc040812_1504-
#                              30v1.bc,
#                              $messenger/kernels/ck/msgr_mdis_sc050727_1003-
#                              02_sub_v1.bc,
#                              $messenger/kernels/ck/msgr_mdis_gm040819_1504-
#                              30v1.bc, $messenger/kernels/fk/msgr_v231.tf)
# Instrument                = $messenger/kernels/ik/msgr_mdis_v160.ti
# SpacecraftClock           = $messenger/kernels/sclk/messenger_2548.tsc
# InstrumentPosition        = (Table,
#                              $messenger/kernels/spk/msgr_040803_150430_150-
#                              430_od431sc_2.bsp)
# InstrumentAddendum        = $messenger/kernels/iak/mdisAddendum009.ti
# ShapeModel                = Null
# InstrumentPositionQuality = Reconstructed
# InstrumentPointingQuality = Reconstructed
# CameraVersion             = 2
# End_Group
# End_Object
#
# Object = Label
# Bytes = 65536
# End_Object
#
# Object = Table
# Name                = InstrumentPointing
# StartByte           = 4267529
# Bytes               = 64
# Records             = 1
# ByteOrder           = Lsb
# TimeDependentFrames = (-236890, -236892, -236880, -236000, 1)
# ConstantFrames      = (-236820, -236800, -236890)
# ConstantRotation    = (0.001686595916635, 0.99996109494739,
#                      0.0086581745086423, 6.3008625209968e-04,
#                      -0.0086592477671008, 0.99996230949942,
#                      0.99999837919145, -0.0016810769512645,
#                      -6.44666390486019e-04)
# CkTableStartTime    = 234361395.94511
# CkTableEndTime      = 234361395.94511
# CkTableOriginalSize = 1
# FrameTypeCode       = 3
# Description         = "Created by spiceinit"
# Kernels             = ($messenger/kernels/ck/msgr_0706_v04.bc,
#                      $messenger/kernels/ck/msgr_mdis_sc040812_150430v1.bc,
#                      $messenger/kernels/ck/msgr_mdis_sc050727_100302_sub_v-
#                      1.bc,
#                      $messenger/kernels/ck/msgr_mdis_gm040819_150430v1.bc,
#                      $messenger/kernels/fk/msgr_v231.tf)
#
# Group = Field
# Name = J2000Q0
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Q1
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Q2
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Q3
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = AV1
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = AV2
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = AV3
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = ET
# Type = Double
# Size = 1
# End_Group
# End_Object
#
# Object = Table
# Name                 = InstrumentPosition
# StartByte            = 4267593
# Bytes                = 56
# Records              = 1
# ByteOrder            = Lsb
# CacheType            = Linear
# SpkTableStartTime    = 234361395.94511
# SpkTableEndTime      = 234361395.94511
# SpkTableOriginalSize = 1.0
# Description          = "Created by spiceinit"
# Kernels              = $messenger/kernels/spk/msgr_040803_150430_150430_od4-
#                      31sc_2.bsp
#
# Group = Field
# Name = J2000X
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Y
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Z
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000XV
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000YV
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000ZV
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = ET
# Type = Double
# Size = 1
# End_Group
# End_Object
#
# Object = Table
# Name                = BodyRotation
# StartByte           = 4267649
# Bytes               = 64
# Records             = 1
# ByteOrder           = Lsb
# TimeDependentFrames = (10012, 1)
# CkTableStartTime    = 234361395.94511
# CkTableEndTime      = 234361395.94511
# CkTableOriginalSize = 1
# FrameTypeCode       = 2
# PoleRa              = (272.76, 0.0, 0.0)
# PoleDec             = (67.16, 0.0, 0.0)
# PrimeMeridian       = (160.2, -1.4813688, 0.0)
# Description         = "Created by spiceinit"
# Kernels             = ($base/kernels/spk/de405.bsp,
#                      $base/kernels/pck/pck00009.tpc)
# SolarLongitude      = 330.75268490609
#
# Group = Field
# Name = J2000Q0
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Q1
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Q2
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Q3
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = AV1
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = AV2
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = AV3
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = ET
# Type = Double
# Size = 1
# End_Group
# End_Object
#
# Object = Table
# Name                 = SunPosition
# StartByte            = 4267713
# Bytes                = 56
# Records              = 1
# ByteOrder            = Lsb
# CacheType            = Linear
# SpkTableStartTime    = 234361395.94511
# SpkTableEndTime      = 234361395.94511
# SpkTableOriginalSize = 1.0
# Description          = "Created by spiceinit"
# Kernels              = $base/kernels/spk/de405.bsp
#
# Group = Field
# Name = J2000X
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Y
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000Z
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000XV
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000YV
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = J2000ZV
# Type = Double
# Size = 1
# End_Group
#
# Group = Field
# Name = ET
# Type = Double
# Size = 1
# End_Group
# End_Object
#
# Object = History
# Name      = IsisCube
# StartByte = 4267769
# Bytes     = 914
# End_Object
#
# Object = OriginalLabel
# Name      = IsisCube
# StartByte = 4259841
# Bytes     = 7688
# End_Object
#
# Object = NaifKeywords
# BODY299_RADII                              = (6051.8, 6051.8, 6051.8)
# BODY_FRAME_CODE                            = 10012
# INS-236820_SWAP_OBSERVER_TARGET            = TRUE
# INS-236820_LIGHTTIME_CORRECTION            = LT+S
# INS-236820_LT_SURFACE_CORRECT              = FALSE
# INS-236820_REFERENCE_FRAME                 = MSGR_SPACECRAFT
# INS-236820_FRAME                           = MSGR_MDIS_NAC
# INS-236820_FOCAL_LENGTH                    = 549.11781953727
# INS-236820_FL_TEMP_COEFFS                  = (549.51204973417,
#                                             0.010185643391234, 0.0, 0.0,
#                                             0.0, 0.0)
# TempDependentFocalLength                   = 549.27136298083462
# INS-236820_PIXEL_PITCH                     = 0.014
# CLOCK_ET_-236_1/0089576657:973000_COMPUTED = 0000000000002040
# INS-236820_TRANSX                          = (0.0, 0.014, 0.0)
# INS-236820_TRANSY                          = (0.0, 0.0, 0.014)
# INS-236820_ITRANSS                         = (0.0, 71.42857143, 0.0)
# INS-236820_ITRANSL                         = (0.0, 0.0, 71.42857143)
# INS-236820_BORESIGHT_SAMPLE                = 512.5
# INS-236820_BORESIGHT_LINE                  = 512.5
# INS-236820_OD_T_X                          = (0.0, 1.0018542696238, 0.0,
#                                             0.0, -5.09444047494111e-04,
#                                             0.0, 1.00401047146886e-05, 0.0,
#                                             1.00401047146886e-05, 0.0)
# INS-236820_OD_T_Y                          = (0.0, 0.0, 1.0,
#                                             9.06001059499675e-04, 0.0,
#                                             3.57484262662076e-04, 0.0,
#                                             1.00401047146886e-05, 0.0,
#                                             1.00401047146886e-05)
# End_Object
# End
# """
#
#     def test_table_data(table_label, file):
#         count = table_label['Records'] * len(table_label.getlist('Field'))
#         doubles = list(range(count))
#         return struct.pack('d' * count, *doubles)
#     monkeypatch.setattr(data_isis, 'read_table_data', test_table_data)
#
#     def test_label(file):
#         return pvl.loads(label)
#     monkeypatch.setattr(pvl, 'load', test_label)
#
#     test_image = type('TestCubeDriver', (base.Driver, data_isis.IsisSpice), {})(label)
#     # test_image._file = 'testfile.cub'
#     return test_image
#
# def test_read(test_cube):
#     assert isinstance(test_cube.label, pvl.PVLModule)
#
# def test_number_of_quaternions(test_cube):
#     assert test_cube.number_of_quaternions == 1
#
# def test_number_of_ephemerides(test_cube):
#     assert test_cube.number_of_ephemerides == 1
#
# def test_ephemeris_start_time(test_cube):
#     assert test_cube.ephemeris_start_time == 8.0
#
# def test_detector_center(test_cube):
#     assert test_cube.detector_center_line  == 512.5
#     assert test_cube.detector_center_sample  == 512.5
#
# def test_ikid(test_cube):
#     assert test_cube.ikid == -236820
#
# def test_focal2pixel_lines(test_cube):
#     assert test_cube.focal2pixel_lines == [0.0, 0.0, 71.42857143]
#
# def test_focal2pixel_samples(test_cube):
#     assert test_cube.focal2pixel_samples == [0.0, 71.42857143, 0.0]
#
# def test_focal_length(test_cube):
#     assert test_cube._focal_length == 549.11781953727
#
# def test_body_radii(test_cube):
#     assert test_cube._body_radii == [6051.8, 6051.8, 6051.8]
#
# def test_semimajor(test_cube):
#     assert test_cube._semimajor == 6051.8
#
# def test_semiminor(test_cube):
#     assert test_cube._semiminor == 6051.8
#
# def test_reference_frame(test_cube):
#     assert test_cube.reference_frame == 10012
#
# def test_sun_position(test_cube):
#     np.testing.assert_equal(test_cube._sun_position, np.array([[0, 1, 2]]))
#
# def test_sun_velocity(test_cube):
#     np.testing.assert_equal(test_cube._sun_velocity, np.array([[3, 4, 5]]))
#
# def test_sensor_position(test_cube):
#     #Make sure the tables are all parsed and loaded before we overwrite anything
#     test_cube.body_orientation_table
#     #Overwrite the body rotation, which is (0, 1, 2, 3) at this point
#     test_cube._body_orientation_table['Rotations'] = [[1.0/np.sqrt(2), 1.0/np.sqrt(2), 0, 0]]
#     np.testing.assert_almost_equal(test_cube._sensor_position, np.asarray([[0.0, -2000.0, 1000.0]]))
#
# def test_sensor_velocity(test_cube):
#     #Make sure the tables are all parsed and loaded before we overwrite anything
#     test_cube.body_orientation_table
#     #Overwrite the body rotation, which is (0, 1, 2, 3) at this point
#     test_cube._body_orientation_table['Rotations'] = [[1.0/np.sqrt(2), 1.0/np.sqrt(2), 0, 0]]
#     np.testing.assert_almost_equal(test_cube._sensor_velocity, np.array([[3000, -5000, 4000]]))
#
# def test_sensor_orientation(test_cube):
#     #Make sure the tables are all parsed and loaded before we overwrite anything
#     test_cube.body_orientation_table
#     test_cube.inst_pointing_table
#     #Overwrite the body rotation, which is (0, 1, 2, 3) at this point
#     test_cube._body_orientation_table['Rotations'] = [[1.0/np.sqrt(2), 1.0/np.sqrt(2), 0, 0]]
#     #Overwrite the instrument pointing, which is (0, 1, 2, 3) at this point
#     test_cube._inst_pointing_table["Rotations"] = [[1.0/np.sqrt(2), 0, 1.0/np.sqrt(2), 0]]
#     print('inverse body rotation', test_cube._body_j2k2bf_rotation.inv().as_matrix())
#     np.testing.assert_almost_equal(test_cube._sensor_orientation, np.asarray([1/2, -1/2, -1/2, -1/2]))
#
# def test_body_orientation(test_cube):
#     np.testing.assert_equal(test_cube.body_orientation, np.asarray([[0, 1, 2, 3]]))
#
# def test_naif_keywords(test_cube):
#     assert isinstance(test_cube.naif_keywords, pvl.PVLObject)
