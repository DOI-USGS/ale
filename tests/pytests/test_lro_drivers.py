import pytest
import numpy as np
import os
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import spiceypy as spice
import json

import ale
from ale import util
from ale.drivers.lro_drivers import LroLrocPds3LabelNaifSpiceDriver
from ale.drivers.lro_drivers import LroLrocIsisLabelNaifSpiceDriver
from ale.transformation import TimeDependentRotation

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts

image_dict = {
    'M103595705LE': {
    'isis': {
        "CameraVersion": 2,
        "NaifKeywords": {
            "BODY301_RADII": [ 1737.4, 1737.4, 1737.4 ],
            "BODY_FRAME_CODE": 31001,
            "BODY_CODE": 301,
            "INS-85600_MULTIPLI_LINE_ERROR": 0.0045,
            "INS-85600_CK_FRAME_ID": -85000,
            "TKFRAME_-85600_RELATIVE": "LRO_SC_BUS",
            "INS-85600_PIXEL_SAMPLES": 5064,
            "INS-85600_WAVELENGTH_RANGE": [ 400, 760 ],
            "INS-85600_ITRANSL": [ 0, 142.857, 0 ],
            "INS-85600_TRANSX": [ 0, 0, 0.007 ],
            "INS-85600_SWAP_OBSERVER_TARGET": "TRUE",
            "INS-85600_TRANSY": [ 0, 0.007, 0 ],
            "INS-85600_ITRANSS": [ 0, 0, 142.857 ],
            "INS-85600_LIGHTTIME_CORRECTION": "NONE",
            "INS-85600_ADDITIVE_LINE_ERROR": 0,
            "INS-85600_FOV_BOUNDARY_CORNERS": [
                0.0000049738,
                -0.025335999999999997,
                1,
                0.0000049747,
                -0.024943,
                1,
                0.0000050026999999999996,
                0,
                1,
                0.000004975
            ],
            "FRAME_-85600_NAME": "LRO_LROCNACL",
            "CK_-85600_SPK": -85,
            "INS-85600_CONSTANT_TIME_OFFSET": 0,
            "CK_-85600_SCLK": -85,
            "INS-85600_PIXEL_LINES": 1,
            "INS-85600_BORESIGHT": [ 0, 0, 1 ],
            "INS-85600_PLATFORM_ID": -85000,
            "INS-85600_BORESIGHT_LINE": 0,
            "FRAME_-85600_CLASS": 3,
            "INS-85600_FOCAL_LENGTH": 699.62,
            "INS-85600_F/RATIO": 3.577,
            "INS-85600_OD_K": 0.0000181,
            "INS-85600_FOV_SHAPE": "POLYGON",
            "INS-85600_PIXEL_SIZE": [ 0.007, 0.007 ],
            "INS-85600_BORESIGHT_SAMPLE": 2548,
            "INS-85600_PIXEL_PITCH": 0.007,
            "INS-85600_ADDITIONAL_PREROLL": 1024,
            "INS-85600_CK_REFERENCE_ID": 1,
            "INS-85600_LT_SURFACE_CORRECT": "TRUE",
            "INS-85600_FOV_FRAME": "LRO_LROCNACL",
            "FRAME_-85600_CLASS_ID": -85600,
            "INS-85600_IFOV": 0.000010005399999999999,
            "FRAME_-85600_CENTER": -85,
            "INS-85600_CCD_CENTER": [ 2532.5, 1 ],
            "BODY301_POLE_RA": [ 269.9949, 0.0031, 0 ],
            "BODY301_NUT_PREC_PM": [ 3.561, 0.1208, -0.0642, 0.0158, 0.0252, -0.0066, -0.0047, -0.0046, 0.0028, 0.0052 ],
            "BODY301_NUT_PREC_RA": [ -3.8787000000000003, -0.1204, 0.07, -0.0172, 0, 0.0072, 0, 0, 0, -0.0052 ],
            "BODY301_LONG_AXIS": 0,
            "BODY301_NUT_PREC_DEC": [ 1.5419, 0.0239, -0.0278, 0.0068, 0, -0.0029, 0.0009, 0, 0, 0.0008 ],
            "BODY301_POLE_DEC": [ 66.5392, 0.013, 0 ],
            "BODY301_PM": [ 38.3213, 13.17635815, -1.3999999999999999e-12 ]
        },
        "InstrumentPointing": {
            "TimeDependentFrames": [ -85600, -85000, 1 ],
            "CkTableStartTime": 302228504.36824864,
            "CkTableEndTime": 302228504.7816205,
            "CkTableOriginalSize": 6,
            "EphemerisTimes": [
                302228504.36824864,
                302228504.450923,
                302228504.5335974,
                302228504.61627173,
                302228504.6989461,
                302228504.7816205
            ],
            "Quaternions": [
                [ 0.22984449090659237, 0.8562360121526155, -0.014576324183122052, 0.4624055928145594 ],
                [ 0.22984459963498413, 0.8562197280950775, -0.014583335437683352, 0.4624354696246142 ],
                [ 0.22984367991941132, 0.8562032842671137, -0.014590196641096627, 0.4624661554895512 ],
                [ 0.2298423219602694, 0.8561872277925395, -0.01459745672913303, 0.46249632675068797 ],
                [ 0.2298413596492449, 0.8561711699907094, -0.014604593844944487, 0.4625263051005321 ],
                [ 0.22984081705372042, 0.8561549931881817, -0.014611505650658618, 0.4625562996626938 ]
            ],
            "AngularVelocity": [
                [ 0.00015796288676771882, -0.0007643782570599635, -0.0003174531428220953 ],
                [ 0.00015631294459532674, -0.0007657803825957866, -0.0003184081089449395 ],
                [ 0.00013745925497849103, -0.0007797744104491687, -0.00032988161600413016 ],
                [ 0.00013211442776748356, -0.0007623315159936997, -0.00033874000799855454 ],
                [ 0.00014047395809259157, -0.0007614279586831931, -0.0003284683667926366 ],
                [ 0.0001443115614238801, -0.0007630657146284228, -0.00032321391571062645 ]
            ],
            "ConstantFrames": [ -85600 ],
            "ConstantRotation": [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ]
        },
        "BodyRotation": {
            "TimeDependentFrames": [ 31006, 1 ],
            "CkTableStartTime": 302228504.36824864,
            "CkTableEndTime": 302228504.7816205,
            "CkTableOriginalSize": 6,
            "EphemerisTimes": [
                302228504.36824864,
                302228504.450923,
                302228504.5335974,
                302228504.61627173,
                302228504.6989461,
                302228504.7816205
            ],
            "Quaternions": [
                [ -0.8896294526439446, 0.18337484217069425, -0.07150784453785884, 0.41209189802380003 ],
                [ -0.8896294073127606, 0.1833748342493258, -0.07150786471014642, 0.41209199590986223 ],
                [ -0.889629361981566, 0.18337482632795524, -0.07150788488243316, 0.41209209379591966 ],
                [ -0.8896293166503605, 0.18337481840658243, -0.07150790505471902, 0.41209219168197186 ],
                [ -0.8896292713191442, 0.18337481048520735, -0.071507925227004, 0.4120922895680191 ],
                [ -0.8896292259879173, 0.18337480256383012, -0.0715079453992881, 0.41209238745406124 ]
            ],
            "AngularVelocity": [
                [ 6.23828510009553e-8, -0.0000010257490014652093, 0.000002455354036200098 ],
                [ 6.238285108687342e-8, -0.0000010257490015380855, 0.0000024553540362115642 ],
                [ 6.238285117279176e-8, -0.000001025749001610962, 0.0000024553540362230297 ],
                [ 6.238285125870996e-8, -0.0000010257490016838385, 0.000002455354036234496 ],
                [ 6.2382851344628e-8, -0.0000010257490017567146, 0.0000024553540362459614 ],
                [ 6.238285143054625e-8, -0.0000010257490018295912, 0.0000024553540362574277 ]
            ],
            "ConstantFrames": [ 31001, 31007, 31006 ],
            "ConstantRotation": [
                0.9999998732547144,
                -0.00032928542237557133,
                0.00038086961867138755,
                0.00032928600021094723,
                0.9999999457843062,
                -0.0000014544409378362713,
                -0.00038086911909607826,
                0.0000015798557868269087,
                0.9999999274681067
            ]
        },
        "InstrumentPosition": {
            "SpkTableStartTime": 302228504.36824864,
            "SpkTableEndTime": 302228504.7816205,
            "SpkTableOriginalSize": 6,
            "EphemerisTimes": [
                302228504.36824864,
                302228504.450923,
                302228504.5335974,
                302228504.61627173,
                302228504.6989461,
                302228504.7816205
            ],
            "Positions": [[-1516.1039882048947, -668.6745734893002, 902.1405183116759 ],
                [ -1516.176000573894, -668.6422150991707, 902.0346703324196 ],
                [ -1516.2480053780712, -668.6098533709328, 901.9288178499854 ],
                [ -1516.320002565111, -668.5774883280569, 901.8229609411912 ],
                [ -1516.3919922384162, -668.5451199240163, 901.7170994539005 ],
                [ -1516.4639743456696, -668.5127481822817, 901.6112334649276 ]],
            "Velocities": [[-0.8710817993164441, 0.3913754105818205, -1.2802723434988814 ],
                [ -0.8709903003882029, 0.39141578800691706, -1.2803268153571778 ],
                [ -0.8708987964969201, 0.3914561637581709, -1.2803812811841886 ],
                [ -0.8708072877088842, 0.391496537806338, -1.2804357409401614 ],
                [ -0.8707157738925385, 0.3915369102094339, -1.2804901947032974 ],
                [ -0.8706242551142269, 0.39157728093812155, -1.2805446424339852 ]]
        },
        "SunPosition": {
            "SpkTableStartTime": 302228504.5749346,
            "SpkTableEndTime": 302228504.5749346,
            "SpkTableOriginalSize": 1,
            "EphemerisTimes": [ 302228504.5749346 ],
            "Positions": [ [ -91885596.62561405, 111066639.06681778, 48186230.75049895 ] ],
            "Velocities": [ [ -23.97582426247181, -15.920790540011309, -6.940052709040858 ] ]
        }
    }
 }
}

@pytest.fixture(scope="module")
def test_kernels():
    updated_kernels = {}
    binary_kernels = {}
    for image in image_dict.keys():
        kernels = get_image_kernels(image)
        updated_kernels[image], binary_kernels[image] = convert_kernels(kernels)
    yield updated_kernels
    for kern_list in binary_kernels.values():
        for kern in kern_list:
            os.remove(kern)

@pytest.mark.parametrize("label_type", ['isis3'])
@pytest.mark.parametrize("formatter", ['isis'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_load_isis(test_kernels, label_type, formatter, image):
    label_file = get_image_label(image, label_type)
    isis_isd = ale.loads(label_file, props={'kernels': test_kernels[image]}, formatter=formatter, verbose=True)
    isis_isd_obj = json.loads(isis_isd)
    print(json.dumps(isis_isd_obj, indent=4))
    assert compare_dicts(isis_isd_obj, image_dict[image][formatter]) == []

# ========= Test pdslabel and naifspice driver =========
class test_pds_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'pds3')
        self.driver = LroLrocPds3LabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name=='lro'

    def test_instrument_id_left(self):
        self.driver.label['FRAME_ID'] = 'LEFT'
        assert self.driver.instrument_id == 'LRO_LROCNACL'

    def test_instrument_id_right(self):
        self.driver.label['FRAME_ID'] = 'RIGHT'
        assert self.driver.instrument_id == 'LRO_LROCNACR'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'LRO'

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 2

    def test_odtk(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.odtk == [1.0]
            gdpool.assert_called_with('INS-12345_OD_K', 0, 1)

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.odtk', \
                   new_callable=PropertyMock) as odtk:
            odtk.return_value = [1.0]
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=5) as scs2e, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_id', \
                   new_callable=PropertyMock) as spacecraft_id:
            exposure_duration.return_value = 0.1
            spacecraft_id.return_value = 1234
            assert self.driver.ephemeris_start_time == 107.4
            scs2e.assert_called_with(1234, "1/270649237:07208")

    def test_exposure_duration(self):
        with patch('ale.base.label_pds3.Pds3Label.exposure_duration', \
                   new_callable=PropertyMock) as exposure_duration:
            exposure_duration.return_value = 1
            assert self.driver.exposure_duration == 1.0045

    @patch('ale.transformation.FrameChain')
    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch('ale.transformation.FrameChain.compute_rotation', return_value=TimeDependentRotation([[0, 0, 1, 0]], [0], 0, 0))
    def test_spacecraft_direction(self, compute_rotation, from_spice, frame_chain):
        with patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.lro_drivers.spice.spkezr', return_value=[[1, 1, 1, 1, 1, 1], 0]) as spkezr, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            bods2c.assert_called_with('LRO_SC_BUS')
            spkezr.assert_called_with(self.driver.spacecraft_name, 0, 'J2000', 'None', self.driver.target_name)
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=[0, 1, 0]) as gdpool, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.ikid', \
             new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.lro_drivers.LroLrocPds3LabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])


# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label('M103595705LE', 'isis3')
        self.driver = LroLrocIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'lro'

    def test_intrument_id(self):
        assert self.driver.instrument_id == 'LRO_LROCNACL'

    def test_usgscsm_distortion_model(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            distortion_model = self.driver.usgscsm_distortion_model
            assert distortion_model['lrolrocnac']['coefficients'] == [1.0]
            gdpool.assert_called_with('INS-12345_OD_K', 0, 1)
            bods2c.assert_called_with('LRO_LROCNACL')

    def test_odtk(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
             assert self.driver.odtk == [1.0]
             gdpool.assert_called_with('INS-12345_OD_K', 0, 1)
             bods2c.assert_called_with('LRO_LROCNACL')

    def test_light_time_correction(self):
        assert self.driver.light_time_correction == 'NONE'

    def test_detector_center_sample(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=np.array([1.0])) as gdpool, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c:
            assert self.driver.detector_center_sample == 0.5
            gdpool.assert_called_with('INS-12345_BORESIGHT_SAMPLE', 0, 1)
            bods2c.assert_called_with('LRO_LROCNACL')

    def test_exposure_duration(self):
        np.testing.assert_almost_equal(self.driver.exposure_duration, .0010334296)

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.lro_drivers.spice.scs2e', return_value=321) as scs2e:
            np.testing.assert_almost_equal(self.driver.ephemeris_start_time, 322.05823191)
            scs2e.assert_called_with(-85, '1/270649237:07208')

    def test_multiplicative_line_error(self):
        assert self.driver.multiplicative_line_error == 0.0045

    def test_additive_line_error(self):
        assert self.driver.additive_line_error == 0

    def test_constant_time_offset(self):
        assert self.driver.constant_time_offset == 0

    def test_additional_preroll(self):
        assert self.driver.additional_preroll == 1024

    def test_sampling_factor(self):
        assert self.driver.sampling_factor == 1

    @patch('ale.transformation.FrameChain')
    @patch('ale.transformation.FrameChain.from_spice', return_value=ale.transformation.FrameChain())
    @patch('ale.transformation.FrameChain.compute_rotation', return_value=TimeDependentRotation([[0, 0, 1, 0]], [0], 0, 0))
    def test_spacecraft_direction(self, compute_rotation, from_spice, frame_chain):
        with patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.target_frame_id', \
             new_callable=PropertyMock) as target_frame_id, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.ephemeris_start_time', \
             new_callable=PropertyMock) as ephemeris_start_time, \
             patch('ale.drivers.lro_drivers.spice.cidfrm', return_value=[-12345]) as cidfrm, \
             patch('ale.drivers.lro_drivers.spice.scs2e', return_value=0) as scs2e, \
             patch('ale.drivers.lro_drivers.spice.bods2c', return_value=-12345) as bods2c, \
             patch('ale.drivers.lro_drivers.spice.spkezr', return_value=[[1, 1, 1, 1, 1, 1], 0]) as spkezr, \
             patch('ale.drivers.lro_drivers.spice.mxv', return_value=[1, 1, 1]) as mxv:
            ephemeris_start_time.return_value = 0
            assert self.driver.spacecraft_direction > 0
            spkezr.assert_called_with(self.driver.spacecraft_name, 0, 'J2000', 'None', self.driver.target_name)
            compute_rotation.assert_called_with(1, -12345)
            np.testing.assert_array_equal(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]), mxv.call_args[0][0])
            np.testing.assert_array_equal(np.array([1, 1, 1]), mxv.call_args[0][1])

    def test_focal2pixel_lines(self):
        with patch('ale.drivers.lro_drivers.spice.gdpool', return_value=[0, 1, 0]) as gdpool, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.ikid', \
             new_callable=PropertyMock) as ikid, \
             patch('ale.drivers.lro_drivers.LroLrocIsisLabelNaifSpiceDriver.spacecraft_direction', \
             new_callable=PropertyMock) as spacecraft_direction:
            spacecraft_direction.return_value = -1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, -1, 0])
            spacecraft_direction.return_value = 1
            np.testing.assert_array_equal(self.driver.focal2pixel_lines, [0, 1, 0])
