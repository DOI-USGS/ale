import pytest
import numpy as np
import os
import unittest
from unittest.mock import PropertyMock, patch

import json

import ale
from ale import util
from ale.drivers.ody_drivers import OdyThemisVisIsisLabelNaifSpiceDriver, OdyThemisIrIsisLabelNaifSpiceDriver

from conftest import get_image_label, get_image_kernels, convert_kernels, compare_dicts


image_dict = {
    "V46475015EDR": {
        "isis": {
            "CameraVersion": 1,
            "NaifKeywords": {
                "BODY499_RADII": [ 3396.19, 3396.19, 3376.2 ],
                "BODY_FRAME_CODE": 10014,
                "BODY_CODE": 499,
                "INS-53032_TRANSX": [ 0, 0.009, 0 ],
                "INS-53032_TRANSY": [ 0, 0, -0.009 ],
                "FRAME_-53032_CLASS_ID": -53032,
                "INS-53032_CK_TIME_BIAS": 0,
                "INS-53032_CK_FRAME_ID": -53000,
                "TKFRAME_-53032_RELATIVE": "M01_THEMIS_OPTICS",
                "FRAME_-53032_NAME": "M01_THEMIS_VIS",
                "TKFRAME_-53032_AXES": [ 1, 2, 3 ],
                "TKFRAME_-53032_SPEC": "ANGLES",
                "INS-53032_CK_TIME_TOLERANCE": 1,
                "FRAME_-53032_CLASS": 4,
                "INS-53032_PLATFORM_ID": -53000,
                "INS-53032_ITRANSL": [ 0, 0, -111.11111111 ],
                "INS-53032_SPK_TIME_BIAS": 0,
                "INS-53032_ITRANSS": [ 0, 111.11111111, 0 ],
                "TKFRAME_-53032_ANGLES": [ 0, 90, -0.25 ],
                "FRAME_-53032_CENTER": -53,
                "INS-53032_CK_REFERENCE_ID": 16,
                "TKFRAME_-53032_UNITS": "DEGREES",
                "BODY499_POLE_DEC": [ 52.8865, -0.0609, 0 ],
                "BODY499_POLE_RA": [ 317.68143, -0.1061, 0 ],
                "BODY499_PM": [ 176.63, 350.89198226, 0 ]
            },
            "InstrumentPointing": {
                "TimeDependentFrames": [ -53000, 16, 1 ],
                "CkTableStartTime": 392211095.6197291,
                "CkTableEndTime": 392211097.5397291,
                "CkTableOriginalSize": 6,
                "EphemerisTimes": [
                    392211095.6197291,
                    392211096.0037291,
                    392211096.3877291,
                    392211096.7717291,
                    392211097.1557291,
                    392211097.5397291
                ],
                "Quaternions": [
                    [ 0.622044822044232, 0.2555212953329484, 0.5142689567685136, -0.5322560916547194 ],
                    [ 0.6219555566538323, 0.2556095368638264, 0.5142243466922363, -0.532361128821991 ],
                    [ 0.6218662733649556, 0.25569777103883873, 0.5141797218177431, -0.532466150669111 ],
                    [ 0.6217764034754604, 0.25578530978663744, 0.514134694883009, -0.5325725254798925 ],
                    [ 0.6216862534405807, 0.25587252406502475, 0.5140894745361436, -0.5326795151709507 ],
                    [ 0.6215960852368998, 0.2559597308655292, 0.5140442391649964, -0.5327864892944326 ]
                ],
                "AngularVelocity": [
                    [ -0.0008105492062180801, -0.00019831431313117837, 0.0002945565809147707 ],
                    [ -0.0008108026806805358, -0.0001980750326173721, 0.00029502731883054514 ],
                    [ -0.0008114627704207764, -0.00019745190628475147, 0.0002962531988088855 ],
                    [ -0.0008119328383766381, -0.00019701160233906684, 0.0002972128619119053 ],
                    [ -0.0008120105314920341, -0.0001969488072413385, 0.00029762281538913397 ],
                    [ -0.0008119477826533893, -0.00019702113314222383, 0.0002978360123824115 ]
                ],
                "ConstantFrames": [ -53032, -53030, -53000 ],
                "ConstantRotation": [
                    0.0003684744441107557,
                    0.004882317815721682,
                    0.9999880135278275,
                    0.29095386119307015,
                    0.9567251530513545,
                    -0.004778302596747774,
                    -0.9567370144838278,
                    0.2909521343651026,
                    -0.0010680004511185448
                ]
            },
            "BodyRotation": {
                "TimeDependentFrames": [ 10014, 1 ],
                "CkTableStartTime": 392211095.6197291,
                "CkTableEndTime": 392211097.5397291,
                "CkTableOriginalSize": 6,
                "EphemerisTimes": [
                    392211095.6197291,
                    392211096.0037291,
                    392211096.3877291,
                    392211096.7717291,
                    392211097.1557291,
                    392211097.5397291
                ],
                "Quaternions": [
                    [ -0.6613202993937386, 0.31813079098385316, 0.010572073619126085, 0.6792175591790568 ],
                    [ -0.661311055603654, 0.31813093483380667, 0.0105677440557396, 0.6792265592743864 ],
                    [ -0.6613018116910848, 0.31813107862483736, 0.010563414490395608, 0.6792355592439125 ],
                    [ -0.6612925676560322, 0.3181312223569454, 0.010559084923095426, 0.6792445590876345 ],
                    [ -0.6612833234984982, 0.31813136603013065, 0.01055475535383936, 0.6792535588055496 ],
                    [ -0.6612740792172486, 0.3181315096444124, 0.010550425782049904, 0.67926255839886 ]
                ],
                "AngularVelocity": [
                    [ 0.000031623595509944944, -0.000028807313539993838, 0.00005651876115812607 ],
                    [ 0.00003162359550994386, -0.0000288073135400059, 0.00005651876115812055 ],
                    [ 0.00003162359550994274, -0.00002880731354001798, 0.00005651876115811503 ],
                    [ 0.000031623595509941664, -0.000028807313540030006, 0.000056518761158109486 ],
                    [ 0.00003162359550994056, -0.000028807313540042064, 0.00005651876115810396 ],
                    [ 0.00003162359550993948, -0.000028807313540054092, 0.00005651876115809842 ]
                ]
            },
            "InstrumentPosition": {
                "SpkTableStartTime": 392211095.6197291,
                "SpkTableEndTime": 392211097.5397291,
                "SpkTableOriginalSize": 6,
                "EphemerisTimes": [
                    392211095.6197291,
                    392211096.0037291,
                    392211096.3877291,
                    392211096.7717291,
                    392211097.1557291,
                    392211097.5397291
                ],
                "Positions": [
                    [ 87.5466120113472, 3034.785875639327, 2293.0758126687438 ],
                    [ 87.03005208648025, 3035.4979563295706, 2292.1348624958596 ],
                    [ 86.5134824945184, 3036.209687378141, 2291.1936480294244 ],
                    [ 85.99690275864972, 3036.921069649284, 2290.2521700219877 ],
                    [ 85.48031228284398, 3037.6321040072453, 2289.3104292410007 ],
                    [ 84.96371228899615, 3038.3427884701114, 2288.3684244645356 ]
                ],
                "Velocities": [
                    [ -1.345193813965703, 1.8548292421158314, -2.450048685058793 ],
                    [ -1.3452205668195178, 1.8539213021825771, -2.450735403178137 ],
                    [ -1.3452471645308965, 1.8530131393894345, -2.4514218450552074 ],
                    [ -1.345273607089621, 1.8521047539790478, -2.452108010497707 ],
                    [ -1.3452998944856378, 1.8511961461942064, -2.4527938993135434 ],
                    [ -1.3453260267206557, 1.8502873158548383, -2.4534795116298573 ]
                ]
            },
            "SunPosition": {
                "SpkTableStartTime": 392211096.5797291,
                "SpkTableEndTime": 392211096.5797291,
                "SpkTableOriginalSize": 1,
                "EphemerisTimes": [ 392211096.5797291 ],
                "Positions": [ [ 216567462.79095057, 97116302.56258373, 38695487.929268934 ] ],
                "Velocities": [ [ -11.437813079573214, 17.847439745021983, 8.495028880862519 ] ]
            }
        }
    },
    "I74199019RDR": {
        "isis": {
            "CameraVersion": 1,
            "NaifKeywords": {
                "BODY499_RADII": [ 3396.19, 3396.19, 3376.2 ],
                "BODY_FRAME_CODE": 10014,
                "BODY_CODE": 499,
                "TKFRAME_-53031_RELATIVE": "M01_THEMIS_OPTICS",
                "TKFRAME_-53031_UNITS": "DEGREES",
                "INS-53031_CK_FRAME_ID": -53000,
                "INS-53031_CK_TIME_BIAS": 0,
                "INS-53031_ITRANSL": [ 0, 0, 20 ],
                "INS-53031_ITRANSS": [ 0, 20, 0 ],
                "TKFRAME_-53031_ANGLES": [ -0.1701, 90.05331, -0.6315 ],
                "INS-53031_CK_TIME_TOLERANCE": 1,
                "FRAME_-53031_CENTER": -53,
                "FRAME_-53031_NAME": "M01_THEMIS_IR",
                "TKFRAME_-53031_AXES": [ 3, 2, 3 ],
                "TKFRAME_-53031_SPEC": "ANGLES",
                "INS-53031_TRANSX": [ 0, 0.05, 0 ],
                "INS-53031_TRANSY": [ 0, 0, 0.05 ],
                "INS-53031_PLATFORM_ID": -53000,
                "FRAME_-53031_CLASS_ID": -53031,
                "INS-53031_CK_REFERENCE_ID": 16,
                "FRAME_-53031_CLASS": 4,
                "INS-53031_SPK_TIME_BIAS": 0,
                "BODY499_POLE_DEC": [ 52.8865, -0.0609, 0 ],
                "BODY499_POLE_RA": [ 317.68143, -0.1061, 0 ],
                "BODY499_PM": [ 176.63, 350.89198226, 0 ]
            },
            "InstrumentPointing": {
                "TimeDependentFrames": [ -53000, 16, 1 ],
                "CkTableStartTime": 589445676.9899044,
                "CkTableEndTime": 589445686.0439956,
                "CkTableOriginalSize": 4,
                "EphemerisTimes": [ 589445676.9899044, 589445680.0079348, 589445683.0259652, 589445686.0439956 ],
                "Quaternions": [
                    [ 0.38598147685908185, -0.5607948046031743, -0.6866376818635513, -0.2550611309016295 ],
                    [ 0.38562579307122824, -0.5617138648879078, -0.6858904675855092, -0.25558667450829115 ],
                    [ 0.38527060742872654, -0.5626353429020607, -0.6851390429149169, -0.25611037043160223 ],
                    [ 0.3849151928868023, -0.563557151464003, -0.6843852032119891, -0.25663305504782297 ]
                ],
                "AngularVelocity": [
                    [ 0.0007324350225698707, -0.00006952577687208014, -0.0005029467858822215 ],
                    [ 0.0007335508142603166, -0.0000691981129943357, -0.0005044432717278556 ],
                    [ 0.0007368387903980461, -0.0000332990142650248, -0.0004221206727530788 ],
                    [ 0.0007393253232865181, -0.0000072290602625411406, -0.0003625424840601517 ]
                ],
                "ConstantFrames": [ -53031, -53030, -53000 ],
                "ConstantRotation": [
                    0.0013835021734054376,
                    0.01152997618685378,
                    0.9999325705120657,
                    0.2881133069543266,
                    0.9575279926584919,
                    -0.011439651709813592,
                    -0.957595325948064,
                    0.288109706404575,
                    -0.0019971975093594496
                ]
            },
            "BodyRotation": {
                "TimeDependentFrames": [ 10014, 1 ],
                "CkTableStartTime": 589445676.9899044,
                "CkTableEndTime": 589445686.0439956,
                "CkTableOriginalSize": 4,
                "EphemerisTimes": [ 589445676.9899044, 589445680.0079348, 589445683.0259652, 589445686.0439956 ],
                "Quaternions": [
                    [ -0.5408451126894908, 0.31557698159693326, -0.04183520300757272, 0.7785547819873538 ],
                    [ -0.5407618335951139, 0.3155725050030793, -0.041868957604701294, 0.7786126275639432 ],
                    [ -0.5406785483110855, 0.3155680247986294, -0.04190271172395818, 0.7786704642344462 ],
                    [ -0.5405952568468566, 0.31556354098409234, -0.04193646536151294, 0.7787282919923001 ]
                ],
                "AngularVelocity": [
                    [ 0.00003162303650185794, -0.000028813502069024346, 0.00005651591926299394 ],
                    [ 0.000031623036501849386, -0.000028813502069119034, 0.00005651591926295044 ],
                    [ 0.00003162303650184082, -0.000028813502069213752, 0.000056515919262906996 ],
                    [ 0.00003162303650183226, -0.000028813502069308437, 0.000056515919262863486 ]
                ]
            },
            "InstrumentPosition": {
                "SpkTableStartTime": 589445676.9899044,
                "SpkTableEndTime": 589445686.0439956,
                "SpkTableOriginalSize": 4,
                "EphemerisTimes": [ 589445676.9899044, 589445680.0079348, 589445683.0259652, 589445686.0439956 ],
                "Positions": [
                    [ -883.3647639832787, 3235.9253823963995, -1734.2678063014955 ],
                    [ -878.0820172062348, 3241.143181714679, -1727.2955108387216 ],
                    [ -872.7929026970128, 3246.3375924365314, -1720.3106940936013 ],
                    [ -867.4974605099152, 3251.5085747718745, -1713.3134080300226 ]
                ],
                "Velocities": [
                    [ 1.7493370110442228, 1.7327474299014805, 2.308134335038546 ],
                    [ 1.7514526413807416, 1.7250030655476496, 2.312291077819496 ],
                    [ 1.7535556099591456, 1.7172463879119024, 2.316431067882768 ],
                    [ 1.7556459015231667, 1.7094774543366804, 2.3205542750988553 ]
                ]
            },
            "SunPosition": {
                "SpkTableStartTime": 589445681.51695,
                "SpkTableEndTime": 589445681.51695,
                "SpkTableOriginalSize": 1,
                "EphemerisTimes": [ 589445681.51695 ],
                "Positions": [ [ -177834612.75017235, 94014077.38389017, 47921572.85876171 ] ],
                "Velocities": [ [ -13.267042406345718, -20.96669045246765, -9.258799613448325 ] ]
            }
        }
    }
}

@pytest.fixture(scope='module')
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
def test_load(test_kernels, label_type, formatter, image):
    label_file = get_image_label(image, label_type)
    usgscsm_isd_str = ale.loads(label_file, props={'kernels': test_kernels[image]}, formatter=formatter)
    usgscsm_isd_obj = json.loads(usgscsm_isd_str)
    print(json.dumps(usgscsm_isd_obj, indent=4))

    assert compare_dicts(usgscsm_isd_obj, image_dict[image][formatter]) == []

class test_themisir_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("I74199019RDR", "isis3")
        self.driver = OdyThemisIrIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "THEMIS_IR"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MARS ODYSSEY"

    def test_line_exposure_duration(self):
        assert self.driver.line_exposure_duration == 0.0332871

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.ody_drivers.spice.scs2e', return_value=0) as scs2e:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.ephemeris_start_time == 10
            scs2e.assert_called_with(-53, '1220641481.102')

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1


class test_themisvis_isis_naif(unittest.TestCase):
    def setUp(self):
        label = get_image_label("V46475015EDR", "isis3")
        self.driver = OdyThemisVisIsisLabelNaifSpiceDriver(label)

    def test_instrument_id(self):
        assert self.driver.instrument_id == "THEMIS_VIS"

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == "MARS ODYSSEY"

    def test_line_exposure_duration(self):
        assert self.driver.line_exposure_duration == 0.0048

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.mro_drivers.spice.scs2e', return_value=0) as scs2e:
            self.driver.label["IsisCube"]["Instrument"]["SpacecraftClockOffset"] = 10
            assert self.driver.ephemeris_start_time == (10 - self.driver.line_exposure_duration/2)
            scs2e.assert_called_with(-53, '1023406812.23')

    def test_sensor_model_version(self):
        assert self.driver.sensor_model_version == 1

