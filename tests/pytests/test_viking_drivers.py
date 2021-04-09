import pytest
import os
import numpy as np
import spiceypy as spice
from importlib import reload
import json

import unittest
from unittest.mock import patch
from conftest import get_image, get_isd, get_image_label, get_image_kernels, convert_kernels, compare_dicts

import ale
from ale.drivers.viking_drivers import VikingIsisLabelNaifSpiceDriver

image_dict = {
    # Viking Orbiter 1 VISCA
    'f004a47': {
    'isis': {
        'CameraVersion': 1,
        'NaifKeywords': {
            'BODY_CODE': 499,
            'BODY499_RADII': [3396.19, 3396.19, 3376.2],
            'BODY_FRAME_CODE': 10014,
            'INS-27001_TRANSX': [0.0, 0.011764705882353, 0.0],
            'INS-27001_TRANSY': [0.0, 0.0, 0.011764705882353],
            'INS-27001_ITRANSS': [0.0, 85.0, 0.0],
            'INS-27001_ITRANSL': [0.0, 0.0, 85.0],
            'FRAME_-27001_CLASS_ID': -27001.0,
            'INS-27001_CK_REFERENCE_ID': 2.0,
            'TKFRAME_-27001_ANGLES': [-7.073500e-01,  7.580000e-03, -8.973569e+01],
            'FRAME_-27001_CENTER': -27.0,
            'TKFRAME_-27001_UNITS': 'DEGREES',
            'TKFRAME_-27001_RELATIVE': 'VO1_PLATFORM',
            'INS-27001_CK_FRAME_ID': -27000.0,
            'FRAME_-27001_NAME': 'VO1_VISA',
            'TKFRAME_-27001_AXES': [1., 2., 3.],
            'TKFRAME_-27001_SPEC': 'ANGLES',
            'FRAME_-27001_CLASS': 4.0,
            'BODY499_POLE_DEC': [52.8865, -0.0609, 0.0],
            'BODY499_POLE_RA': [317.68143, -0.1061, 0.0],
            'BODY499_PM': [176.63, 350.89198226, 0.0]},
        'InstrumentPointing': {
            'TimeDependentFrames': [-27000, 2, 1],
            'ConstantFrames': [-27001, -27000],
            'ConstantRotation': [0.0046130633441499, 0.99991314725849,
                                 0.012345751747229, -0.99998935101548,
                                 0.0046143450547129, -7.53349414399155e-05,
                                 -1.32295956915258e-04, -0.012345272752653,
                                 0.99992378546489],
            'CkTableStartTime': -742324621.5707,
            'CkTableEndTime': -742324621.5707,
            'CkTableOriginalSize': 1,
            'EphemerisTimes': [-742324621.5707],
            'Quaternions': [[0.52681010146894,-0.23107660776762,0.74879344146168,-0.32921588715745]],
            'AngularVelocity': [[0.0,0.0,0.0]]},
        'BodyRotation': {
            'TimeDependentFrames': [10014, 1],
            'CkTableStartTime': -742324621.5707,
            'CkTableEndTime': -742324621.5707,
            'CkTableOriginalSize': 1,
            'EphemerisTimes': [-742324621.5707],
            'Quaternions': [[-0.61809902466787,0.31800072486684,-0.0089011036547319,0.71885318740513]],
            'AngularVelocity': [[3.16267887431502e-05,-2.87717183114938e-05,5.65351035056844e-05]]},
        'InstrumentPosition': {
            'SpkTableStartTime': -742324621.5707,
            'SpkTableEndTime': -742324621.5707,
            'SpkTableOriginalSize': 1.0,
            'EphemerisTimes': [-742324621.5707],
            'Positions': [[2968.0235853136,3479.4553212221,1904.2575671941]],
            'Velocities': [[-1.9793481062728,-0.62668469080666,3.3004150826369]]},
        'SunPosition': {
            'SpkTableStartTime': -742324621.5707,
            'SpkTableEndTime': -742324621.5707,
            'SpkTableOriginalSize': 1.0,
            'EphemerisTimes': [-742324621.5707],
            'Positions': [[244748920.95492,-35758272.270036,-23031611.614447]],
            'Velocities': [[3.182248924045,19.865617818691,9.025318966287]]}}},

    # Viking Orbiter 1 VISCB
    'f735a00': {
    'isis': {
        'CameraVersion': 1,
        'NaifKeywords': {'BODY_CODE': 499,
                         'BODY499_RADII': [3396.19, 3396.19, 3376.2],
                         'BODY_FRAME_CODE': 10014,
                         'INS-27002_TRANSX' : [0.0, 0.011764705882353, 0.0],
                         'INS-27002_TRANSY' : [0.0, 0.0, 0.011764705882353],
                         'INS-27002_ITRANSS' : [0.0, 85.0, 0.0],
                         'INS-27002_ITRANSL' : [0.0, 0.0, 85.0],
                         'INS-27002_CK_REFERENCE_ID': 2.0,
                         'TKFRAME_-27002_ANGLES': [ 6.81000e-01,  3.20000e-02, -9.00228e+01],
                         'FRAME_-27002_CENTER': -27.0,
                         'FRAME_-27002_NAME': 'VO1_VISB',
                         'TKFRAME_-27002_RELATIVE': 'VO1_PLATFORM',
                         'INS-27002_CK_FRAME_ID': -27000.0,
                         'TKFRAME_-27002_AXES': [1., 2., 3.],
                         'TKFRAME_-27002_SPEC': 'ANGLES',
                         'FRAME_-27002_CLASS_ID': -27002.0,
                         'FRAME_-27002_CLASS': 4.0,
                         'TKFRAME_-27002_UNITS': 'DEGREES',
                         'BODY499_POLE_DEC': [52.8865, -0.0609,  0.    ],
                         'BODY499_POLE_RA': [ 3.1768143e+02, -1.0610000e-01,  0.0000000e+00],
                         'BODY499_PM': [176.63      , 350.89198226,   0.        ]
                         },
        'InstrumentPointing': {'TimeDependentFrames': [-27000, 2, 1],
                               'CkTableStartTime': -679343589.99066,
                               'CkTableEndTime': -679343589.99066,
                               'CkTableOriginalSize': 1,
                               'ConstantFrames': [-27002, -27000],
                               'ConstantRotation': [-3.97934996888762e-04, 0.99992928417985, -0.011885633652183, -0.99999976485974,
                                                    -4.04545016850005e-04, -5.53736215647129e-04,-5.58505331602587e-04, 0.011885410506373,
                                                    0.99992921003884],
                               'EphemerisTimes': [-679343589.99066],
                               'Quaternions': [[0.027631069885584,-0.042844653015238,0.63550118209549,0.77041489292473]],
                               'AngularVelocity': [[0.0,0.0,0.0]]},
        'BodyRotation': {'TimeDependentFrames': [10014, 1],
                         'CkTableStartTime': -679343589.99066,
                         'CkTableEndTime': -679343589.99066,
                         'CkTableOriginalSize': 1,
                         'EphemerisTimes': [-679343589.99066],
                         'Quaternions': [[-0.7279379541667,0.013628253415821,0.31784329352737,-0.60736829547822]],
                         'AngularVelocity': [[3.16266124745133e-05,-2.87736941823106e-05,5.6534196515818e-05]]},
        'InstrumentPosition': {'SpkTableStartTime': -679343589.99066,
                               'SpkTableEndTime': -679343589.99066,
                               'SpkTableOriginalSize': 1,
                               'EphemerisTimes': [-679343589.99066],
                               'Positions': [[3804.7657059701,-29029.379045591,-8152.7028148086]],
                               'Velocities': [[0.61887065223952,-0.54866918471373,-0.037514399215278]]},
        'SunPosition': {'SpkTableStartTime': -679343589.99066,
                       'SpkTableEndTime': -679343589.99066,
                       'SpkTableOriginalSize': 1,
                       'EphemerisTimes': [-679343589.99066],
                       'Positions': [[242154815.26715,36901072.557543,10366793.725074]],
                       'Velocities': [[-4.6653907628395,19.836822435663,9.2246820568878]]}}},

    # Viking Orbiter 2 VISCA
    'f004b65': {
    'isis': {
    'CameraVersion': 1,
    'NaifKeywords': {'BODY_CODE': 499,
                     'BODY499_RADII': [3396.19, 3396.19, 3376.2],
                     'BODY_FRAME_CODE': 10014,
                     'INS-30001_TRANSX': [0.0, 0.011764705882353, 0.0],
                     'INS-30001_TRANSY': [0.0, 0.0, 0.011764705882353],
                     'INS-30001_ITRANSS': [0.0, 85.0, 0.0],
                     'INS-30001_ITRANSL': [0.0, 0.0, 85.0],
                     'TKFRAME_-30001_UNITS': 'DEGREES',
                     'TKFRAME_-30001_ANGLES': [-6.7933000e-01,  2.3270000e-02, -8.9880691e+01],
                     'FRAME_-30001_CENTER': -30.0,
                     'FRAME_-30001_NAME': 'VO2_VISA',
                     'TKFRAME_-30001_AXES': [1., 2., 3.],
                     'TKFRAME_-30001_SPEC': 'ANGLES',
                     'FRAME_-30001_CLASS_ID': -30001.0,
                     'TKFRAME_-30001_RELATIVE': 'VO2_PLATFORM',
                     'INS-30001_CK_FRAME_ID': -30000.0,
                     'FRAME_-30001_CLASS': 4.0,
                     'INS-30001_CK_REFERENCE_ID': 2.0,
                     'BODY499_POLE_DEC': [52.8865, -0.0609, 0.0],
                     'BODY499_POLE_RA': [317.68143, -0.1061, 0.0],
                     'BODY499_PM': [176.63, 350.89198226, 0.0]},
    'InstrumentPointing': {'TimeDependentFrames': [-30000, 2, 1],
                           'ConstantFrames': [-30001, -30000],
                           'ConstantRotation': [0.0020823332006485, 0.99992753405817,
                                                0.011857087365694, -0.99999774946761,
                                                0.0020870022808706, -3.81419977354999e-04,
                                                -4.06138105773791e-04, -0.011856266437452,
                                                0.999929629523],
                           'CkTableStartTime': -738066654.57901,
                           'CkTableEndTime': -738066654.57901,
                           'CkTableOriginalSize': 1,
                           'EphemerisTimes': [-738066654.57901],
                           'Quaternions': [[0.24875871307312,-0.29788492418696,0.65237809296024,0.65098886199219]],
                           'AngularVelocity': [[0.0,0.0,0.0]]},
    'BodyRotation': {'TimeDependentFrames': [10014, 1],
                     'CkTableStartTime': -738066654.57901,
                     'CkTableEndTime': -738066654.57901,
                     'CkTableOriginalSize': 1,
                     'EphemerisTimes': [-738066654.57901],
                     'Quaternions': [[-0.53502287519129,0.31507831776488,-0.043929225721799,0.7826534353237]],
                     'AngularVelocity': [[3.16267768298231e-05,-2.87718518940354e-05,5.65350421875015e-05]]},
    'InstrumentPosition': {'SpkTableStartTime': -738066654.57901,
                           'SpkTableEndTime': -738066654.57901,
                           'SpkTableOriginalSize': 1.0,
                           'EphemerisTimes': [-738066654.57901],
                           'Positions': [[4553.9327290117,-1511.8974073078,1092.1996085889]],
                           'Velocities': [[-0.24727917113306,2.4028815935793,3.0999864190804]]},
    'SunPosition': {'SpkTableStartTime': -738066654.57901,
                    'SpkTableEndTime': -738066654.57901,
                    'SpkTableOriginalSize': 1.0,
                    'EphemerisTimes': [-738066654.57901],
                    'Positions': [[238736005.86526,49393960.380247,16187072.695853]],
                    'Velocities': [[-6.0452401714101,19.592116149966,9.1498650350044]]}}},

    # Viking Orbiter 2 VISCB
    'f704b28': {
    'isis': {
    'CameraVersion': 1,
    'NaifKeywords': {'BODY_CODE': 499,
                     'BODY499_RADII': [3396.19, 3396.19, 3376.2],
                     'BODY_FRAME_CODE': 10014,
                     'INS-30002_TRANSX': [0.0, 0.011764705882353, 0.0],
                     'INS-30002_TRANSY': [0.0, 0.0, 0.011764705882353],
                     'INS-30002_ITRANSS': [0.0, 85.0, 0.0],
                     'INS-30002_ITRANSL': [0.0, 0.0, 85.0],
                     'TKFRAME_-30002_AXES': [1., 2., 3.],
                     'TKFRAME_-30002_SPEC': 'ANGLES',
                     'TKFRAME_-30002_ANGLES':[ 6.630000e-01,  4.400000e-02, -8.966379e+01],
                     'FRAME_-30002_CENTER': -30.0,
                     'FRAME_-30002_CLASS': 4.0,
                     'INS-30002_CK_FRAME_ID': -30000.0,
                     'FRAME_-30002_CLASS_ID': -30002.0,
                     'TKFRAME_-30002_UNITS': 'DEGREES',
                     'TKFRAME_-30002_RELATIVE': 'VO2_PLATFORM',
                     'INS-30002_CK_REFERENCE_ID': 2.0,
                     'FRAME_-30002_NAME': 'VO2_VISB',
                     'BODY499_POLE_DEC': [52.8865, -0.0609, 0.0],
                     'BODY499_POLE_RA': [ 317.68143, -0.1061, 0.0],
                     'BODY499_PM': [176.63, 350.89198226, 0.0]},
    'InstrumentPointing': {'TimeDependentFrames': [-30000, 2, 1],
                           'ConstantFrames': [-30002, -30000],
                           'ConstantRotation': [0.0058679360725137, 0.99991588736024,
                                                -0.011566569536285, -0.9999824886402,
                                                0.0058586590004418, -8.35779681749578e-04,
                                                -7.67944795396292e-04, 0.011571271291668,
                                                0.99993275570985],
                           'CkTableStartTime': -676709864.3045,
                           'CkTableEndTime': -676709864.3045,
                           'CkTableOriginalSize': 1,
                           'EphemerisTimes': [-676709864.3045],
                           'Quaternions': [[0.14671376032888,-0.20207775752139,0.96766874650245,0.03545207310812]],
                           'AngularVelocity': [[0.0,0.0,0.0]]},
    'BodyRotation': {'TimeDependentFrames': [10014, 1],
                      'CkTableStartTime': -676709864.3045,
                      'CkTableEndTime': -676709864.3045,
                      'CkTableOriginalSize': 1,
                      'EphemerisTimes': [-676709864.3045],
                      'Quaternions': [[-0.028648237381698,0.24168074000221,-0.20688348409661,0.94761222154157]],
                      'AngularVelocity': [[3.16266051008033e-05,-2.87737768090931e-05,5.6534158586979e-05]]},
    'InstrumentPosition': {'SpkTableStartTime': -676709864.3045,
                           'SpkTableEndTime': -676709864.3045,
                           'SpkTableOriginalSize': 1.0,
                           'EphemerisTimes': [-676709864.3045],
                           'Positions': [[3297.8743337482,966.46050377782,10725.740343691]],
                           'Velocities': [[-0.44211958524766,1.2883896612093,1.9012944589374]]},
    'SunPosition': {'SpkTableStartTime': -676709864.3045,
                    'SpkTableEndTime': -676709864.3045,
                    'SpkTableOriginalSize': 1.0,
                    'EphemerisTimes': [-676709864.3045],
                    'Positions': [[222364052.49946,87428117.498071,34077452.005079]],
                    'Velocities': [[-10.327617595005,18.315940117221,8.6804664968805]]}}}
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

@pytest.mark.parametrize("label_type, kernel_type", [('isis3', 'naif'), ('isis3', 'isis')])
@pytest.mark.parametrize("formatter", ['isis'])
@pytest.mark.parametrize("image", image_dict.keys())
def test_viking1_load(test_kernels, label_type, formatter, image, kernel_type):
    if kernel_type == "naif":
        label_file = get_image_label(image, label_type)
        isis_isd = ale.loads(label_file, props={'kernels': test_kernels[image]}, formatter=formatter)
        compare_dict = image_dict[image][formatter]
    else: 
        label_file = get_image(image)
        isis_isd = ale.loads(label_file, verbose=True)
        isd_name = image+'_isis'
        compare_dict = get_isd(isd_name)

    print(isis_isd)
    assert compare_dicts(json.loads(isis_isd), compare_dict) == []

# ========= Test isislabel and naifspice driver =========
class test_isis_naif(unittest.TestCase):

    def setUp(self):
        label = get_image_label("f735a00", "isis3")
        self.driver = VikingIsisLabelNaifSpiceDriver(label)

    def test_short_mission_name(self):
        assert self.driver.short_mission_name == 'viking'

    def test_spacecraft_name(self):
        assert self.driver.spacecraft_name == 'VIKING ORBITER 1'

    def test_ikid(self):
        assert self.driver.ikid == -27002

    def test_alt_ikid(self):
        assert self.driver.alt_ikid == -27999

    def test_ephemeris_start_time(self):
        with patch('ale.drivers.viking_drivers.spice.scs2e', return_value=54321) as scs2e:
             assert self.driver.ephemeris_start_time == 54324.99
             scs2e.assert_called_with(-27999, '40031801')

    @patch('ale.base.label_isis.IsisLabel.exposure_duration', 0.43)
    def test_ephemeris_start_time_different_exposure(self):
        with patch('ale.drivers.viking_drivers.spice.scs2e', return_value=54321) as scs2e:
             assert self.driver.ephemeris_start_time == 54322.75
             scs2e.assert_called_with(-27999, '40031801')
