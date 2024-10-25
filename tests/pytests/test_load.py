import pytest
from importlib import reload
import json
import os

import ale
from ale import util
from ale.drivers import sort_drivers
from ale.base.data_naif import NaifSpice
from ale.base.data_isis import IsisSpice

from conftest import get_image_label, get_image_kernels, convert_kernels

@pytest.fixture()
def mess_kernels():
    kernels = get_image_kernels('EN1072174528M')
    updated_kernels, binary_kernels = convert_kernels(kernels)
    yield updated_kernels
    for kern in binary_kernels:
        os.remove(kern)

def test_priority(tmpdir, monkeypatch):
    drivers = [type('FooNaifSpice', (NaifSpice,), {}), type('BarIsisSpice', (IsisSpice,), {}), type('BazNaifSpice', (NaifSpice,), {}), type('FubarIsisSpice', (IsisSpice,), {})]
    sorted_drivers = sort_drivers(drivers)
    assert all([IsisSpice in klass.__bases__ for klass in sorted_drivers[2:]])

@pytest.mark.parametrize(("class_truth, return_val"), [({"only_isis_spice": False,  "only_naif_spice": False}, True), 
                                                       ({"only_isis_spice": True,  "only_naif_spice": False}, False)])
def test_mess_load(class_truth, return_val, mess_kernels):
    label_file = get_image_label('EN1072174528M')

    try:
        usgscsm_isd_str = ale.loads(label_file, {'kernels': mess_kernels}, 'usgscsm', False, **class_truth)
        usgscsm_isd_obj = json.loads(usgscsm_isd_str)

        assert return_val is True
        assert usgscsm_isd_obj['name_platform'] == 'MESSENGER'
        assert usgscsm_isd_obj['name_sensor'] == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'
        assert usgscsm_isd_obj['name_model'] == 'USGS_ASTRO_FRAME_SENSOR_MODEL'
    except Exception as load_failure:
        assert str(load_failure) == "No Such Driver for Label"
        assert return_val is False

def test_load_invalid_label():
    with pytest.raises(Exception):
        ale.load('Not a label path')

def test_loads_invalid_label():
    with pytest.raises(Exception):
        ale.loads('Not a label path')

def test_load_invalid_spice_root(monkeypatch):
    monkeypatch.delenv('ALESPICEROOT', raising=False)
    reload(ale)

    label_file = get_image_label('EN1072174528M')
    with pytest.raises(Exception):
        ale.load(label_file)


def test_load_mes_from_metakernels(tmpdir, monkeypatch, mess_kernels):
    monkeypatch.setenv('ALESPICEROOT', str(tmpdir))

    # reload module to repopulate ale.spice_root
    reload(ale)

    updated_kernels = mess_kernels
    label_file = get_image_label('EN1072174528M')
    tmpdir.mkdir('mess')
    with open(tmpdir.join('mess', 'mess_2015_v1.tm'), 'w+') as mk_file:
        mk_str = util.write_metakernel_from_kernel_list(updated_kernels)
        print(mk_str)
        mk_file.write(mk_str)

    usgscsm_isd_obj = ale.load(label_file, verbose=True)
    assert usgscsm_isd_obj['name_platform'] == 'MESSENGER'
    assert usgscsm_isd_obj['name_sensor'] == 'MERCURY DUAL IMAGING SYSTEM NARROW ANGLE CAMERA'
    assert usgscsm_isd_obj['name_model'] == 'USGS_ASTRO_FRAME_SENSOR_MODEL'

def test_load_mes_with_no_metakernels(tmpdir, monkeypatch, mess_kernels):
    monkeypatch.setenv('ALESPICEROOT', str(tmpdir))

    # reload module to repopulate ale.spice_root
    reload(ale)

    updated_kernels = mess_kernels
    label_file = get_image_label('EN1072174528M')
    tmpdir.mkdir('mes')

    # intentionally make an mk file with wrong year
    with open(tmpdir.join('mes', 'mes_2016_v1.tm'), 'w+') as mk_file:
        mk_str = util.write_metakernel_from_kernel_list(updated_kernels)
        print(mk_str)
        mk_file.write(mk_str)

    with pytest.raises(Exception):
        usgscsm_isd_obj = ale.load(label_file, verbose=True)
