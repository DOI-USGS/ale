import os

import pytest
import tempfile
import spiceypy as spice
import pvl
from unittest import mock

from ale import util

@pytest.fixture
def cube_kernels():
   return """
    Object = IsisCube
    Group = Kernels
      TargetAttitudeShape = attitudeshape
      TargetPosition = (targetposition0, targetposition1)
      Instrument = instrument
      InstrumentPointing = (Table, instrumentpointing0, instrumentpointing1)
      SpacecraftClock = clock
      InstrumentPosition = instrumentposition
      InstrumentAddendum = Null
      ShapeModel = Null
    End_Group
    End_Object
    End
    """

@pytest.fixture
def pvl_one_group():
    return  """
    Group = Test
        t = t1
    EndGroup
    """

@pytest.fixture
def pvl_two_group():
    return """
    Group = Data
      b = b2
      a = a2
      r = r2
    End_Group

    Group = Test
      t = t2
    End_Group
    """

@pytest.fixture
def pvl_three_group():
    # intentionally mixup End_Group and EndGroup since
    # isis likes to mix them up as well
    return """
    Group = Data
      b = b3
      a = a3
      r = r3
    EndGroup

    Group = Test
      t = t3
    EndGroup

    Group = Settings
      delsystem32 = yes
    End_Group
    """

def test_kernel_from_cube_order(cube_kernels):
    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = util.generate_kernels_from_cube(cube.name)
    assert kernels == ['targetposition0', 'targetposition1','instrumentposition', 'instrumentpointing0', 'instrumentpointing1', 'attitudeshape', 'instrument', 'clock']

def test_kernel_from_cube_no_kernel_group():
    with pytest.raises(KeyError):
       with tempfile.NamedTemporaryFile('w+') as cube:
           cube.write('')
           cube.flush()
           util.generate_kernels_from_cube(cube.name)

def test_get_preferences_arg(tmpdir, pvl_one_group):
    with open(tmpdir.join('IsisPrefrences'), 'w+') as pvl_file:
        pvl_file.write(pvl_one_group)
        pvl_file.flush()

        pvl_obj = util.get_isis_preferences(pvl_file.name)
        pvl_obj_from_dict = util.get_isis_preferences({**pvl_obj})

        assert pvl_obj['Test']['t'] == 't1'
        assert pvl_obj == pvl_obj_from_dict

def test_get_prefernces_arg_isisroot(monkeypatch, tmpdir, pvl_one_group, pvl_two_group):
    monkeypatch.setenv('ISISROOT', str(tmpdir))

    with open(tmpdir.join('IsisPreferences'), 'w+') as pvl_isisroot_file:
        with open(tmpdir.join('arg_prefs.pvl'), 'w+') as pvl_arg_file:
            pvl_arg_file.write(pvl_two_group)
            pvl_arg_file.flush()

            pvl_isisroot_file.write(pvl_one_group)
            pvl_isisroot_file.flush()

            pvl_obj = util.get_isis_preferences(pvl_arg_file.name)

            assert pvl_obj['Test']['t'] == 't2'
            assert pvl_obj['Data']['b'] == 'b2'
            assert pvl_obj['Data']['a'] == 'a2'
            assert pvl_obj['Data']['r'] == 'r2'

def test_dict_to_lower():
    data = {'F': {'O' : {'O': 1}, 'B': {'A': 2}}}
    expected  = {'f': {'o' : {'o': 1}, 'b': {'a': 2}}}
    assert util.dict_to_lower(data) == expected

def test_get_prefernces_arg_isisroot_home(monkeypatch, tmpdir, pvl_one_group, pvl_two_group, pvl_three_group):
    tmpdir.mkdir('.Isis')
    monkeypatch.setenv('ISISROOT', str(tmpdir))
    monkeypatch.setenv('HOME', str(tmpdir))

    with open(tmpdir.join('arg_prefs.pvl'), 'w+') as pvl_arg_file:
        with open(tmpdir.join('.Isis', 'IsisPreferences'), 'w+') as pvl_home_file:
            with open(tmpdir.join('IsisPreferences'), 'w+') as pvl_isisroot_file:
                pvl_arg_file.write(pvl_one_group)
                pvl_arg_file.flush()

                pvl_home_file.write(pvl_three_group)
                pvl_home_file.flush()

                pvl_isisroot_file.write(pvl_two_group)
                pvl_isisroot_file.flush()

                pvl_obj = util.get_isis_preferences(pvl_arg_file.name)

                assert pvl_obj['Test']['t'] == 't1'
                assert pvl_obj['Data']['b'] == 'b3'

@pytest.mark.parametrize('string,expected,case_sensative', [('$bar/baz', '/bar/baz', False), ('$bar/$foo/baz', '/bar//foo/baz', True), ('$BAR/$FOO/baz', '/bar//foo/baz', False)])
def test_expand_vars(string, expected, case_sensative):
    user_vars = {'foo': '/foo', 'bar': '/bar'}
    result = util.expandvars(string, env_dict=user_vars, case_sensative=case_sensative)
    assert result == expected


