from importlib import reload
from os.path import join

import pytest
import tempfile
import pvl
from unittest.mock import MagicMock, patch

from collections import OrderedDict

import ale
from ale import kernel_access

@pytest.fixture
def cube_kernels():
   return """
    Object = IsisCube
    Group = Instrument
      StartTime = 2016-332T05:40:45.020
      StopTime  = 2016-332T05:40:46.820
      InstrumentId = fake
      SpacecraftName = fake
    End_Group

    Group = Kernels
      TargetAttitudeShape = $base/attitudeshape
      TargetPosition = ($messenger/targetposition0, $messenger/targetposition1)
      Instrument = $messenger/instrument
      InstrumentPointing = (Table, $messenger/instrumentpointing0, $messenger/instrumentpointing1)
      SpacecraftClock = $base/clock
      InstrumentPosition = $messenger/instrumentposition
      InstrumentAddendum = Null
      ShapeModel = Null
    End_Group
    End_Object
    End
    """

@pytest.fixture
def pvl_four_group():
    # Mock of the DataDirectory group
    return """
    Group = DataDirectory
      Base         = $ISIS3DATA/base
      Messenger    = $ISIS3DATA/messenger
    EndGroup
    """

def test_find_kernels(cube_kernels, tmpdir):
    ck_db = """
    Object = Pointing
    Group = Selection
        Time = ( "2016 JAN 01 00:00:00.000000 TDB", "2016 DEC 31 00:00:00.000000 TDB" )
        Type = Reconstructed
        File = $MRO/fake
    End_Group
    End_Object
    """

    ik_db = """
    Object = instrument
    Group = Selection
        Match = ("Instrument", "InstrumentId", "fake")
        File = ("fake", "not/a/real/file")
    End_Group
    End_Object
    """
    translation = """
    Group = MissionName
      InputKey      = SpacecraftName
      InputGroup    = "IsisCube,Instrument"
      InputPosition = (IsisCube, Instrument)
      Translation   = (fake, "fake")
    End_Group
    """

    tmpdir.mkdir("fake").mkdir("kernels").mkdir("ik")
    tmpdir.mkdir("base").mkdir("kernels").mkdir("ck")
    tmpdir.mkdir("base", "translations")

    ck_db_file = tmpdir.join("base", "kernels", "ck", "kernel.01.db")
    ik_db_file = tmpdir.join("fake", "kernels", "ik", "kernel.01.db")
    translation_file = tmpdir.join("base", "translations", "MissionName2DataDir.trn")
    cube_file = tmpdir.join("test.cub")

    with open(translation_file, "w") as f:
        f.write(translation)

    with open(ck_db_file, "w") as f:
        f.write(ck_db)

    with open(ik_db_file, "w") as f:
        f.write(ik_db)

    with open(cube_file, "w") as cube:
        cube.write(cube_kernels)

    print(pvl.load(str(cube_file)))
    kernels = kernel_access.find_kernels(str(cube_file), str(tmpdir))
    assert kernels == {'Pointing': {'kernels': [str(tmpdir / 'MRO/fake')], 'types': ['Reconstructed']}, 'instrument': {'kernels': [str(tmpdir / 'fake/not/a/real/file')]}}

def test_kernel_from_cube_list(cube_kernels):
    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = kernel_access.generate_kernels_from_cube(cube.name)
    assert kernels == ['$messenger/targetposition0', '$messenger/targetposition1','$messenger/instrumentposition', '$messenger/instrumentpointing0', '$messenger/instrumentpointing1', '$base/attitudeshape', '$messenger/instrument', '$base/clock']

def test_kernel_from_cube_list_expanded(monkeypatch, tmpdir, pvl_four_group, cube_kernels):
    with patch.dict('os.environ', {'ISISROOT': str(tmpdir), 'ISIS3DATA': '$ISISDATA', 'ISISDATA': '/test/path'}):

        with open(tmpdir.join('IsisPreferences'), 'w+') as pvl_isisroot_file:
            pvl_isisroot_file.write(pvl_four_group)
            pvl_isisroot_file.flush()

        with tempfile.NamedTemporaryFile('r+') as cube:
            cube.write(cube_kernels)
            cube.flush()
            kernels = kernel_access.generate_kernels_from_cube(cube.name, expand=True)
        assert kernels == ['/test/path/messenger/targetposition0', '/test/path/messenger/targetposition1', '/test/path/messenger/instrumentposition', '/test/path/messenger/instrumentpointing0', '/test/path/messenger/instrumentpointing1', '/test/path/base/attitudeshape', '/test/path/messenger/instrument', '/test/path/base/clock']

def test_kernel_from_cube_dict(cube_kernels):
    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = kernel_access.generate_kernels_from_cube(cube.name, format_as='dict')
    assert kernels == OrderedDict([('TargetPosition', ['$messenger/targetposition0', '$messenger/targetposition1']), ('InstrumentPosition', ['$messenger/instrumentposition']), ('InstrumentPointing', ['$messenger/instrumentpointing0', '$messenger/instrumentpointing1']), ('Frame', [None]), ('TargetAttitudeShape', ['$base/attitudeshape']), ('Instrument', ['$messenger/instrument']), ('InstrumentAddendum', [None]), ('LeapSecond', [None]), ('SpacecraftClock', ['$base/clock']), ('Extra', [None]), ('Clock', [None])])

def test_kernel_from_cube_dict_expanded(monkeypatch, tmpdir, pvl_four_group, cube_kernels):
    with patch.dict('os.environ', {'ISISROOT': str(tmpdir), 'ISIS3DATA': '$ISISDATA', 'ISISDATA': '/test/path'}):

        with open(tmpdir.join('IsisPreferences'), 'w+') as pvl_isisroot_file:
            pvl_isisroot_file.write(pvl_four_group)
            pvl_isisroot_file.flush()

        with tempfile.NamedTemporaryFile('r+') as cube:
            cube.write(cube_kernels)
            cube.flush()
            kernels = kernel_access.generate_kernels_from_cube(cube.name, expand=True, format_as='dict')
        assert kernels == OrderedDict([('TargetPosition', ['/test/path/messenger/targetposition0', '/test/path/messenger/targetposition1']), ('InstrumentPosition', ['/test/path/messenger/instrumentposition']), ('InstrumentPointing', ['/test/path/messenger/instrumentpointing0', '/test/path/messenger/instrumentpointing1']), ('Frame', [None]), ('TargetAttitudeShape', ['/test/path/base/attitudeshape']), ('Instrument', ['/test/path/messenger/instrument']), ('InstrumentAddendum', [None]), ('LeapSecond', [None]), ('SpacecraftClock', ['/test/path/base/clock']), ('Extra', [None]), ('Clock', [None])])

def test_kernel_from_cube_no_kernel_group():
    with pytest.raises(KeyError):
       with tempfile.NamedTemporaryFile('w+') as cube:
           cube.write('')
           cube.flush()
           kernel_access.generate_kernels_from_cube(cube.name)

@pytest.mark.parametrize('search_kwargs,expected',
    [({'years':'2009', 'versions':'v01'}, {'count':1, 'data':[{'path':join('foo-b-v01', 'foo_2009_v01.tm'), 'year':'2009', 'mission':'foo', 'version':'v01'}]}),
     ({'versions':'v02', 'years':2010}, {'count': 1,  'data': [{'path':join('bar-b-v01', 'bar_2010_v02.tm'), 'year':'2010', 'mission':'bar', 'version': 'v02'}]})])
def test_get_metakernels(tmpdir, search_kwargs, expected):
    tmpdir.mkdir('foo-b-v01')
    tmpdir.mkdir('bar-b-v01')

    open(tmpdir.join('foo-b-v01', 'foo_2009_v01.tm'), 'w').close()
    open(tmpdir.join('bar-b-v01', 'bar_2010_v02.tm'), 'w').close()

    search_result =  kernel_access.get_metakernels(str(tmpdir), **search_kwargs)
    # we can't know the tmpdir at parameterization, append it here
    for r in expected['data']:
        r['path'] = str(tmpdir.join(r['path']))

    assert search_result == expected

@pytest.mark.parametrize('search_kwargs, expected',
    [({'years':'2009', 'versions':'v01'}, {'count':0, 'data':[]})])
def test_get_metakernels_no_alespiceroot(monkeypatch, search_kwargs, expected):
    with pytest.warns(UserWarning, match="Unable to search mission directories without" +
                                        "ALESPICEROOT being set. Defaulting to empty list"):
        search_result =  ale.kernel_access.get_metakernels(**search_kwargs)
    print(search_result)
    with patch.dict('os.environ', {'ALESPICEROOT': '/foo/bar'}):
        reload(ale)

        assert search_result == expected
    reload(ale)
    assert not ale.spice_root

@pytest.mark.parametrize('search_kwargs', [{'years':'2010'}, {'years':2010}, {'years': [2010]}, {'years': ['2010']}, {'years': set(['2010', '1999', '1776'])},
    {'missions':'bar', 'versions':'v20'}, {'missions': ['bar'], 'versions':'v20'}, {'missions': 'bar', 'versions':['v20', 'v03']}, {'missions':set(['bar']),'years': 2010, 'versions': 'latest'} ])
def test_get_metakernels_search_args(tmpdir, search_kwargs):
    tmpdir.mkdir('foo-b-v01')
    tmpdir.mkdir('bar-b-v01')

    open(tmpdir.join('foo-b-v01', 'foo_2009_v01.tm'), 'w').close()
    open(tmpdir.join('bar-b-v01', 'bar_9009_v01.tm'), 'w').close()
    open(tmpdir.join('bar-b-v01', 'bar_2009_v10.tm'), 'w').close()

    test_mk = tmpdir.join('bar-b-v01', 'bar_2010_v20.tm')
    open(test_mk, 'w').close()

    search_result =  kernel_access.get_metakernels(str(tmpdir), **search_kwargs)

    expected = {
        'count' : 1,
        'data' : [{
                'year' : '2010',
                'mission' : 'bar',
                'version': 'v20',
                'path': test_mk
            }]
    }

    assert search_result == expected

@pytest.mark.parametrize('search_kwargs,expected_count', [({'years':'2010'}, 2), ({'years': ['1990', '2009']}, 4), ({'years':'9009'}, 1), ({'years':'all'}, 7), ({'years':[]},7),  ({'versions':'latest'}, 6), ({'versions':'all'},7), ({'versions':[]}, 7), ({'versions':None}, 7),  ({'versions':['v20']}, 2), ({'versions':['v10', 'v01']}, 4), ({'missions': 'foo'}, 3), ({'missions':'bar'},3), ({'missions':'baz'},1), ({'missions':'all'}, 7), ({'missions':['foo', 'bar'], 'versions': 'v01', 'years':
    2009}, 1), ({}, 7), ({'versions': 'latest', 'missions':'foo'}, 2), ({'missions': 'not_real'}, 0)])
def test_get_metakernels_search_counts(tmpdir, search_kwargs, expected_count):
    tmpdir.mkdir('foo-b-v01')
    tmpdir.mkdir('bar-b-v01')
    tmpdir.mkdir('baz-b-v100')

    open(tmpdir.join('foo-b-v01', 'foo_2009_v01.tm'), 'w').close()
    open(tmpdir.join('foo-b-v01', 'foo_2009_v20.tm'), 'w').close()
    open(tmpdir.join('foo-b-v01', 'foo_2010_v20.tm'), 'w').close()
    open(tmpdir.join('bar-b-v01', 'bar_9009_v01.tm'), 'w').close()
    open(tmpdir.join('bar-b-v01', 'bar_2009_v10.tm'), 'w').close()
    open(tmpdir.join('bar-b-v01', 'bar_2010_v02.tm'), 'w').close()
    open(tmpdir.join('baz-b-v100', 'baz_1990_v10.tm'), 'w').close()


    search_result =  kernel_access.get_metakernels(str(tmpdir), **search_kwargs)
    assert search_result['count'] == expected_count