import os
from os.path import join
from importlib import reload
import subprocess
import networkx as nx

import pytest
import tempfile
import spiceypy as spice
import pvl
from unittest import mock
from unittest.mock import MagicMock, patch

from collections import OrderedDict

import ale
from ale import util

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

@pytest.fixture
def pvl_four_group():
    # Mock of the DataDirectory group
    return """
    Group = DataDirectory
      Base         = $ISIS3DATA/base
      Messenger    = $ISIS3DATA/messenger
    EndGroup
    """

def test_pvl_parser(pvl_three_group):
    obj = util.JBFPvlParser(pvl_three_group)
    assert obj["Data"]["a"] == "a3"
    assert obj["Test"]["t"] == "t3"
    assert obj["Settings"]["delsystem32"] == "yes"


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
    kernels = util.find_kernels(str(cube_file), str(tmpdir))
    assert kernels == {'Pointing': {'kernels': [str(tmpdir / 'MRO/fake')], 'types': ['Reconstructed']}, 'instrument': {'kernels': [str(tmpdir / 'fake/not/a/real/file')]}}


def test_kernel_from_cube_list(cube_kernels):
    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = util.generate_kernels_from_cube(cube.name)
    assert kernels == ['$messenger/targetposition0', '$messenger/targetposition1','$messenger/instrumentposition', '$messenger/instrumentpointing0', '$messenger/instrumentpointing1', '$base/attitudeshape', '$messenger/instrument', '$base/clock']

def test_kernel_from_cube_list_expanded(monkeypatch, tmpdir, pvl_four_group, cube_kernels):
    monkeypatch.setenv('ISISROOT', str(tmpdir))
    monkeypatch.setenv('ISIS3DATA', '/test/path')

    with open(tmpdir.join('IsisPreferences'), 'w+') as pvl_isisroot_file:
        pvl_isisroot_file.write(pvl_four_group)
        pvl_isisroot_file.flush()

    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = util.generate_kernels_from_cube(cube.name, expand=True)
    assert kernels == ['/test/path/messenger/targetposition0', '/test/path/messenger/targetposition1', '/test/path/messenger/instrumentposition', '/test/path/messenger/instrumentpointing0', '/test/path/messenger/instrumentpointing1', '/test/path/base/attitudeshape', '/test/path/messenger/instrument', '/test/path/base/clock']

def test_kernel_from_cube_dict(cube_kernels):
    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = util.generate_kernels_from_cube(cube.name, format_as='dict')
    assert kernels == OrderedDict([('TargetPosition', ['$messenger/targetposition0', '$messenger/targetposition1']), ('InstrumentPosition', ['$messenger/instrumentposition']), ('InstrumentPointing', ['$messenger/instrumentpointing0', '$messenger/instrumentpointing1']), ('Frame', [None]), ('TargetAttitudeShape', ['$base/attitudeshape']), ('Instrument', ['$messenger/instrument']), ('InstrumentAddendum', [None]), ('LeapSecond', [None]), ('SpacecraftClock', ['$base/clock']), ('Extra', [None]), ('Clock', [None])])

def test_kernel_from_cube_dict_expanded(monkeypatch, tmpdir, pvl_four_group, cube_kernels):
    monkeypatch.setenv('ISISROOT', str(tmpdir))
    monkeypatch.setenv('ISIS3DATA', '/test/path')

    with open(tmpdir.join('IsisPreferences'), 'w+') as pvl_isisroot_file:
        pvl_isisroot_file.write(pvl_four_group)
        pvl_isisroot_file.flush()

    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = util.generate_kernels_from_cube(cube.name, expand=True, format_as='dict')
    assert kernels == OrderedDict([('TargetPosition', ['/test/path/messenger/targetposition0', '/test/path/messenger/targetposition1']), ('InstrumentPosition', ['/test/path/messenger/instrumentposition']), ('InstrumentPointing', ['/test/path/messenger/instrumentpointing0', '/test/path/messenger/instrumentpointing1']), ('Frame', [None]), ('TargetAttitudeShape', ['/test/path/base/attitudeshape']), ('Instrument', ['/test/path/messenger/instrument']), ('InstrumentAddendum', [None]), ('LeapSecond', [None]), ('SpacecraftClock', ['/test/path/base/clock']), ('Extra', [None]), ('Clock', [None])])

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


@pytest.mark.parametrize('filename', [os.path.join('.Isis', 'IsisPreferences'), 'IsisPreferences'])
def test_get_prefrences_malformed_files(monkeypatch, tmpdir, filename):
    monkeypatch.setenv('ISISROOT', str(tmpdir))
    monkeypatch.setenv('HOME', str(tmpdir))
    tmpdir.mkdir('.Isis')

    with pytest.raises(pvl.parser.LexerError):
        with open(tmpdir.join(filename), 'w+') as brokenpref:
            brokenpref.write('Totally not PVL')
            brokenpref.flush()
            util.get_isis_preferences()

    with pytest.raises(pvl.parser.LexerError):
        util.get_isis_preferences(tmpdir.join(filename))


@pytest.mark.parametrize('string,expected,case_sensitive', [('$bar/baz', '/bar/baz', False), ('$bar/$foo/baz', '/bar//foo/baz', True), ('$BAR/$FOO/baz', '/bar//foo/baz', False)])
def test_expand_vars(string, expected, case_sensitive):
    user_vars = {'foo': '/foo', 'bar': '/bar'}
    result = util.expandvars(string, env_dict=user_vars, case_sensitive=case_sensitive)
    assert result == expected


@pytest.mark.parametrize('search_kwargs,expected',
    [({'years':'2009', 'versions':'v01'}, {'count':1, 'data':[{'path':join('foo-b-v01', 'foo_2009_v01.tm'), 'year':'2009', 'mission':'foo', 'version':'v01'}]}),
     ({'versions':'v02', 'years':2010}, {'count': 1,  'data': [{'path':join('bar-b-v01', 'bar_2010_v02.tm'), 'year':'2010', 'mission':'bar', 'version': 'v02'}]})])
def test_get_metakernels(tmpdir, search_kwargs, expected):
    tmpdir.mkdir('foo-b-v01')
    tmpdir.mkdir('bar-b-v01')

    open(tmpdir.join('foo-b-v01', 'foo_2009_v01.tm'), 'w').close()
    open(tmpdir.join('bar-b-v01', 'bar_2010_v02.tm'), 'w').close()

    search_result =  util.get_metakernels(str(tmpdir), **search_kwargs)
    # we can't know the tmpdir at parameterization, append it here
    for r in expected['data']:
        r['path'] = str(tmpdir.join(r['path']))

    assert search_result == expected

@pytest.mark.parametrize('search_kwargs, expected',
    [({'years':'2009', 'versions':'v01'}, {'count':0, 'data':[]})])
def test_get_metakernels_no_alespiceroot(monkeypatch, search_kwargs, expected):
    monkeypatch.delenv('ALESPICEROOT', raising=False)
    reload(ale)
    with pytest.warns(UserWarning, match="Unable to search mission directories without" +
                                         "ALESPICEROOT being set. Defaulting to empty list"):
        search_result =  ale.util.get_metakernels(**search_kwargs)
    print(search_result)
    monkeypatch.setenv('ALESPICEROOT', '/foo/bar')
    reload(ale)

    assert search_result == expected

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

    search_result =  util.get_metakernels(str(tmpdir), **search_kwargs)

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


    search_result =  util.get_metakernels(str(tmpdir), **search_kwargs)
    assert search_result['count'] == expected_count

def test_create_spk_dependency_tree():
    de430_output ="""
BRIEF -- Version 4.0.0, September 8, 2010 -- Toolkit Version N0066


Summary for: de430.bsp

Bodies: MERCURY BARYCENTER (1) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        VENUS BARYCENTER (2) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        EARTH BARYCENTER (3) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        MARS BARYCENTER (4) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        JUPITER BARYCENTER (5) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        SATURN BARYCENTER (6) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        URANUS BARYCENTER (7) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        NEPTUNE BARYCENTER (8) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        PLUTO BARYCENTER (9) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        SUN (10) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        MERCURY (199) w.r.t. MERCURY BARYCENTER (1)
        VENUS (299) w.r.t. VENUS BARYCENTER (2)
        MOON (301) w.r.t. EARTH BARYCENTER (3)
        EARTH (399) w.r.t. EARTH BARYCENTER (3)
        Start of Interval (ET)              End of Interval (ET)
        -----------------------------       -----------------------------
        1549 DEC 31 00:00:00.000            2650 JAN 25 00:00:00.000
"""
    msgr_20040803_20150430_od431sc_2_output ="""
BRIEF -- Version 4.0.0, September 8, 2010 -- Toolkit Version N0066


Summary for: msgr_20040803_20150430_od431sc_2.bsp

Body: MESSENGER (-236) w.r.t. MERCURY BARYCENTER (1)
      Start of Interval (ET)              End of Interval (ET)
      -----------------------------       -----------------------------
      2008 JAN 13 19:18:06.919            2008 JAN 15 18:52:08.364
      2008 OCT 05 06:12:17.599            2008 OCT 07 11:08:05.670
      2009 SEP 28 05:25:44.180            2009 OCT 01 14:07:55.870
      2011 MAR 15 11:20:36.100            2015 APR 30 19:27:09.351

Body: MESSENGER (-236) w.r.t. VENUS BARYCENTER (2)
      Start of Interval (ET)              End of Interval (ET)
      -----------------------------       -----------------------------
      2006 OCT 16 19:26:46.293            2006 OCT 31 22:15:29.222
      2007 MAY 29 09:28:46.998            2007 JUN 13 15:30:49.997

Body: MESSENGER (-236) w.r.t. SUN (10)
      Start of Interval (ET)              End of Interval (ET)
      -----------------------------       -----------------------------
      2004 AUG 10 04:06:05.612            2005 JUL 26 22:39:53.204
      2005 AUG 09 16:04:35.204            2006 OCT 16 19:26:46.293
      2006 OCT 31 22:15:29.222            2007 MAY 29 09:28:46.998
      2007 JUN 13 15:30:49.997            2008 JAN 13 19:18:06.919
      2008 JAN 15 18:52:08.364            2008 OCT 05 06:12:17.599
      2008 OCT 07 11:08:05.670            2009 SEP 28 05:25:44.180
      2009 OCT 01 14:07:55.870            2011 MAR 15 11:20:36.100

Body: MESSENGER (-236) w.r.t. EARTH (399)
      Start of Interval (ET)              End of Interval (ET)
      -----------------------------       -----------------------------
      2004 AUG 03 07:14:39.393            2004 AUG 10 04:06:05.612
      2005 JUL 26 22:39:53.204            2005 AUG 09 16:04:35.204

Bodies: MERCURY BARYCENTER (1) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        EARTH BARYCENTER (3) w.r.t. SOLAR SYSTEM BARYCENTER (0)
        Start of Interval (ET)              End of Interval (ET)
        -----------------------------       -----------------------------
        2004 AUG 03 07:14:39.393            2015 APR 30 19:27:09.351

"""
    de430_mock = MagicMock(spec=subprocess.CompletedProcess)
    de430_mock.stdout = de430_output
    msgr_20040803_20150430_od431sc_2_mock = MagicMock(spec=subprocess.CompletedProcess)
    msgr_20040803_20150430_od431sc_2_mock.stdout = msgr_20040803_20150430_od431sc_2_output

    expected_tree = nx.DiGraph()
    expected_tree.add_edge(1, 0, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    expected_tree.add_edge(2, 0, kernel='de430.bsp')
    expected_tree.add_edge(3, 0, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    expected_tree.add_edge(4, 0, kernel='de430.bsp')
    expected_tree.add_edge(5, 0, kernel='de430.bsp')
    expected_tree.add_edge(6, 0, kernel='de430.bsp')
    expected_tree.add_edge(7, 0, kernel='de430.bsp')
    expected_tree.add_edge(8, 0, kernel='de430.bsp')
    expected_tree.add_edge(9, 0, kernel='de430.bsp')
    expected_tree.add_edge(10, 0, kernel='de430.bsp')
    expected_tree.add_edge(199, 1, kernel='de430.bsp')
    expected_tree.add_edge(299, 2, kernel='de430.bsp')
    expected_tree.add_edge(301, 3, kernel='de430.bsp')
    expected_tree.add_edge(399, 3, kernel='de430.bsp')
    expected_tree.add_edge(-236, 1, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    expected_tree.add_edge(-236, 2, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    expected_tree.add_edge(-236, 10, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    expected_tree.add_edge(-236, 399, kernel='msgr_20040803_20150430_od431sc_2.bsp')

    with patch('subprocess.run', side_effect = [de430_mock, msgr_20040803_20150430_od431sc_2_mock]) as run_mock:
        dep_tree = util.create_spk_dependency_tree(['de430.bsp', 'msgr_sc_EN1072174528M.bsp'])
        run_mock.assert_any_call(["brief", "-c de430.bsp"],
                                 capture_output=True,
                                 check=True,
                                 text=True)
        run_mock.assert_any_call(["brief", "-c msgr_sc_EN1072174528M.bsp"],
                                 capture_output=True,
                                 check=True,
                                 text=True)
    assert nx.is_isomorphic(dep_tree, expected_tree)

def test_spkmerge_config_string():
    dep_tree = nx.DiGraph()
    dep_tree.add_edge(1, 0, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    dep_tree.add_edge(2, 0, kernel='de430.bsp')
    dep_tree.add_edge(3, 0, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    dep_tree.add_edge(4, 0, kernel='de430.bsp')
    dep_tree.add_edge(5, 0, kernel='de430.bsp')
    dep_tree.add_edge(6, 0, kernel='de430.bsp')
    dep_tree.add_edge(7, 0, kernel='de430.bsp')
    dep_tree.add_edge(8, 0, kernel='de430.bsp')
    dep_tree.add_edge(9, 0, kernel='de430.bsp')
    dep_tree.add_edge(10, 0, kernel='de430.bsp')
    dep_tree.add_edge(199, 1, kernel='de430.bsp')
    dep_tree.add_edge(299, 2, kernel='de430.bsp')
    dep_tree.add_edge(301, 3, kernel='de430.bsp')
    dep_tree.add_edge(399, 3, kernel='de430.bsp')
    dep_tree.add_edge(-236, 1, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    dep_tree.add_edge(-236, 2, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    dep_tree.add_edge(-236, 10, kernel='msgr_20040803_20150430_od431sc_2.bsp')
    dep_tree.add_edge(-236, 399, kernel='msgr_20040803_20150430_od431sc_2.bsp')

    config_string = util.spkmerge_config_string(dep_tree,
                                                'theoutput.bsp',
                                                [-236, 199, 10],
                                                'thelsk.tls',
                                                'the start UTC string',
                                                'the stop UTC string')
    assert config_string == """LEAPSECONDS_KERNEL     = thelsk.tls
SPK_KERNEL             = theoutput.bsp
   BODIES              = 0, 1, 199, 10, -236
   BEGIN_TIME          = the start UTC string
   END_TIME            = the stop UTC string
   SOURCE_SPK_KERNEL   = de430.bsp
      INCLUDE_COMMENTS = no
   SOURCE_SPK_KERNEL   = msgr_20040803_20150430_od431sc_2.bsp
      INCLUDE_COMMENTS = no
""" or config_string == """LEAPSECONDS_KERNEL     = thelsk.tls
SPK_KERNEL             = theoutput.bsp
   BODIES              = 0, 1, 199, 10, -236
   BEGIN_TIME          = the start UTC string
   END_TIME            = the stop UTC string
   SOURCE_SPK_KERNEL   = msgr_20040803_20150430_od431sc_2.bsp
      INCLUDE_COMMENTS = no
   SOURCE_SPK_KERNEL   = de430.bsp
      INCLUDE_COMMENTS = no
"""

def test_get_ck_frames():
    msgr_0905_v02_output = """
CKBRIEF -- Version 6.1.0, June 27, 2014 -- Toolkit Version N0066


Summary for: msgr_0905_v02.bc

Objects  Interval Begin ET        Interval End ET          AV
-------- ------------------------ ------------------------ ---
-236001  NEED LSK AND SCLK FILES  NEED LSK AND SCLK FILES  Y
-236002  NEED LSK AND SCLK FILES  NEED LSK AND SCLK FILES  Y
-236002  NEED LSK AND SCLK FILES  NEED LSK AND SCLK FILES  Y
-236000  NEED LSK AND SCLK FILES  NEED LSK AND SCLK FILES  Y
-236002  NEED LSK AND SCLK FILES  NEED LSK AND SCLK FILES  Y

"""
    msgr_0905_v02_mock = MagicMock(spec=subprocess.CompletedProcess)
    msgr_0905_v02_mock.stdout = msgr_0905_v02_output
    with patch('subprocess.run', side_effect = [msgr_0905_v02_mock]) as run_mock:
        frames = util.get_ck_frames('msgr_0905_v02.bc')
        run_mock.assert_any_call(["ckbrief", "-t msgr_0905_v02.bc"],
                                 capture_output=True,
                                 check=True,
                                 text=True)
    assert frames == [-236002, -236001, -236000]
