from importlib import reload
import os
from os.path import join
from pathlib import Path

import pytest
import tempfile
import pvl
from unittest.mock import MagicMock, patch

from conftest import get_image_label, get_image_kernels, convert_kernels

from collections import OrderedDict

import ale
from ale import kernel_access
from ale.drivers.mro_drivers import MroCtxIsisLabelNaifSpiceDriver
from ale import spice_root

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

def test_get_kernels_from_metakernel():

    mro_test_mk = join(Path(__file__).parent.absolute(), 'data', 'kernel_access', 'mro_test_mk.tm')
    mro_test_path = join(Path(__file__).parent.absolute(), 'data', 'B10_013341_1010_XN_79S172W')

    kernels_from_mk = kernel_access.get_kernels_from_metakernel(mro_test_mk, mro_test_path)

    mro_test_kernels = ['B10_013341_1010_XN_79S172W_0.xsp',
                        'B10_013341_1010_XN_79S172W_1.xsp',
                        'mro_ctx_v11.ti',
                        'mro_sc_psp_090526_090601_0_sliced_-74000.xc',
                        'mro_sc_psp_090526_090601_1_sliced_-74000.xc',
                        'mro_sclkscet_00082_65536.tsc']
    
    for index, kernel in enumerate(mro_test_kernels):
        mro_test_kernels[index] = join(mro_test_path, kernel)

    assert kernels_from_mk == mro_test_kernels

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


def test_kernel_from_cube_list_spiceql(cube_kernels):
    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = kernel_access.generate_kernels_from_cube(cube.name, format_as="spiceql")
    print(kernels)
    expected_kernels = {'ck': ['messenger/instrumentpointing0', 'messenger/instrumentpointing1'], 'spk': ['messenger/instrumentposition'], 'pck': ['base/attitudeshape'], 'tspk': ['messenger/targetposition0', 'messenger/targetposition1'], 'fk': [], 'ik': ['messenger/instrument'], 'iak': [], 'sclk': ['base/clock'], 'lsk': [], 'extra': []} 
    assert kernels == expected_kernels


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
    expected_dict = OrderedDict([('TargetPosition', ['$messenger/targetposition0', '$messenger/targetposition1']), 
                                 ('InstrumentPosition', ['$messenger/instrumentposition']), 
                                 ('InstrumentPointing', ['$messenger/instrumentpointing0', '$messenger/instrumentpointing1']), 
                                 ('Frame', [None]), 
                                 ('TargetAttitudeShape', ['$base/attitudeshape']), 
                                 ('Instrument', ['$messenger/instrument']), 
                                 ('InstrumentAddendum', [None]), 
                                 ('LeapSecond', [None]), 
                                 ('SpacecraftClock', ['$base/clock']), 
                                 ('Extra', [None]), 
                                 ('Clock', [None]), 
                                 ('ShapeModel', [None])])
    assert kernels == expected_dict

def test_kernel_from_cube_dict_expanded(monkeypatch, tmpdir, pvl_four_group, cube_kernels):
    with patch.dict('os.environ', {'ISISROOT': str(tmpdir), 'ISIS3DATA': '$ISISDATA', 'ISISDATA': '/test/path'}):

        with open(tmpdir.join('IsisPreferences'), 'w+') as pvl_isisroot_file:
            pvl_isisroot_file.write(pvl_four_group)
            pvl_isisroot_file.flush()

        with tempfile.NamedTemporaryFile('r+') as cube:
            cube.write(cube_kernels)
            cube.flush()
            kernels = kernel_access.generate_kernels_from_cube(cube.name, expand=True, format_as='dict')
        print(kernels.keys())
        expected_dict = OrderedDict([('TargetPosition', ['/test/path/messenger/targetposition0', '/test/path/messenger/targetposition1']), 
                                     ('InstrumentPosition', ['/test/path/messenger/instrumentposition']), 
                                     ('InstrumentPointing', ['/test/path/messenger/instrumentpointing0', '/test/path/messenger/instrumentpointing1']), 
                                     ('Frame', [None]), 
                                     ('TargetAttitudeShape', ['/test/path/base/attitudeshape']), 
                                     ('Instrument', ['/test/path/messenger/instrument']), 
                                     ('InstrumentAddendum', [None]), 
                                     ('LeapSecond', [None]), 
                                     ('SpacecraftClock', ['/test/path/base/clock']), 
                                     ('Extra', [None]), 
                                     ('Clock', [None]), 
                                     ('ShapeModel', [None])])
        assert kernels == expected_dict

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

def test_get_metakernels_year_only_filename(tmpdir):
    """A metakernel filename with only mission_<year> (no version segment),
    e.g. 'lro_2013.tm', should parse as year='2013', version='N/A' so that
    filtering by year picks the right file. Previously the parser inserted
    'N/A' as the year and treated '2013' as the version, which caused
    versions='latest' to pick lro_2018.tm over lro_2013.tm regardless of
    cube date.
    """
    tmpdir.mkdir('lro-b-v01')
    open(tmpdir.join('lro-b-v01', 'lro_2013.tm'), 'w').close()
    open(tmpdir.join('lro-b-v01', 'lro_2018.tm'), 'w').close()

    res_2013 = kernel_access.get_metakernels(
        str(tmpdir), missions='lro', years=2013, versions='latest')
    assert res_2013['count'] == 1
    assert res_2013['data'][0]['path'].endswith('lro_2013.tm')
    assert res_2013['data'][0]['year'] == '2013'
    assert res_2013['data'][0]['version'] == 'N/A'

    res_2018 = kernel_access.get_metakernels(
        str(tmpdir), missions='lro', years=2018, versions='latest')
    assert res_2018['count'] == 1
    assert res_2018['data'][0]['path'].endswith('lro_2018.tm')

def test_get_metakernels_version_only_filename(tmpdir):
    """A metakernel filename with mission_<version> (no year segment), e.g.
    'ch2_v01.tm' or 'msl_v01.tm', should keep parsing as year='N/A',
    version='v01' so it matches any-year filter. The fix that made
    'lro_2013.tm' work must preserve this legacy behavior for files whose
    second segment isn't a 4-digit year.
    """
    tmpdir.mkdir('ch2-b-v01')
    open(tmpdir.join('ch2-b-v01', 'ch2_v01.tm'), 'w').close()
    tmpdir.mkdir('msl-b-v01')
    open(tmpdir.join('msl-b-v01', 'msl_v01.tm'), 'w').close()

    res_ch2 = kernel_access.get_metakernels(
        str(tmpdir), missions='ch2', years=2023, versions='latest')
    assert res_ch2['count'] == 1
    assert res_ch2['data'][0]['path'].endswith('ch2_v01.tm')
    assert res_ch2['data'][0]['year'] == 'N/A'
    assert res_ch2['data'][0]['version'] == 'v01'

    res_msl = kernel_access.get_metakernels(
        str(tmpdir), missions='msl', years=2014, versions='latest')
    assert res_msl['count'] == 1
    assert res_msl['data'][0]['path'].endswith('msl_v01.tm')
    assert res_msl['data'][0]['year'] == 'N/A'
    assert res_msl['data'][0]['version'] == 'v01'

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

def test_get_metakernels_all_isisdata_shapes(tmpdir):
    """Regression coverage for every metakernel name shape seen in a real ISISDATA.
    There is no formal metakernel naming standard, so this exercises each observed
    form with a real example. Pure filename stubs (no SPICE, no kernels). Confirms
    the year and version are read by pattern (not by fixed position), forecast and
    planning metakernels (predicted, plan, flip) are skipped, and the observation
    metakernel is selected.

    Shapes covered (one real example each):
      mission_year_version              mro_2005_v08, msgr_2004_v13
      mission_subset_year_version       orx_noola_2020_v06
      mission_year                      lro_2013
      mission_year_VERSION (upper V)    lro_2009_V04
      mission_version (no year)         msl_v01
      mission_body_provider_m<N>_v<N>   dawn_ceres_dlr_m135_v1  (Dawn, no year)
      SEMANTIC (no year, no version)    MEX_OPS, ROS_OPS, SMART1_OPS, em16_cassis
      SEMANTIC_Vver_date_build (5 seg)  MEX_OPS_V324_20250321_001, em16_cassis_v533_20250325_002
      MISSION_PREDICTED_Vver            CH1_PREDICTED_V00  (skipped)
      MISSION_Vver                      CH1_V00
      planning / flip                   em16_plan, em16_flip (skipped)
    """
    def mk(mission, *names):
        d = tmpdir.mkdir(mission).mkdir('kernels').mkdir('mk')
        for n in names:
            open(d.join(n), 'w').close()

    mk('mro',    'mro_2005_v01.tm', 'mro_2005_v08.tm')
    mk('msgr',   'msgr_2004_v08.tm', 'msgr_2004_v13.tm')
    mk('orx',    'orx_2016_v01.tm', 'orx_noola_2020_v01.tm', 'orx_noola_2020_v06.tm')
    mk('lro',    'lro_2013.tm', 'lro_2018.tm', 'lro_2009_V02.tm', 'lro_2009_V04.tm')
    mk('msl',    'msl_v01.tm')
    mk('dawn',   'dawn_ceres_dlr_m135_v1.tm', 'dawn_ceres_grv_m100_v1.tm', 'dawn_vesta_grv_m50_v1.tm')
    mk('mex',    'MEX_OPS.TM', 'MEX_OPS_V324_20250321_001.TM')
    mk('ros',    'ROS_OPS.TM', 'ROS_OPS_V350_20220906_001.TM')
    mk('smart1', 'SMART1_OPS.TM')
    mk('ch1',    'CH1_V00.TM', 'CH1_PREDICTED_V00.TM')
    mk('tgo',    'em16_cassis.tm', 'em16_ops.tm', 'em16_plan.tm', 'em16_flip.tm',
                 'em16_cassis_v533_20250325_002.tm', 'em16_plan_v533_20250318_001.tm')

    def pick(m, y):
        r = kernel_access.get_metakernels(str(tmpdir), missions=m, years=y, versions='latest')
        got = [os.path.basename(d['path']) for d in r['data']]
        assert r['count'] == 1, f"{m} {y}: expected one metakernel, got {got}"
        return got[0]

    assert pick('mro', 2005)    == 'mro_2005_v08.tm'          # latest version wins
    assert pick('msgr', 2004)   == 'msgr_2004_v13.tm'
    assert pick('orx', 2020)    == 'orx_noola_2020_v06.tm'    # 4-segment name
    assert pick('lro', 2013)    == 'lro_2013.tm'              # year-only
    assert pick('lro', 2009)    == 'lro_2009_V04.tm'          # uppercase version
    assert pick('msl', 2023)    == 'msl_v01.tm'               # version-only (year N/A)
    assert pick('dawn', 2015).startswith('dawn_')            # body/product name, no year
    assert pick('mex', 2005)    == 'MEX_OPS.TM'               # generic over dated snapshot
    assert pick('ros', 2005)    == 'ROS_OPS.TM'
    assert pick('smart1', 2005) == 'SMART1_OPS.TM'            # single semantic metakernel
    assert pick('ch1', 2009)    == 'CH1_V00.TM'               # predicted is skipped
    assert pick('tgo', 2018)    == 'em16_cassis.tm'           # planning and flip skipped

    # build-dated variants parse without corrupting the path field: the 8-digit
    # build date is read as the year, the v<N> segment as the version.
    dated = [m for m in kernel_access.get_metakernels(str(tmpdir), missions='tgo')['data']
             if m['path'].endswith('em16_cassis_v533_20250325_002.tm')][0]
    assert dated['year'] == '20250325'
    assert dated['version'] == 'v533'


def test_get_kernels_from_metakernel_relative_paths(tmpdir, monkeypatch):
    """A metakernel with relative PATH_VALUES (e.g. '..', as ESA ships) must
    resolve its kernels against the metakernel's own directory, not the current
    working directory. Regression: os.path.isfile('../ck/...') was checked
    relative to CWD and failed unless ALE ran from kernels/mk/.
    """
    kroot = tmpdir.mkdir('kernels')
    mkdir = kroot.mkdir('mk')
    ckdir = kroot.mkdir('ck')
    open(ckdir.join('stub.bc'), 'w').close()
    mk = mkdir.join('test.tm')
    mk.write("\\begindata\n"
             "    PATH_VALUES     = ( '..' )\n"
             "    PATH_SYMBOLS    = ( 'KERNELS' )\n"
             "    KERNELS_TO_LOAD = ( '$KERNELS/ck/stub.bc' )\n"
             "\\begintext\n")

    # run from a DIFFERENT directory to prove resolution is anchored to the mk dir
    monkeypatch.chdir(str(tmpdir.mkdir('elsewhere')))
    kernels = kernel_access.get_kernels_from_metakernel(str(mk))
    assert [os.path.realpath(k) for k in kernels] == [os.path.realpath(str(ckdir.join('stub.bc')))]
