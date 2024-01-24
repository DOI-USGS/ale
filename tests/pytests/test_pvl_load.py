import pytest
import pvl
import ale

# I need to figure out how to get some kind of test cube in the test without
# referencing it locally
testCube = "/Users/ahibl/astro_efs/test_imgs/uvvis/LUA3107H.161.clem.cub_ISIS.cub"

@pytest.fixture
def test_loads():
    isis_kerns = ale.util.generate_kernels_from_cube(testCube, expand=True)
    pvl_obj = pvl.load(testCube)
    res = ale.loads(pvl_obj, props={"kernels": isis_kerns}, only_naif_spice=True)
    return res

def test_pass(test_loads):
    pass