import pytest
import tempfile
import spiceypy as spice

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

def test_kernel_from_cube_order(cube_kernels):
    with tempfile.NamedTemporaryFile('r+') as cube:
        cube.write(cube_kernels)
        cube.flush()
        kernels = util.generate_kernels_from_cube(cube.name)
    assert kernels == ['targetposition0', 'targetposition1','instrumentposition', 'instrumentpointing0', 'instrumentpointing1', 'attitudeshape', 'instrument', 'clock']

def test_kernel_from_cube_no_kernel_group():
   with pytest.raises(KeyError):
       with tempfile.NamedTemporaryFile('r+') as cube:
           cube.write('')
           cube.flush()
           util.generate_kernels_from_cube(cube.name)

