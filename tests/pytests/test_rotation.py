import pytest

import numpy as np
from scipy.spatial.transform import Rotation
from ale.rotation import ConstantRotation, TimeDependentRotation

def test_constant_constant_composition():
    rot1_2 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 1, 2)
    rot2_3 = ConstantRotation([0, 1.0/np.sqrt(2), 0, 1.0/np.sqrt(2)], 2, 3)
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, ConstantRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    np.testing.assert_equal(rot1_3.quat, [0.5, 0.5, -0.5, 0.5])

def test_constant_time_dependent_composition():
    quats = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times = [0, 1]
    av = [[np.pi/2, 0, 0], [np.pi/2, 0, 0]]
    rot1_2 = TimeDependentRotation(quats, times, 1, 2, av=av)
    rot2_3 = ConstantRotation([0, 1.0/np.sqrt(2), 0, 1.0/np.sqrt(2)], 2, 3)
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_quats = [[0.5, 0.5, -0.5, 0.5],[1.0/np.sqrt(2), 0, -1.0/np.sqrt(2), 0]]
    np.testing.assert_equal(rot1_3.times, times)
    np.testing.assert_almost_equal(rot1_3.quats, expected_quats)
    np.testing.assert_almost_equal(rot1_3.av, av)

def test_time_dependent_constant_composition():
    rot1_2 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 1, 2)
    quats = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times = [0, 1]
    av = [[np.pi/2, 0, 0], [np.pi/2, 0, 0]]
    rot2_3 = TimeDependentRotation(quats, times, 2, 3, av=av)
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_quats = [[1, 0, 0, 0],[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)]]
    np.testing.assert_equal(rot1_3.times, times)
    np.testing.assert_almost_equal(rot1_3.quats, expected_quats)
    np.testing.assert_almost_equal(rot1_3.av, av)

def test_time_dependent_time_dependent_composition():
    # 90 degree rotation about the X-axis to a 180 degree rotation about the X-axis
    quats1_2 = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times1_2 = [0, 1]
    av1_2 = [[np.pi/2, 0, 0], [np.pi/2, 0, 0]]
    rot1_2 = TimeDependentRotation(quats1_2, times1_2, 1, 2, av=av1_2)
    # -90 degree rotation about the X-axis to a 90 degree rotation about the X-axis
    quats2_3 = [[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)],[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]]
    times2_3 = [0, 2]
    av2_3 = [[np.pi/2, 0, 0], [np.pi/2, 0, 0]]
    rot2_3 = TimeDependentRotation(quats2_3, times2_3, 2, 3, av=av2_3)

    # compose to get no rotation to a 180 degree rotation about the X-axis to no rotation
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_times = [0, 1, 2]
    expected_quats = [[0, 0, 0, -1], [1, 0, 0, 0], [0, 0, 0, -1]]
    expected_av = [[np.pi, 0, 0], [np.pi, 0, 0], [np.pi, 0, 0]]
    np.testing.assert_equal(rot1_3.times, expected_times)
    np.testing.assert_almost_equal(rot1_3.quats, expected_quats)
    np.testing.assert_almost_equal(rot1_3.av, expected_av)

def test_constant_inverse():
    rot1_2 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 1, 2)
    rot2_1 = rot1_2.inverse()
    assert rot2_1.source == 2
    assert rot2_1.dest == 1
    expected_quats = [1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)]
    np.testing.assert_almost_equal(rot2_1.quat, expected_quats)

def test_time_dependent_inverse():
    quats1_2 = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times1_2 = [0, 1]
    av1_2 = [[np.pi/2, 0, 0], [np.pi/2, 0, 0]]
    rot1_2 = TimeDependentRotation(quats1_2, times1_2, 1, 2, av=av1_2)
    rot2_1 = rot1_2.inverse()
    assert rot2_1.source == 2
    assert rot2_1.dest == 1
    expected_quats = [[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)],[1, 0, 0, 0]]
    expected_av = [[-np.pi/2, 0, 0], [-np.pi/2, 0, 0]]
    np.testing.assert_equal(rot2_1.times, times1_2)
    np.testing.assert_almost_equal(rot2_1.quats, expected_quats)
    np.testing.assert_almost_equal(rot2_1.av, expected_av)

def test_time_dependent_inverse_no_av():
    quats1_2 = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times1_2 = [0, 1]
    rot1_2 = TimeDependentRotation(quats1_2, times1_2, 1, 2)
    rot2_1 = rot1_2.inverse()
    assert rot2_1.source == 2
    assert rot2_1.dest == 1
    expected_quats = [[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)],[1, 0, 0, 0]]
    np.testing.assert_equal(rot2_1.times, times1_2)
    np.testing.assert_almost_equal(rot2_1.quats, expected_quats)
    assert rot2_1.av is None

def test_rotation_matrix():
    rot = ConstantRotation([0, 0, 0, 1], 1, 2)
    mat = rot.rotation_matrix()
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (3, 3)

def test_from_euler():
    angles = [[np.pi/2, np.pi/2, 0],
              [-np.pi/2, -np.pi/2, 0]]
    times = [0, 1]
    seq = 'XYZ'
    rot = TimeDependentRotation.from_euler(seq, angles, times, 0, 1)
    expected_quats = [[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]]
    np.testing.assert_almost_equal(rot.quats, expected_quats)
    assert rot.av is None
    np.testing.assert_equal(rot.times, times)
    assert rot.source == 0
    assert rot.dest == 1

def test_from_euler_degrees():
    rad_angles = [[np.pi/2, np.pi/2, 0],
                  [-np.pi/2, -np.pi/2, 0]]
    degree_angles = [[90, 90, 0],
                     [-90, -90, 0]]
    rad_rot = TimeDependentRotation.from_euler('XYZ', rad_angles, [0, 1], 0, 1)
    degree_rot = TimeDependentRotation.from_euler('XYZ', degree_angles, [0, 1], 0, 1, degrees=True)
    np.testing.assert_almost_equal(rad_rot.quats, degree_rot.quats)
    assert degree_rot.av is None

def test_from_matrix():
    mat = [[0, 0, 1],
           [1, 0 ,0],
           [0, 1, 0]]
    rot = ConstantRotation.from_matrix(mat, 0, 1)
    expected_quats = np.asarray([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_almost_equal(rot.quat, expected_quats)
    assert rot.source == 0
    assert rot.dest == 1

def test_slerp():
    test_quats = Rotation.from_euler('x', np.array([-135, -90, 0, 45, 90]), degrees=True).as_quat()
    rot = TimeDependentRotation(test_quats, [-0.5, 0, 1, 1.5, 2], 1, 2)
    new_rots, new_avs = rot._slerp(np.arange(-3, 5))
    expected_rot = Rotation.from_euler('x',
                                       [-360, -270, -180, -90, 0, 90, 180, 270],
                                       degrees=True)
    np.testing.assert_almost_equal(new_rots.as_quat(),
                                   expected_rot.as_quat())
    np.testing.assert_almost_equal(np.degrees(new_avs),
                                   np.repeat([[90, 0, 0]], 8, 0))

def test_slerp_constant_rotation():
    rot = TimeDependentRotation([[0, 0, 0, 1]], [0], 1, 2)
    new_rot, new_avs = rot._slerp([-1, 3])
    np.testing.assert_equal(new_rot.as_quat(),
                            [[0, 0, 0, 1], [0, 0, 0, 1]])
    np.testing.assert_equal(new_avs,
                            [[0, 0, 0], [0, 0, 0]])

def test_slerp_single_time():
    rot = TimeDependentRotation([[0, 0, 0, 1]], [0], 1, 2, av=[[np.pi/2, 0, 0]])
    new_rot, new_avs = rot._slerp([-1, 3])
    expected_quats = [[-1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]]
    expected_av = [[np.pi/2, 0, 0], [np.pi/2, 0, 0]]
    np.testing.assert_almost_equal(new_rot.as_quat(),
                                   expected_quats)
    np.testing.assert_equal(new_avs,
                            expected_av)

def test_slerp_variable_velocity():
    test_quats = Rotation.from_euler('xyz',
                                     [[0, 0, 0],
                                      [-90, 0, 0],
                                      [-90, 180, 0],
                                      [-90, 180, 90]],
                                     degrees=True).as_quat()
    rot = TimeDependentRotation(test_quats, [0, 1, 2, 3], 1, 2)
    new_rots, new_avs = rot._slerp([-0.5, 0.5, 1.5, 2.5, 3.5])
    expected_rot = Rotation.from_euler('xyz',
                                       [[45, 0, 0],
                                        [-45, 0, 0],
                                        [-90, 90, 0],
                                        [-90, 180, 45],
                                        [-90, 180, 135]],
                                       degrees=True)
    np.testing.assert_almost_equal(new_rots.as_quat(),
                                   expected_rot.as_quat())
    np.testing.assert_almost_equal(np.degrees(new_avs),
                                   [[-90, 0, 0],
                                    [-90, 0 ,0],
                                    [0, 180, 0],
                                    [0, 0, 90],
                                    [0, 0, 90]])

def test_reinterpolate():
    rot = TimeDependentRotation([[0, 0, 0, 1], [0, 0, 0, 1]], [0, 1], 1, 2)
    new_rot = rot.reinterpolate(np.arange(-3, 5))
    assert new_rot.source == rot.source
    assert new_rot.dest == rot.dest
    np.testing.assert_equal(new_rot.times, np.arange(-3, 5))

def test_apply_at_single_time():
    test_quats = Rotation.from_euler('x', np.array([-90, 0, 45]), degrees=True).as_quat()
    rot = TimeDependentRotation(test_quats, [0, 1, 1.5], 1, 2)
    input_vec = np.asarray([1, 2, 3])
    rot_vec = rot.apply_at(input_vec, 0)
    np.testing.assert_almost_equal(rot_vec, np.asarray([[1, 3, -2]]))

def test_apply_at_vector_time():
    test_quats = Rotation.from_euler('x', np.array([-90, 0, 45]), degrees=True).as_quat()
    rot = TimeDependentRotation(test_quats, [0, 1, 1.5], 1, 2)
    input_vec = np.asarray([[1, 2, 3], [1, 2, 3]])
    rot_vec = rot.apply_at(input_vec, [0, 2])
    np.testing.assert_almost_equal(rot_vec, np.asarray([[1, 3, -2], [1, -3, 2]]))

def test_rotate_velocity_at():
    test_quats = Rotation.from_euler('xyz',
                                     [[0, 0, 0],
                                      [-90, 0, 0],
                                      [-90, 180, 0],
                                      [-90, 180, 90]],
                                     degrees=True).as_quat()
    rot = TimeDependentRotation(test_quats, [0, 1, 2, 3], 1, 2)
    input_pos = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    input_vel = [[-1, -2, -3], [-1, -2, -3], [-1, -2, -3]]
    input_times = [1, 2, 3]
    rot_vel = rot.rotate_velocity_at(input_pos, input_vel, input_times)
    np.testing.assert_almost_equal(rot_vel,
                                   [[-1, -3 + np.pi, 2 + 3*np.pi/2],
                                    [1 + 3*np.pi, -3 + np.pi, -2],
                                    [3, 1 - np.pi, -2 - np.pi/2]])
