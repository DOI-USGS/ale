import pytest

import numpy as np
from scipy.spatial.transform import Rotation
from ale.rotation import ConstantRotation, TimeDependentRotation

def test_constant_constant_composition():
    # Two 90 degree rotation about the X-axis
    rot1_2 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 1, 2)
    rot2_3 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 2, 3)
    # compose to get a 180 degree rotation about the X-axis
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, ConstantRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    np.testing.assert_equal(rot1_3.quat, np.array([1, 0, 0, 0]))

def test_constant_time_dependent_composition():
    # 90 degree rotation about the X-axis to a 180 degree rotation about the X-axis
    quats = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times = [0, 1]
    rot1_2 = TimeDependentRotation(quats, times, 1, 2)
    # 90 degree rotation about the X-axis
    rot2_3 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 2, 3)
    # compose to get a 180 degree rotation about the X-axis to a 270 degree rotation about the X-axis
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_quats = np.array([[1, 0, 0, 0],[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)]])
    np.testing.assert_equal(rot1_3.times, np.array(times))
    np.testing.assert_almost_equal(rot1_3.quats, expected_quats)

def test_time_dependent_constant_composition():
    # 90 degree rotation about the X-axis
    rot1_2 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 1, 2)
    # 90 degree rotation about the X-axis to a 180 degree rotation about the X-axis
    quats = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times = [0, 1]
    rot2_3 = TimeDependentRotation(quats, times, 2, 3)
    # compose to get a 180 degree rotation about the X-axis to a 270 degree rotation about the X-axis
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_quats = np.array([[1, 0, 0, 0],[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)]])
    np.testing.assert_equal(rot1_3.times, np.array(times))
    np.testing.assert_almost_equal(rot1_3.quats, expected_quats)

def test_time_dependent_time_dependent_composition():
    # 90 degree rotation about the X-axis to a 180 degree rotation about the X-axis
    quats1_2 = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times1_2 = [0, 1]
    rot1_2 = TimeDependentRotation(quats1_2, times1_2, 1, 2)
    # -90 degree rotation about the X-axis to a 90 degree rotation about the X-axis
    quats2_3 = [[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)],[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]]
    times2_3 = [0, 2]
    rot2_3 = TimeDependentRotation(quats2_3, times2_3, 2, 3)
    # compose to get no rotation to a 180 degree rotation about the X-axis to no rotation
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_times = np.array([0, 1])
    expected_quats = np.array([[0, 0, 0, -1],[-1, 0, 0, 0]])
    np.testing.assert_equal(rot1_3.times, expected_times)
    np.testing.assert_almost_equal(rot1_3.quats, expected_quats)

def test_constant_inverse():
    rot1_2 = ConstantRotation([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)], 1, 2)
    rot2_1 = rot1_2.inverse()
    assert rot2_1.source == 2
    assert rot2_1.dest == 1
    expected_quats = np.array([1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)])
    np.testing.assert_almost_equal(rot2_1.quat, expected_quats)

def test_time_dependent_inverse():
    quats1_2 = [[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]]
    times1_2 = [0, 1]
    rot1_2 = TimeDependentRotation(quats1_2, times1_2, 1, 2)
    rot2_1 = rot1_2.inverse()
    assert rot2_1.source == 2
    assert rot2_1.dest == 1
    expected_quats = np.array([[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)],[1, 0, 0, 0]])
    np.testing.assert_equal(rot2_1.times, np.array(times1_2))
    np.testing.assert_almost_equal(rot2_1.quats, expected_quats)

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
    expected_quats = np.asarray([[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]])
    np.testing.assert_almost_equal(rot.quats, expected_quats)
    np.testing.assert_equal(rot.times, np.asarray(times))
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

def test_from_matrix():
    mat = [[0, 0, 1],
           [1, 0 ,0],
           [0, 1, 0]]
    rot = ConstantRotation.from_matrix(mat, 0, 1)
    expected_quats = np.asarray([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_almost_equal(rot.quat, expected_quats)
    assert rot.source == 0
    assert rot.dest == 1

def test_reinterpolate():
    test_quats = Rotation.from_euler('x', np.array([-135, -90, 0, 45, 90]), degrees=True).as_quat()
    rot = TimeDependentRotation(test_quats, [-0.5, 0, 1, 1.5, 2], 1, 2)
    new_rot = rot.reinterpolate(np.arange(-3, 5))
    assert new_rot.source == rot.source
    assert new_rot.dest == rot.dest
    np.testing.assert_equal(new_rot.times, np.arange(-3, 5))
    np.testing.assert_almost_equal(new_rot.quats,
                                   np.array([[0, 0, 0, -1],
                                             [-1/np.sqrt(2), 0, 0, -1/np.sqrt(2)],
                                             [-1, 0, 0, 0],
                                             [-1/np.sqrt(2), 0, 0, 1/np.sqrt(2)],
                                             [0, 0, 0, 1],
                                             [1/np.sqrt(2), 0, 0, 1/np.sqrt(2)],
                                             [1, 0, 0, 0],
                                             [1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]]))

def test_reinterpolate_single_time():
    rot = TimeDependentRotation([[0, 0, 0, 1]], [0], 1, 2)
    new_rot = rot.reinterpolate([-1, 3])
    assert new_rot.source == rot.source
    assert new_rot.dest == rot.dest
    np.testing.assert_equal(new_rot.times, np.asarray([-1 ,3]))
    np.testing.assert_almost_equal(new_rot.quats,
                                   np.asarray([[0, 0, 0, 1], [0, 0, 0, 1]]))
