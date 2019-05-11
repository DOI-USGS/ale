import pytest

import numpy as np
from ale.rotation import ConstantRotation, TimeDependentRotation

def test_constant_constant_composition():
    # Two 90 degree rotation about the X-axis
    rot1_2 = ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 1, 2)
    rot2_3 = ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 2, 3)
    # compose to get a 180 degree rotation about the X-axis
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, ConstantRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    np.testing.assert_equal(rot1_3.quat, np.array([1, 0, 0, 0]))

def test_constant_time_dependent_composition():
    # 90 degree rotation about the X-axis to a 180 degree rotation about the X-axis
    quats = np.array([[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]])
    times = np.array([0, 1])
    rot1_2 = TimeDependentRotation(quats, times, 1, 2)
    # 90 degree rotation about the X-axis
    rot2_3 = ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 2, 3)
    # compose to get a 180 degree rotation about the X-axis to a 270 degree rotation about the X-axis
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_equats = np.array([[1, 0, 0, 0],[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)]])
    np.testing.assert_equal(rot1_3.times, times)
    np.testing.assert_almost_equal(rot1_3.quats, expected_equats)

def test_time_dependent_constant_composition():
    # 90 degree rotation about the X-axis
    rot1_2 = ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 1, 2)
    # 90 degree rotation about the X-axis to a 180 degree rotation about the X-axis
    quats = np.array([[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]])
    times = np.array([0, 1])
    rot2_3 = TimeDependentRotation(quats, times, 2, 3)
    # compose to get a 180 degree rotation about the X-axis to a 270 degree rotation about the X-axis
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_equats = np.array([[1, 0, 0, 0],[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)]])
    np.testing.assert_equal(rot1_3.times, times)
    np.testing.assert_almost_equal(rot1_3.quats, expected_equats)

def test_time_dependent_time_dependent_composition():
    # 90 degree rotation about the X-axis to a 180 degree rotation about the X-axis
    quats1_2 = np.array([[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)],[1, 0, 0, 0]])
    times1_2 = np.array([0, 1])
    rot1_2 = TimeDependentRotation(quats1_2, times1_2, 1, 2)
    # -90 degree rotation about the X-axis to a 90 degree rotation about the X-axis
    quats2_3 = np.array([[1.0/np.sqrt(2), 0, 0, -1.0/np.sqrt(2)],[1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]])
    times2_3 = np.array([0, 2])
    rot2_3 = TimeDependentRotation(quats2_3, times2_3, 2, 3)
    # compose to get no rotation to a 180 degree rotation about the X-axis to no rotation
    rot1_3 = rot2_3*rot1_2
    assert isinstance(rot1_3, TimeDependentRotation)
    assert rot1_3.source == 1
    assert rot1_3.dest == 3
    expected_times = np.array([0, 1])
    expected_equats = np.array([[0, 0, 0, -1],[-1, 0, 0, 0]])
    np.testing.assert_equal(rot1_3.times, expected_times)
    np.testing.assert_almost_equal(rot1_3.quats, expected_equats)
