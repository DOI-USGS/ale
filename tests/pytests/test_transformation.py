import pytest

import numpy as np
from ale.rotation import ConstantRotation, TimeDependentRotation
from ale.transformation import FrameChain, create_rotations

def test_create_quaternion_rotations():
    test_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'ET' : [0, 1],
        'TimeDependentFrames' : [-1000, -100, 1]}
    rotations = create_rotations(test_table)
    assert len(rotations) == 1
    assert rotations[0].source == 1
    assert rotations[0].dest == -1000
    np.testing.assert_equal(rotations[0].times, [0, 1])
    np.testing.assert_almost_equal(rotations[0].quats, [[0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5]])

def test_create_quaternion_av_rotations():
    test_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'AV1' : [0, 1],
        'AV2' : [2, 3],
        'AV3' :[4, 5],
        'ET' : [0, 1],
        'TimeDependentFrames' : [-1000, -100, 1]}
    rotations = create_rotations(test_table)
    assert len(rotations) == 1
    assert rotations[0].source == 1
    assert rotations[0].dest == -1000
    np.testing.assert_equal(rotations[0].times, [0, 1])
    np.testing.assert_almost_equal(rotations[0].quats, [[0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5]])
    np.testing.assert_almost_equal(rotations[0].av, [[0, 2, 4], [1, 3, 5]])

def test_create_two_rotations():
    test_table = {
        'J2000Q0' : [1, 0.5],
        'J2000Q1' : [0, 0.5],
        'J2000Q2' : [0, 0.5],
        'J2000Q3' : [0, 0.5],
        'ET' : [0, 1],
        'TimeDependentFrames' : [-1000, -100, 1],
        'ConstantRotation' : [0, 0, 1, 1, 0 , 0, 0, 1, 0],
        'ConstantFrames' : [-1020, -1000]}
    rotations = create_rotations(test_table)
    assert len(rotations) == 2
    assert rotations[1].source == -1000
    assert rotations[1].dest == -1020
    np.testing.assert_almost_equal(rotations[1].quat, np.array([0.5, 0.5, 0.5, 0.5]))

def test_create_euler_rotations():
    test_table = {
        'J2000Ang1' : [0, 0, 10],
        'J2000Ang2' : [-np.pi/2, np.pi/2, 2],
        'J2000Ang3' : [-np.pi/2, np.pi/2, 1],
        'CkTableStartTime' : 10,
        'CkTableEndTime' : 12,
        'CkTableOriginalSize' : 2,
        'TimeDependentFrames' : [-1000, -100, 1]}
    rotations = create_rotations(test_table)
    assert len(rotations) == 1
    assert rotations[0].source == 1
    assert rotations[0].dest == -1000
    np.testing.assert_equal(rotations[0].times, np.array([10, 12]))
    np.testing.assert_almost_equal(rotations[0].quats, np.array([[0.5, 0.5, 0.5, 0.5], [0, 0, 0, 1]]))

@pytest.fixture(scope='function')
def frame_tree(request):
    """
    Test frame tree structure:

          1
         / \
        /   \
       2     4
      /
     /
    3
    """
    frame_chain = FrameChain()

    rotations = [
        ConstantRotation(np.array([1, 0, 0, 0]), 2, 1),
        ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 3, 2),
        ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 4, 1)
    ]
    frame_chain.add_edge(rotation = rotations[0])
    frame_chain.add_edge(rotation = rotations[1])
    frame_chain.add_edge(rotation = rotations[2])

    return frame_chain, rotations

def test_parent_rotation(frame_tree):
    frame_chain, rotations = frame_tree
    child_to_root = frame_chain.compute_rotation(2, 1)
    root_to_child = frame_chain.compute_rotation(1, 2)

    assert child_to_root.source == 2
    assert child_to_root.dest == 1
    np.testing.assert_equal(child_to_root.quat, rotations[0].quat)
    assert root_to_child.source == 1
    assert root_to_child.dest == 2
    np.testing.assert_equal(root_to_child.quat, rotations[0].inverse().quat)

def test_grand_parent_rotation(frame_tree):
    frame_chain, rotations = frame_tree
    child_2_to_root = frame_chain.compute_rotation(3, 1)
    root_to_child_2 = frame_chain.compute_rotation(1, 3)

    assert child_2_to_root.source == 3
    assert child_2_to_root.dest == 1
    expected_rotation_1 = rotations[0] * rotations[1]
    np.testing.assert_equal(child_2_to_root.quat, expected_rotation_1.quat)
    assert root_to_child_2.source == 1
    assert root_to_child_2.dest == 3
    expected_rotation_2 = rotations[1].inverse() * rotations[0].inverse()
    np.testing.assert_equal(root_to_child_2.quat, expected_rotation_2.quat)

def test_common_parent_rotation(frame_tree):
    frame_chain, rotations = frame_tree
    child_2_to_child_3 = frame_chain.compute_rotation(3, 4)
    child_3_to_child_2 = frame_chain.compute_rotation(4, 3)

    assert child_2_to_child_3.source == 3
    assert child_2_to_child_3.dest == 4
    expected_rotation_1 = rotations[2].inverse() * rotations[0] * rotations[1]
    np.testing.assert_equal(child_2_to_child_3.quat, expected_rotation_1.quat)
    assert child_3_to_child_2.source == 4
    assert child_3_to_child_2.dest == 3
    expected_rotation_2 = rotations[1].inverse() * rotations[0].inverse() * rotations[2]
    np.testing.assert_equal(child_3_to_child_2.quat, expected_rotation_2.quat)

def test_self_rotation(frame_tree):
    frame_chain, _ = frame_tree
    rotation = frame_chain.compute_rotation(1, 1)
    assert rotation.source == 1
    assert rotation.dest == 1
    np.testing.assert_equal(rotation.quat, np.array([0, 0, 0, 1]))

def test_no_dependent_frames_between(frame_tree):
    frame_chain, _ = frame_tree
    last_frame = frame_chain.last_time_dependent_frame_between(1, 3)
    assert last_frame == None

def test_last_time_dependent_frame_between():
    """
    Test frame tree structure:

          1
         / \
        /   \
       2     4
      /       \
     /         \
    3           5

    The rotations from 3 to 2 and 1 to 4 are time dependent.
    All other rotations are constant.
    """
    frame_chain = FrameChain()

    rotations = [
        ConstantRotation(np.array([1, 0, 0, 0]), 2, 1),
        TimeDependentRotation(
            np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]),
            np.array([1]), 3, 2),
        TimeDependentRotation(
            np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]),
            np.array([1]), 4, 1),
        ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 5, 4)
    ]
    frame_chain.add_edge(rotation = rotations[0])
    frame_chain.add_edge(rotation = rotations[1])
    frame_chain.add_edge(rotation = rotations[2])
    frame_chain.add_edge(rotation = rotations[3])

    # last frame from node 1 to node 3
    s31, d31, _ = frame_chain.last_time_dependent_frame_between(1, 3)
    assert s31 == 2
    assert d31 == 3
    # last frame from node 3 to node 1
    s13, d13, _ = frame_chain.last_time_dependent_frame_between(3, 1)
    assert s13 == 3
    assert d13 == 2
    # last frame from node 3 to node 5
    s35, d35, _ = frame_chain.last_time_dependent_frame_between(3, 5)
    assert s35 == 1
    assert d35 == 4
    # last frame from node 5 to node 3
    s53, d53, _ = frame_chain.last_time_dependent_frame_between(5, 3)
    assert s53 == 2
    assert d53 == 3
