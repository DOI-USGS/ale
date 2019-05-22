import pytest

import numpy as np
from ale.rotation import ConstantRotation
from ale.transformation import FrameNode

def test_one_parent_rotation():
    test_rotation = ConstantRotation(np.array([0, 0, 0, 1]), 2, 1)
    parent_node = FrameNode(1)
    child_node = FrameNode(2, parent = parent_node, rotation = test_rotation)

    child_to_parent = child_node.rotation_to(parent_node)
    parent_to_child = parent_node.rotation_to(child_node)

    assert child_to_parent.source == 2
    assert child_to_parent.dest == 1
    np.testing.assert_equal(child_to_parent.quat, test_rotation.quat)
    assert parent_to_child.source == 1
    assert parent_to_child.dest == 2
    np.testing.assert_equal(parent_to_child.quat, test_rotation.inverse().quat)

def test_multiple_parent_rotation():
    first_rotation = ConstantRotation(np.array([0, 0, 0, 1]), 2, 1)
    parent_node = FrameNode(1)
    child_node_1 = FrameNode(2, parent = parent_node, rotation = first_rotation)
    second_rotation = ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 3, 2)
    child_node_2 = FrameNode(2, parent = child_node_1, rotation = second_rotation)

    child_2_to_parent = child_node_2.rotation_to(parent_node)
    parent_to_child_2 = parent_node.rotation_to(child_node_2)

    assert child_2_to_parent.source == 3
    assert child_2_to_parent.dest == 1
    expected_rotation_1 = first_rotation * second_rotation
    np.testing.assert_equal(child_2_to_parent.quat, expected_rotation_1.quat)
    assert parent_to_child_2.source == 1
    assert parent_to_child_2.dest == 3
    expected_rotation_2 = second_rotation.inverse() * first_rotation.inverse()
    np.testing.assert_equal(parent_to_child_2.quat, expected_rotation_2.quat)
