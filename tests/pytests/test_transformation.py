import pytest

import numpy as np
from ale.rotation import ConstantRotation
from ale.transformation import FrameNode

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
    rotations = [
        ConstantRotation(np.array([1, 0, 0, 0]), 2, 1),
        ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 3, 2),
        ConstantRotation(np.array([1.0/np.sqrt(2), 0, 0, 1.0/np.sqrt(2)]), 4, 1)
    ]
    root_node = FrameNode(1)
    child_node_1 = FrameNode(2, parent = root_node, rotation = rotations[0])
    child_node_2 = FrameNode(3, parent = child_node_1, rotation = rotations[1])
    child_node_3 = FrameNode(4, parent = root_node, rotation = rotations[2])
    nodes = [
        root_node,
        child_node_1,
        child_node_2,
        child_node_3
    ]
    return (nodes, rotations)

def test_del_node(frame_tree):
    nodes, _ = frame_tree
    node3 = nodes[3]
    node2 = nodes[2]
    del node3

    assert len(node2.children) == 0

def test_parent_nodes(frame_tree):
    nodes, _ = frame_tree
    root_parents = nodes[0].parent_nodes()
    child_1_parents = nodes[1].parent_nodes()
    child_2_parents = nodes[2].parent_nodes()
    child_3_parents = nodes[3].parent_nodes()

    assert root_parents == [nodes[0]]
    assert child_1_parents == [nodes[1], nodes[0]]
    assert child_2_parents == [nodes[2], nodes[1], nodes[0]]
    assert child_3_parents == [nodes[3], nodes[0]]

def test_path_to_parent(frame_tree):
    nodes, _ = frame_tree
    forward_path, reverse_path = nodes[2].path_to(nodes[0])
    assert forward_path == [nodes[2], nodes[1], nodes[0]]
    assert reverse_path == []

def test_path_to_common_parent(frame_tree):
    nodes, _ = frame_tree
    forward_path, reverse_path = nodes[2].path_to(nodes[3])
    assert forward_path == [nodes[2], nodes[1], nodes[0]]
    assert reverse_path == [nodes[3]]

def test_path_to_child(frame_tree):
    nodes, _ = frame_tree
    forward_path, reverse_path = nodes[0].path_to(nodes[3])
    assert forward_path == [nodes[0]]
    assert reverse_path == [nodes[3]]

def test_path_to_self():
    node = FrameNode(1)
    forward_path, reverse_path = node.path_to(node)
    assert forward_path == [node]
    assert reverse_path == []


def test_parent_rotation(frame_tree):
    nodes, rotations = frame_tree
    child_to_root = nodes[1].rotation_to(nodes[0])
    root_to_child = nodes[0].rotation_to(nodes[1])

    assert child_to_root.source == 2
    assert child_to_root.dest == 1
    np.testing.assert_equal(child_to_root.quat, rotations[0].quat)
    assert root_to_child.source == 1
    assert root_to_child.dest == 2
    np.testing.assert_equal(root_to_child.quat, rotations[0].inverse().quat)

def test_grand_parent_rotation(frame_tree):
    nodes, rotations = frame_tree
    child_2_to_root = nodes[2].rotation_to(nodes[0])
    root_to_child_2 = nodes[0].rotation_to(nodes[2])

    assert child_2_to_root.source == 3
    assert child_2_to_root.dest == 1
    expected_rotation_1 = rotations[0] * rotations[1]
    np.testing.assert_equal(child_2_to_root.quat, expected_rotation_1.quat)
    assert root_to_child_2.source == 1
    assert root_to_child_2.dest == 3
    expected_rotation_2 = rotations[1].inverse() * rotations[0].inverse()
    np.testing.assert_equal(root_to_child_2.quat, expected_rotation_2.quat)

def test_common_parent_rotation(frame_tree):
    nodes, rotations = frame_tree
    child_2_to_child_3 = nodes[2].rotation_to(nodes[3])
    child_3_to_child_2 = nodes[3].rotation_to(nodes[2])

    assert child_2_to_child_3.source == 3
    assert child_2_to_child_3.dest == 4
    expected_rotation_1 = rotations[2].inverse() * rotations[0] * rotations[1]
    np.testing.assert_equal(child_2_to_child_3.quat, expected_rotation_1.quat)
    assert child_3_to_child_2.source == 4
    assert child_3_to_child_2.dest == 3
    expected_rotation_2 = rotations[1].inverse() * rotations[0].inverse() * rotations[2]
    np.testing.assert_equal(child_3_to_child_2.quat, expected_rotation_2.quat)

def test_self_rotation():
    node = FrameNode(1)
    rotation = node.rotation_to(node)
    assert rotation.source == 1
    assert rotation.dest == 1
    np.testing.assert_equal(rotation.quat, np.array([0, 0, 0, 1]))

def test_find_child_frame(frame_tree):
    nodes, rotations = frame_tree
    child_1 = nodes[0].find_child_frame(2)
    child_2 = nodes[0].find_child_frame(3)
    child_3 = nodes[0].find_child_frame(4)
    assert child_1 == nodes[1]
    assert child_2 == nodes[2]
    assert child_3 == nodes[3]
