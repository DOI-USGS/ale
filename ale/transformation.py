import numpy as np

from ale.rotation import ConstantRotation

class FrameNode():
    """
    A single frame in a frame tree. This class is largely adapted from the Node
    class in the vispy scene graph implementation.

    Attributes
    __________
    id : int
         The NAIF ID code for the frame
    parent : FrameNode
             The parent node in the frame tree
    children : List of FrameNode
               The children nodes in the frame tree
    rotation : ConstantRotation or TimeDependentRotation
               The rotation from this frame to the frame of the parent node
    """

    def __init__(self, id, parent=None, rotation=None):
        """
        Construct a frame node with or without a parent. If a parent is specified
        then a rotation from this frame to the parent node's frame must be
        specified and visa-versa.

        Parameters
        ----------
        id : int
             The NAIF ID code for the frame
        parent : FrameNode
                 The parent node in the frame chain
        rotation : ConstantRotation or TimeDependentRotation
                   The rotation from this frame to the frame of the parent node
        """

        if (parent is None) != (rotation is None):
            raise TypeError("parent and rotation must both be entered or both be None.")

        self.children = []
        self.id = id
        if parent is not None:
            self._parent = parent
        if rotation is not None:
            self.rotation = rotation

    def __del__(self):
        """
        Custom deletor for a FrameNode. The child node is always responsible
        for updating the parent nodes. So this removes this node from the
        children of its parent node.
        """
        if self.parent is not None:
            self.parent.children.remove(self)

    @property
    def parent(self):
        """
        The parent node of this node. Returns None if this is a root node.
        """
        if hasattr(self, '_parent'):
            return self._parent
        else:
            return None

    @parent.setter
    def parent(self, new_parent):
        """
        Sets a new parent node. The child node is always responsible for
        updating the parent node. So, this removes this node from the children
        of the old parent and adds it to the children of the new parent.
        """
        if self.parent is not None:
            self.parent.children.remove(self)

        new_parent.children.append(self)
        self._parent = new_parent

    def parent_nodes(self):
        """
        Returns the ordered list of parents starting with this node going to
        the root node.
        """
        chain = [self]
        current_parent = self.parent
        while current_parent is not None:
            chain.append(current_parent)
            current_parent = current_parent.parent
        return chain

    def path_to(self, other):
        """
        Returns the path to another node as two lists. The first list
        starts with this node and ends with the common parent. The second
        list contains the remainder of the path.

        Parameters
        ----------
        other : FrameNode
                The other node to find the path to.
        """
        parents_1 = self.parent_nodes()
        parents_2 = other.parent_nodes()
        common_parent = None
        for node in parents_1:
            if node in parents_2:
                common_parent = node
                break
        if common_parent is None:
            raise RuntimeError('No common parent between nodes')

        first_path = parents_1[:parents_1.index(common_parent)+1]
        second_path = parents_2[:parents_2.index(common_parent)][::-1]

        return first_path, second_path

    def rotation_to(self, other):
        """
        Returns the rotation to another node. Returns the identity rotation
        if the other node is this node.

        Parameters
        ----------
        other : FrameNode
                The other node to find the rotation to.
        """
        if other == self:
            return ConstantRotation(np.array([0, 0, 0, 1]), self.id, other.id)
        forward_path, reverse_path = self.path_to(other)
        rotations = [node.rotation for node in forward_path[:-1]]
        rotations.extend([node.rotation.inverse() for node in reverse_path])
        rotation = rotations[0]
        for next_rotation in rotations[1:]:
            rotation = next_rotation * rotation
        return rotation
