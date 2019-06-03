import numpy as np

from ale.rotation import ConstantRotation, TimeDependentRotation

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
            self.parent = parent
        if rotation is not None:
            self.rotation = rotation

    def __repr__(self):
        return f'ID:{self.id}\nChildren:{self.children}'

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
            self._parent.children.remove(self)

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

    def find_child_frame(self, id):
        """
        Find a child frame by its frame ID.

        Recursively search this frame and all of its children for a specific
        reference frame.

        Parameters
        ----------
        id : int
            The NAIF frame ID of the frame to find

        Returns
        -------
        FrameNode
            The specified frame. If no child frame with the given ID exists,
            None is returned.
        """
        if self.id == id:
            return self
        node = None
        for child in self.children:
            node = child.find_child_frame(id)
            if node is not None:
                return node
        return node

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

    def last_time_dependent_frame_between(self, other):
        """
        Find the last time dependent frame between this frame and another frame.

        Parameters
        ----------
        other : FrameNode
            The frame to find the last time dependent frame between

        Returns
        -------
        FrameNode
            The first frame between the this frame and the other frame such
            that the rotation from this frame to the in-between frame is time
            dependent and the rotation from the in-between frame to the other
            frame is constant. If there are no time dependent frames between
            this frame and the other frame, this frame is returned.
        """
        forward_path, reverse_path = self.path_to(other)
        # Reverse search the rotation chain for the last time dependent rotation
        for node in reverse_path[::-1]:
            if isinstance(node.rotation, TimeDependentRotation):
                return node
        for node in forward_path[:-1][::-1]:
            if isinstance(node.rotation, TimeDependentRotation):
                return node.parent
        return self
