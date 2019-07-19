import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

import spiceypy as spice

from ale.rotation import ConstantRotation, TimeDependentRotation

class FrameChain(nx.DiGraph):
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
    def __init__(self, *args, frame_changes = [], ephemeris_time=[], **kwargs):
        super(FrameChain, self).__init__(*args, **kwargs)

        times = np.array(ephemeris_time)
        quats = np.zeros((len(times), 4))

        for s, d in frame_changes:
            for i, time in enumerate(times):
                rotation_matrix = spice.pxform(spice.frmnam(s), spice.frmnam(d), time)
                quat_from_rotation = spice.m2q(rotation_matrix)
                quats[i,:3] = quat_from_rotation[1:]
                quats[i,3] = quat_from_rotation[0]
            rotation = TimeDependentRotation(quats, times, s, d)
            self.add_edge(s, d, rotation = rotation)
            self.add_edge(d, s, rotation = rotation.inverse())

    def compute_rotation(self, source, destination):
        """
        Returns the rotation to another node. Returns the identity rotation
        if the other node is this node.

        Parameters
        ----------
        other : FrameNode
                The other node to find the rotation to.
        """
        if source == destination:
            return ConstantRotation(np.array([0, 0, 0, 1]), source, destination)

        path = shortest_path(self, source, destination)
        rotations = [self.edges[path[i], path[i+1]]['rotation'] for i in range(len(path) - 1)]
        rotation = rotations[0]
        for next_rotation in rotations[1:]:
            rotation = next_rotation * rotation
        return rotation

    def last_time_dependent_frame_between(self, source, destination):
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
        path = shortest_path(self, source, destination)
        path.reverse()
        for i in range(len(path) - 1):
            edge = self.edges[path[i+1], path[i]]
            if isinstance(edge['rotation'], TimeDependentRotation):
                return path[i+1], path[i], edge

        return None
