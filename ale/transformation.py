import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

import spiceypy as spice

from ale.rotation import ConstantRotation, TimeDependentRotation

class FrameChain(nx.DiGraph):
    """
    This class is responsible for handling rotations between reference frames.
    Every node is the reference frame and every edge represents the rotation to
    between those two nodes. Each edge is directional, where the source --> destination
    is one rotation and destination --> source is the inverse of that rotation.

    Attributes
    __________
    frame_changes : list
                    A list of tuples that represent the rotation from one frame
                    to another. These tuples should all be NAIF codes for
                    reference frames
    ephemeris_time : list
                     A of ephemeris times that need to be rotated for each set
                     of frame rotations in the frame chain
    """
    @classmethod
    def from_spice(cls, *args, frame_changes = [], ephemeris_time=[], **kwargs):
        frame_chain = cls()

        times = np.array(ephemeris_time)
        quats = np.zeros((len(times), 4))

        for s, d in frame_changes:
            for i, time in enumerate(times):
                rotation_matrix = spice.pxform(spice.frmnam(s), spice.frmnam(d), time)
                quat_from_rotation = spice.m2q(rotation_matrix)
                quats[i,:3] = quat_from_rotation[1:]
                quats[i,3] = quat_from_rotation[0]
            rotation = TimeDependentRotation(quats, times, s, d)
            frame_chain.add_edge(s, d, rotation=rotation)
        return frame_chain

    def add_edge(self, s, d, rotation, **kwargs):
        super(FrameChain, self).add_edge(s, d, rotation=rotation, **kwargs)
        super(FrameChain, self).add_edge(d, s, rotation=rotation.inverse(), **kwargs)

    def compute_rotation(self, source, destination):
        """
        Returns the rotation to another node. Returns the identity rotation
        if the other node is this node.

        Parameters
        ----------
        source : int
                 Integer id for the source node to rotate from
        destination : int
                      Integer id for the node to rotate into from the source node

        Returns
        -------
        rotation : Object
                   Returns either a TimeDependentRotation object or ConstantRotation
                   object depending on the number of rotations being multiplied
                   together
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
        Find the last time dependent frame between the source frame and the
        destination frame.

        Parameters
        ----------
        source : int
                 Integer id of the source node

        destination : int
                      Integer of the destination node

        Returns
        -------
        : tuple, None
          Returns the source node id, destination node id, and edge dictionary
          which contains the rotation from source to destination.
        """
        path = shortest_path(self, source, destination)
        # Reverse the path to search bottom up to find the last time dependent
        # frame between the source and destination
        path.reverse()
        for i in range(len(path) - 1):
            edge = self.edges[path[i+1], path[i]]
            if isinstance(edge['rotation'], TimeDependentRotation):
                return path[i+1], path[i], edge

        return None
