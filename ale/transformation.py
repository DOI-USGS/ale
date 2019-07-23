import numpy as np
from numpy.polynomial.polynomial import polyval, polyder
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

import spiceypy as spice

from ale.rotation import ConstantRotation, TimeDependentRotation

def create_rotations(rotation_table):
    """
    Convert an ISIS rotation table into rotation objects.

    Parameters
    ----------
    rotation_table : dict
                     The rotation ISIS table as a dictionary

    Returns
    -------
    : list
      A list of time dependent or constant rotation objects from the table. This
      list will always have either 1 or 2 elements. The first rotation will be
      time dependent and the second rotation will be constant. The rotations will
      be ordered such that the reference frame the first rotation rotates to is
      the reference frame the second rotation rotates from.
    """
    rotations = []
    root_frame = rotation_table['TimeDependentFrames'][-1]
    last_time_dep_frame = rotation_table['TimeDependentFrames'][0]
    # Case 1: It's a table of quaternions and times
    if 'J2000Q0' in rotation_table:
        # SPICE quaternions are (W, X, Y, Z) and ALE uses (X, Y, Z, W).
        quats = np.array([rotation_table['J2000Q1'],
                          rotation_table['J2000Q2'],
                          rotation_table['J2000Q3'],
                          rotation_table['J2000Q0']]).T
        time_dep_rot = TimeDependentRotation(quats,
                                             rotation_table['ET'],
                                             root_frame,
                                             last_time_dep_frame)
        rotations.append(time_dep_rot)
    # Case 2: It's a table of Euler angle coefficients
    elif 'J2000Ang1' in rotation_table:
        ephemeris_times = np.linspace(rotation_table['CkTableStartTime'],
                                      rotation_table['CkTableEndTime'],
                                      rotation_table['CkTableOriginalSize'])
        base_time = rotation_table['J2000Ang1'][-1]
        time_scale = rotation_table['J2000Ang2'][-1]
        scaled_times = (ephemeris_times - base_time) / time_scale
        coeffs = np.array([rotation_table['J2000Ang1'][:-1],
                           rotation_table['J2000Ang2'][:-1],
                           rotation_table['J2000Ang3'][:-1]]).T
        angles = polyval(scaled_times, coeffs).T
        # ISIS is hard coded to ZXZ (313) Euler angle axis order.
        time_dep_rot = TimeDependentRotation.from_euler('zxz',
                                                        angles,
                                                        ephemeris_times,
                                                        root_frame,
                                                        last_time_dep_frame,
                                                        degrees=True)
        rotations.append(time_dep_rot)

    if 'ConstantRotation' in rotation_table:
        last_constant_frame = rotation_table['ConstantFrames'][0]
        rot_mat =  np.reshape(np.array(rotation_table['ConstantRotation']), (3, 3))
        constant_rot = ConstantRotation.from_matrix(rot_mat,
                                                    last_time_dep_frame,
                                                    last_constant_frame)
        rotations.append(constant_rot)
    return rotations

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

    @classmethod
    def from_isis_tables(cls, *args, inst_pointing = {}, body_orientation={}, **kwargs):
        frame_chain = cls()

        for rotation in create_rotations(inst_pointing):
            frame_chain.add_edge(rotation.source,
                                 rotation.dest,
                                 rotation=rotation)

        for rotation in create_rotations(body_orientation):
            frame_chain.add_edge(rotation.source,
                                 rotation.dest,
                                 rotation=rotation)
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
