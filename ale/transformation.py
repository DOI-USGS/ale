from multiprocessing.pool import ThreadPool

import numpy as np
from numpy.polynomial.polynomial import polyval
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

from ale.spiceql_access import get_ephem_data, spiceql_call
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
        if 'AV1' in rotation_table:
            av = np.array([rotation_table['AV1'],
                           rotation_table['AV2'],
                           rotation_table['AV3']]).T
        else:
            av = []
        time_dep_rot = TimeDependentRotation(quats,
                                             rotation_table['ET'],
                                             root_frame,
                                             last_time_dep_frame,
                                             av=av)
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
        # SPICE also interprets Euler angle rotations as negative rotations,
        # so negate them before passing to scipy.
        time_dep_rot = TimeDependentRotation.from_euler('zxz',
                                                        -angles,
                                                        ephemeris_times,
                                                        root_frame,
                                                        last_time_dep_frame)
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
    def __init__(self, use_web=False, search_kernels=False, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.use_web = use_web
        self.search_kernels = search_kernels

    @classmethod
    def from_spice(cls, sensor_frame, target_frame, center_ephemeris_time, 
                                                    ephemeris_times=[], 
                                                    nadir=False, 
                                                    exact_ck_times=False,
                                                    inst_time_bias=0,
                                                    use_web=False, 
                                                    search_kernels=False, 
                                                    mission=""):
        frame_chain = cls(use_web, search_kernels)
        # Default assume one time
        target_times = ephemeris_times
        if len(target_times) > 1:
            target_times = [ephemeris_times[0], ephemeris_times[-1]]
        
        sensor_times = []
        if exact_ck_times and len(ephemeris_times) > 1 and not nadir:
            try:
                sensor_times = spiceql_call("extractExactCkTimes", {"observStart": ephemeris_times[0] + inst_time_bias, 
                                                                    "observEnd": ephemeris_times[-1] + inst_time_bias, 
                                                                    "targetFrame": sensor_frame,
                                                                    "mission": mission,
                                                                    "searchKernels": frame_chain.search_kernels},
                                                                    use_web=frame_chain.use_web)
                print("TIMES: ", sensor_times)
            except Exception as e:
                pass

        if (len(sensor_times) == 0):
            sensor_times = ephemeris_times
            if isinstance(sensor_times, np.ndarray):
                sensor_times = sensor_times.tolist()

        frames = frame_chain.frame_trace(center_ephemeris_time, sensor_frame, target_frame, nadir, mission)
        sensor_time_dependent_frames, sensor_constant_frames = frames[0]
        target_time_dependent_frames, target_constant_frames = frames[1]

        sensor_time_dependent_frames = list(zip(sensor_time_dependent_frames[:-1], sensor_time_dependent_frames[1:]))
        constant_frames = list(zip(sensor_constant_frames[:-1], sensor_constant_frames[1:]))
        target_time_dependent_frames = list(zip(target_time_dependent_frames[:-1], target_time_dependent_frames[1:]))
        target_constant_frames = list(zip(target_constant_frames[:-1], target_constant_frames[1:]))

        constant_frames.extend(target_constant_frames)

        frame_tasks = []
        # Add all time dependent frame edges to the graph
        frame_tasks.append([sensor_time_dependent_frames, sensor_times, inst_time_bias, TimeDependentRotation, mission])
        frame_tasks.append([target_time_dependent_frames, target_times, 0, TimeDependentRotation, mission])

        # Add all constant frames to the graph
        frame_tasks.append([constant_frames, [ephemeris_times[0]], 0, ConstantRotation, mission])

        # Build graph async
        with ThreadPool() as pool:
            jobs = pool.starmap_async(frame_chain.generate_rotations, frame_tasks)
            jobs.get()

        return frame_chain

    def frame_trace(self, time, sensorFrame, targetFrame, nadir=False, mission=""):
        jobs = []
        if not nadir:
            jobs.append({"et": time, 
                        "initialFrame": sensorFrame,
                        "mission": mission,
                        "searchKernels": self.search_kernels})
        jobs.append({"et": time, 
                     "initialFrame": targetFrame,
                     "mission": mission,
                     "searchKernels": self.search_kernels})
        with ThreadPool() as pool:
            jobs = pool.starmap_async(spiceql_call, [("frameTrace", job, self.use_web) for job in jobs])
            results = jobs.get()

        if nadir:
            results.insert(0, [[], []])

        return results

    @classmethod
    def from_isis_tables(cls, *args, inst_pointing={}, body_orientation={}, **kwargs):
        frame_chain = cls()

        for rotation in create_rotations(inst_pointing):
            frame_chain.add_edge(rotation=rotation)

        for rotation in create_rotations(body_orientation):
            frame_chain.add_edge(rotation=rotation)
        return frame_chain

    def add_edge(self, rotation, **kwargs):
        super(FrameChain, self).add_edge(rotation.source, rotation.dest, rotation=rotation, **kwargs)
        rotation = rotation.inverse()
        super(FrameChain, self).add_edge(rotation.source, rotation.dest, rotation=rotation, **kwargs)

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


    def generate_rotations(self, frames, times, time_bias, rotation_type, mission=""):
        """
        Computes the rotations based on a list of tuples that define the
        relationships between frames as (source, destination) and a list of times to
        compute the rotation at. The rotations are then appended to the frame chain
        object

        frames : list
                 A list of tuples that define the relationships between frames

        times : list
                A list of times to compute the rotation at
        """
        # Convert list of np.floats to ndarray
        if isinstance(times, list) and isinstance(times[0], np.floating):
            times = np.array(times)

        frame_tasks = []
        for s, d in frames:
            function_args = {"toFrame": d, "refFrame": s, "mission": mission, "searchKernels": self.search_kernels}
            frame_tasks.append([times, "getTargetOrientations", 400, self.use_web, function_args])
        with ThreadPool() as pool:
            jobs = pool.starmap_async(get_ephem_data, frame_tasks)
            quats_and_avs_per_frame = jobs.get()

        for i, frame in enumerate(frames):
            quats_and_avs = quats_and_avs_per_frame[i]
            quats = np.zeros((len(times), 4))
            avs = []
            _quats = np.array(quats_and_avs)[:, 0:4]
            for j, quat in enumerate(_quats):
                quats[j,:3] = quat[1:]
                quats[j,3] = quat[0]

            if (len(quats_and_avs[0]) > 4):
                avs = np.array(quats_and_avs)[:, 4:]

            biased_times = [time - time_bias for time in times]
            if rotation_type == TimeDependentRotation:
                rotation = TimeDependentRotation(quats, biased_times, frame[0], frame[1], av=avs)
            else:
                rotation = ConstantRotation(quats[0], frame[0], frame[1])
            self.add_edge(rotation=rotation)
