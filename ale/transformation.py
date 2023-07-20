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
        if 'AV1' in rotation_table:
            av = np.array([rotation_table['AV1'],
                           rotation_table['AV2'],
                           rotation_table['AV3']]).T
        else:
            av = None
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
    @classmethod
    def from_spice(cls, sensor_frame, target_frame, center_ephemeris_time, ephemeris_times=[], nadir=False, exact_ck_times=False, inst_time_bias=0):
        frame_chain = cls()
        sensor_times = []
        # Default assume one time
        target_times = np.asarray(ephemeris_times)
        if len(target_times) > 1:
            target_times = np.asarray([ephemeris_times[0], ephemeris_times[-1]])

        if exact_ck_times and len(ephemeris_times) > 1 and not nadir:
            try:
                sensor_times = cls.extract_exact_ck_times(ephemeris_times[0] + inst_time_bias, ephemeris_times[-1] + inst_time_bias, sensor_frame)
            except Exception as e:
                pass

        if (len(sensor_times) == 0):
            sensor_times = np.array(ephemeris_times)

        sensor_time_dependent_frames, sensor_constant_frames = cls.frame_trace(sensor_frame, center_ephemeris_time, nadir)
        target_time_dependent_frames, target_constant_frames = cls.frame_trace(target_frame, center_ephemeris_time)

        sensor_time_dependent_frames = list(zip(sensor_time_dependent_frames[:-1], sensor_time_dependent_frames[1:]))
        constant_frames = list(zip(sensor_constant_frames[:-1], sensor_constant_frames[1:]))
        target_time_dependent_frames = list(zip(target_time_dependent_frames[:-1], target_time_dependent_frames[1:]))
        target_constant_frames = list(zip(target_constant_frames[:-1], target_constant_frames[1:]))

        constant_frames.extend(target_constant_frames)

        frame_chain.compute_time_dependent_rotiations(sensor_time_dependent_frames, sensor_times, inst_time_bias)
        frame_chain.compute_time_dependent_rotiations(target_time_dependent_frames, target_times, 0)

        for s, d in constant_frames:
            quats = np.zeros(4)
            rotation_matrix = spice.pxform(spice.frmnam(s), spice.frmnam(d), ephemeris_times[0])
            quat_from_rotation = spice.m2q(rotation_matrix)
            quats[:3] = quat_from_rotation[1:]
            quats[3] = quat_from_rotation[0]

            rotation = ConstantRotation(quats, s, d)

            frame_chain.add_edge(rotation=rotation)

        return frame_chain

    @staticmethod
    def frame_trace(reference_frame, ephemeris_time, nadir=False):
        if nadir:
            return [], []

        frame_codes = [reference_frame]
        _, frame_type, _ = spice.frinfo(frame_codes[-1])
        frame_types = [frame_type]


        while(frame_codes[-1] != 1):
            try:
                center, frame_type, frame_type_id = spice.frinfo(frame_codes[-1])
            except Exception as e:
                print(e)
                break

            if frame_type == 1 or frame_type == 2:
                frame_code = 1

            elif frame_type == 3:
                try:
                    matrix, frame_code = spice.ckfrot(frame_type_id, ephemeris_time)
                except:
                    raise Exception(f"The ck rotation from frame {frame_codes[-1]} can not " +
                                    f"be found due to no pointing available at requested time {ephemeris_time} " +
                                     "or a problem with the frame")
            elif frame_type == 4:
                try:
                    matrix, frame_code = spice.tkfram(frame_type_id)
                except:
                    raise Exception(f"The tk rotation from frame {frame_codes[-1]} can not " +
                                     "be found")
            elif frame_type == 5:
                matrix, frame_code = spice.zzdynrot(frame_type_id, center, ephemeris_time)

            else:
                raise Exception(f"The frame {frame_codes[-1]} has a type {frame_type_id} " +
                                  "not supported by your version of Naif Spicelib. " +
                                  "You need to update.")

            frame_codes.append(frame_code)
            frame_types.append(frame_type)
        constant_frames = []
        while frame_codes:
            if frame_types[0] == 4:
                constant_frames.append(frame_codes.pop(0))
                frame_types.pop(0)
            else:
                break

        time_dependent_frames = []
        if len(constant_frames) != 0:
            time_dependent_frames.append(constant_frames[-1])

        while frame_codes:
            time_dependent_frames.append(frame_codes.pop(0))

        return time_dependent_frames, constant_frames

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

    @staticmethod
    def extract_exact_ck_times(observStart, observEnd, targetFrame):
        """
        Generates all exact ephemeris data assocaited with a specific frame as
        defined by targetFrame, between a start and end interval defined by 
        observStart and observEnd

        Parameters
        ----------
        observStart : float
                      Start time in ephemeris time to extract ephemeris data from

        observEnd : float
                    End time in ephemeris time to extract ephemeris data to

        targetFrame : int
                      Target reference frame to get ephemeris data in

        Returns
        -------
        times : list
                A list of times where exact ephemeris data where recorded for
                the targetFrame
        """
        times = []

        FILESIZ = 128;
        TYPESIZ = 32;
        SOURCESIZ = 128;

        currentTime = observStart

        count = spice.ktotal("ck")
        if (count > 1):
            msg = "Unable to get exact CK record times when more than 1 CK is loaded, Aborting"
            raise Exception(msg)

        _, _, _, handle = spice.kdata(0, "ck", FILESIZ, TYPESIZ, SOURCESIZ)
        spice.dafbfs(handle)
        found = spice.daffna()
        spCode = int(targetFrame / 1000) * 1000

        while found:
            observationSpansToNextSegment = False
            summary = spice.dafgs()
            dc, ic = spice.dafus(summary, 2, 6)

            # Don't read type 5 ck here
            if ic[2] == 5:
                break

            if (ic[0] == spCode and ic[2] == 3):
                segStartEt = spice.sct2e(int(spCode/1000), dc[0])
                segStopEt = spice.sct2e(int(spCode/1000), dc[1])

                if (currentTime >= segStartEt  and  currentTime <= segStopEt):
                    # Check for a gap in the time coverage by making sure the time span of the observation
                    #  does not cross a segment unless the next segment starts where the current one ends
                    if (observationSpansToNextSegment and currentTime > segStartEt):
                        msg = "Observation crosses segment boundary--unable to interpolate pointing"
                        raise Exception(msg)
                    if (observEnd > segStopEt):
                        observationSpansToNextSegment = True

                    dovelocity = ic[3]
                    end = ic[5]
                    val = spice.dafgda(handle, int(end - 1), int(end))
                    # int nints = (int) val[0];
                    ninstances = int(val[1])
                    numvel  =  dovelocity * 3
                    quatnoff  =  ic[4] + (4 + numvel) * ninstances - 1
                    # int nrdir = (int) (( ninstances - 1 ) / DIRSIZ); /* sclkdp directory records */
                    sclkdp1off  =  int(quatnoff + 1)
                    sclkdpnoff  =  int(sclkdp1off + ninstances - 1)
                    # int start1off = sclkdpnoff + nrdir + 1;
                    # int startnoff = start1off + nints - 1;
                    sclkSpCode = int(spCode / 1000)

                    sclkdp = spice.dafgda(handle, sclkdp1off, sclkdpnoff)

                    instance = 0
                    et = spice.sct2e(sclkSpCode, sclkdp[0])

                    while (instance < (ninstances - 1)  and  et < currentTime):
                        instance = instance + 1
                        et = spice.sct2e(sclkSpCode, sclkdp[instance])

                    if (instance > 0):
                        instance = instance - 1
                    et = spice.sct2e(sclkSpCode, sclkdp[instance])

                    while (instance < (ninstances - 1) and et < observEnd):
                        times.append(et)
                        instance = instance + 1
                        et = spice.sct2e(sclkSpCode, sclkdp[instance])
                    times.append(et)

                    if not observationSpansToNextSegment:
                        break
                    else:
                        currentTime = segStopEt
            spice.dafcs(handle)     # Continue search in daf last searched
            found = spice.daffna()   # Find next forward array in current daf

        return times

    def compute_time_dependent_rotiations(self, frames, times, time_bias):
        """
        Computes the time dependent rotations based on a list of tuples that define the
        relationships between frames as (source, destination) and a list of times to
        compute the rotation at. The rotations are then appended to the frame chain
        object

        frames : list
                 A list of tuples that define the relationships between frames

        times : list
                A list of times to compute the rotation at
        """
        for s, d in frames:
            quats = np.zeros((len(times), 4))
            avs = []
            for j, time in enumerate(times):
                try:
                    state_matrix = spice.sxform(spice.frmnam(s), spice.frmnam(d), time)
                    rotation_matrix, av = spice.xf2rav(state_matrix)
                    avs.append(av)
                except:
                    rotation_matrix = spice.pxform(spice.frmnam(s), spice.frmnam(d), time)
                quat_from_rotation = spice.m2q(rotation_matrix)
                quats[j,:3] = quat_from_rotation[1:]
                quats[j,3] = quat_from_rotation[0]

            if not avs:
                avs = None
            biased_times = [time - time_bias for time in times]
            rotation = TimeDependentRotation(quats, biased_times, s, d, av=avs)
            self.add_edge(rotation=rotation)