from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation

import numpy as np

class ConstantRotation:
    """
    A constant rotation between two 3D reference frames.

    Attributes
    __________
    source : int
             The NAIF ID code for the source frame
    dest : int
           The NAIF ID code for the destination frame
    """

    def from_matrix(mat, source, dest):
        """
        Create a constant rotation from a directed cosine matrix

        Parameters
        ----------
        mat : 2darray
              The rotation matrix
        source : int
                 The NAIF ID code for the source frame
        dest : int
               The NAIF ID code for the destination frame

        See Also
        --------
        scipy.spatial.transform.Rotation.from_dcm
        """
        rot = Rotation.from_dcm(mat)
        return ConstantRotation(rot.as_quat(), source, dest)

    def __init__(self, quat, source, dest):
        """
        Construct a constant rotation

        Parameters
        ----------
        quat : array
               The quaternion representation of the rotation as a numpy array.
               The quaternion must be in scalar last format (x, y, z, w).
        source : int
                 The NAIF ID code for the source frame
        dest : int
               The NAIF ID code for the destination frame
        """
        self.source = source
        self.dest = dest
        self.quat = np.asarray(quat)

    def __repr__(self):
        return f'ConstantRotation Source: {self.source}, Destination: {self.dest}, Quat: {self.quat}'

    @property
    def quat(self):
        """
        The quaternion that rotates from the source reference frame to
        the destination reference frame. The quaternion is in scalar last
        format (x, y, z, w).
        """
        return self._rot.as_quat()

    @quat.setter
    def quat(self, new_quat):
        """
        Change the rotation to a different quaternion

        Parameters
        ----------
        new_quat : array
                   The new quaternion as an array.
                   The quaternion must be in scalar last format (x, y, z, w).
        """
        self._rot = Rotation.from_quat(np.asarray(new_quat))

    def rotation_matrix(self):
        """
        The rotation matrix representation of the constant rotation
        """
        return self._rot.as_dcm()

    def inverse(self):
        """
        Get the inverse rotation, that is the rotation from the destination
        reference frame to the source reference frame.
        """
        return ConstantRotation(self._rot.inv().as_quat(), self.dest, self.source)

    def __mul__(self, other):
        """
        Compose this rotation with another rotation.

        The destination frame of the right rotation (other) and the source
        frame of the left rotation (self) must be the same. I.E. if A and B are
        rotations, then for A*B to be valid, A.source must equal B.dest.

        Parameters
        ----------
        other : Rotation
                Another rotation object, it can be constant or time dependent.
        """
        if self.source != other.dest:
            raise ValueError("Destination frame of first rotation {} is not the same as source frame of second rotation {}.".format(other.dest, self.source))
        if isinstance(other, ConstantRotation):
            new_rot = self._rot * other._rot
            return ConstantRotation(new_rot.as_quat(), other.source, self.dest)
        elif isinstance(other, TimeDependentRotation):
            return TimeDependentRotation((self._rot * other._rots).as_quat(), other.times, other.source, self.dest)
        else:
            raise TypeError("Rotations can only be composed with other rotations.")

class TimeDependentRotation:
    """
    A time dependent rotation between two 3D reference frames.

    Attributes
    __________
    source : int
             The NAIF ID code for the source frame
    dest : int
           The NAIF ID code for the destination frame
    """

    def from_euler(sequence, euler, times, source, dest):
        """
        Create a time dependent rotation from a set of Euler angles.

        Parameters
        ----------
        sequence : string
                   The axis sequence that the Euler angles are applied in. I.E. 'XYZ'
                   or 'ZXZ'.
        euler : 2darray
                2D numpy array of the euler angle rotations in radians.
        times : array
                The time for each rotation in euler. This array must be sorted
                in ascending order.
        source : int
                 The NAIF ID code for the source frame
        dest : int
               The NAIF ID code for the destination frame

        See Also
        --------
        scipy.spatial.transform.Rotation.from_euler
        """
        rot = Rotation.from_euler(sequence, np.asarray(euler))
        return TimeDependentRotation(rot.as_quat(), times, source, dest)

    def __init__(self, quats, times, source, dest):
        """
        Construct a time dependent rotation

        Parameters
        ----------
        quats : 2darray
               The quaternion representations of the rotation as a 2d numpy array.
               Each inner array represents the rotation at the time at the same index
               in the times argument. The quaternions must be in scalar last format
               (x, y, z, w).
        times : array
                The time for each rotation in quats. This array must be sorted
                in ascending order.
        source : int
                 The NAIF ID code for the source frame
        dest : int
               The NAIF ID code for the destination frame
        """
        self.source = source
        self.dest = dest
        self.quats = np.asarray(quats)
        self.times = np.asarray(times)

    def __repr__(self):
        return f'Time Dependent Rotation Source: {self.source}, Destination: {self.dest}, Quat: {self.quats}'

    @property
    def quats(self):
        """
        The quaternions that rotates from the source reference frame to
        the destination reference frame. The quaternions are in scalar
        last format (x, y, z, w).
        """
        return self._rots.as_quat()

    @quats.setter
    def quats(self, new_quats):
        """
        Change the rotations to interpolate over

        Parameters
        ----------
        new_quats : 2darray
                    The new quaternions as a 2d array. The quaternions must be
                    in scalar last format (x, y, z, w).
        """
        self._rots = Rotation.from_quat(np.asarray(new_quats))

    def inverse(self):
        """
        Get the inverse rotation, that is the rotation from the destination
        reference frame to the source reference frame.
        """
        return TimeDependentRotation(self._rots.inv().as_quat(), self.times, self.dest, self.source)

    def __mul__(self, other):
        """
        Compose this rotation with another rotation.

        The destination frame of the right rotation (other) and the source
        frame of the left rotation (self) must be the same. I.E. if A and B are
        rotations, then for A*B to be valid, A.source must equal B.dest.

        If the other rotation is a time dependent rotation, then the time range
        for the resultant rotation will be the time covered by both rotations.
        I.E. if A covers 0 to 2 and B covers 1 to 4, then A*B will cover 1 to 2.

        Parameters
        ----------
        other : Rotation
                Another rotation object, it can be constant or time dependent.
        """
        if self.source != other.dest:
            raise ValueError("Destination frame of first rotation {} is not the same as source frame of second rotation {}.".format(other.dest, self.source))
        if isinstance(other, ConstantRotation):
            return TimeDependentRotation((self._rots * other._rot).as_quat(), self.times, other.source, self.dest)
        elif isinstance(other, TimeDependentRotation):
            # if self and other each have the same time and one rotation, don't interpolate.
            if (self.times.size == 1) and (other.times.size == 1) and (self.times == other.times):
                return TimeDependentRotation((self._rots * other._rots).as_quat(), self.times, other.source, self.dest)
            merged_times = np.union1d(np.asarray(self.times), np.asarray(other.times))
            # we cannot extrapolate so clip to the time range both cover
            first_time = max(min(self.times), min(other.times))
            last_time = min(max(self.times), max(other.times))
            new_times = merged_times[np.logical_and(merged_times>=first_time, merged_times<=last_time)]
            first_rotation_interp = Slerp(other.times, other._rots)
            second_rotation_interp = Slerp(self.times, self._rots)
            new_quats = (second_rotation_interp(new_times) * first_rotation_interp(new_times)).as_quat()
            return TimeDependentRotation(new_quats, new_times, other.source, self.dest)
        else:
            raise TypeError("Rotations can only be composed with other rotations.")
