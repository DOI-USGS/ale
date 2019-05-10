from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation

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

    def __init__(self, quat, source, dest):
        """
        Construct a constant rotation

        Parameters
        ----------
        quat : array
               The quaternion representation of the rotation as a numpy array
        source : int
                 The NAIF ID code for the source frame
        dest : int
               The NAIF ID code for the destination frame
        """
        self.source = source
        self.dest = dest
        self.quat = quat

    @property
    def quat(self):
        """
        The quaternion that rotates from the source reference frame to
        the destination reference frame.
        """
        return self._rot.as_quat()

    @quat.setter
    def quat(self, new_quat):
        """
        Change the rotation to a different quaternion

        Parameters
        ----------
        new_quat : array
                   The new quaternion as an array
        """
        self._rot = Rotation.from_quat(new_quat)

    def inverse(self):
        """
        Get the inverse rotation, that is the rotation from the destination
        reference frame to the source reference frame.
        """
        return ConstantRotation(self._rot.inv().as_quat(), self.dest, self.source)

    def __mul__(self, other):
        """
        Compose this rotation with another rotation. The destination frame of
        the right rotation (other) and the source frame of the left
        rotation (self) must be the same. I.E. if A and B are rotations, then
        for A*B to be valid, A.source must equal B.dest.

        Parameters
        ----------
        other : Rotation
                Another rotation object, it can be constant or time dependent.
        """
        if self.source != other.dest:
            raise ValueError("Destination frame of first rotation is not the same as source frame of second rotation.")
        if isinstance(other, ConstantRotation):
            new_rot = self._rot * other._rot
            return ConstantRotation(new_rot.as_quat(), other.source, self.dest)
        elif isinstance(other, TimeDependentRotation):
            new_quats = np.array([(self._rot * rot).as_quat() for rot in other._rots])
            return TimeDependentRotation(new_quats, other.times, other.source, self.dest)
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

    def __init__(self, quats, times, source, dest):
        """
        Construct a constant rotation

        Parameters
        ----------
        quats : 2darray
               The quaternion representations of the rotation as a 2d numpy array.
               Each inner array represents the rotation at the time at the same index
               in the times argument.
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
        self.quats = quats
        self.times = times

    @property
    def quats(self):
        """
        The quaternion that rotates from the source reference frame to
        the destination reference frame.
        """
        return np.array([rot.as_quat() for rot in self._rots])

    @quat.setter
    def quats(self, new_quats):
        """
        Change the rotations to interpolate over

        Parameters
        ----------
        new_quats : 2darray
                    The new quaternions as a 2d array.
        """
        self._rot = [Rotation.from_quat(quat) for quat in new_quats]

    def inverse(self):
        """
        Get the inverse rotation, that is the rotation from the destination
        reference frame to the source reference frame.
        """
        new_quats = np.array([rot.inv().as_quat() for rot in self._rots])
        return TimeDependentRotation(new_quats, self.dest, self.source)

    def __mul__(self, other):
        """
        Compose this rotation with another rotation. The destination frame of
        the right rotation (other) and the source frame of the left
        rotation (self) must be the same. I.E. if A and B are rotations, then
        for A*B to be valid, A.source must equal B.dest.

        Parameters
        ----------
        other : Rotation
                Another rotation object, it can be constant or time dependent.
        """
        if self.source != other.dest:
            raise ValueError("Destination frame of first rotation is not the same as source frame of second rotation.")
        if isinstance(other, ConstantRotation):
            new_quats = np.array([(rot * other._rot).as_quat() for rot in self._rots])
            return TimeDependentRotation(new_quats, self.times, other.source, self.dest)
        elif isinstance(other, TimeDependentRotation):
            new_times = np.union1d(np.asarray(self.times), np.asarray(other.times))
            first_rotation_interp = Slerp(other.times, other._rots)
            second_rotation_interp = Slerp(self.times, self._rots)
            new_quats = []
            for time in new_times:
                new_quats.append( (second_rotation_interp(time) * first_rotation_interp(time)).as_quat() )
            return TimeDependentRotation(np.array(new_quats), new_times, other.source, self.dest)
        else:
            raise TypeError("Rotations can only be composed with other rotations.")
