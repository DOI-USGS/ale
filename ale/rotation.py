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
        rot = Rotation.from_matrix(mat)
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
        return self._rot.as_matrix()

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
            return TimeDependentRotation((self._rot * other._rots).as_quat(), other.times, other.source, self.dest, av=other.av)
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

    def from_euler(sequence, euler, times, source, dest, degrees=False):
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
        degrees : bool
                  If the angles are in degrees. If false, then degrees are
                  assumed to be in radians. Defaults to False.

        See Also
        --------
        scipy.spatial.transform.Rotation.from_euler
        """
        rot = Rotation.from_euler(sequence, np.asarray(euler), degrees=degrees)
        return TimeDependentRotation(rot.as_quat(), times, source, dest)

    def __init__(self, quats, times, source, dest, av=None):
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
        av : 2darray
             The angular velocity of the rotation at each time as a 2d numpy array.
             If not entered, then angular velocity will be computed by assuming constant
             angular velocity between times.
        """
        self.source = source
        self.dest = dest
        self.quats = quats
        self.times = np.atleast_1d(times)
        if av is not None:
            self.av = np.asarray(av)
        else:
            self.av = av

    def __repr__(self):
        return f'Time Dependent Rotation Source: {self.source}, Destination: {self.dest}, Quats: {self.quats}, AV: {self.av}, Times: {self.times}'

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
        self._rots = Rotation.from_quat(new_quats)

    def inverse(self):
        """
        Get the inverse rotation, that is the rotation from the destination
        reference frame to the source reference frame.
        """
        if self.av is not None:
            new_av = -self._rots.apply(self.av)
        else:
            new_av = None
        return TimeDependentRotation(self._rots.inv().as_quat(), self.times, self.dest, self.source, av=new_av)

    def _slerp(self, times):
        """
        Using SLERP interpolate the rotation and angular velocity at
        specific times.

        Times outside of the range covered by this rotation are extrapolated
        assuming constant angular velocity. If the rotation has angular velocities
        stored, then the first and last angular velocity are used for extrapolation.
        Otherwise, the angular velocities from the first and last interpolation
        interval are used for extrapolation.

        Parameters
        ----------
        times : 1darray or float
                The new times to interpolate at.

        Returns
        -------
         : Rotation
           The new rotations at the input times
         : 2darray
           The angular velocity vectors
        """
        # Convert non-vector input to vector and check input
        vec_times = np.atleast_1d(times)
        if vec_times.ndim > 1:
            raise ValueError('Input times must be either a float or a 1d iterable of floats')

        # Compute constant angular velocity for interpolation intervals
        avs = np.zeros((len(self.times) + 1, 3))
        if len(self.times) > 1:
            steps = self.times[1:] - self.times[:-1]
            rotvecs = (self._rots[1:] * self._rots[:-1].inv()).as_rotvec()
            avs[1:-1] = rotvecs / steps[:, None]

        # If available use actual angular velocity for extrapolation
        # Otherwise use the adjacent interpolation interval
        if self.av is not None:
            avs[0] = self.av[0]
            avs[-1] = self.av[-1]
        else:
            avs[0] = avs[1]
            avs[-1] = avs[-2]

        # Determine interpolation intervals for input times
        av_idx = np.searchsorted(self.times, vec_times)
        rot_idx = av_idx
        rot_idx[rot_idx > len(self.times) - 1] = len(self.times) - 1

        # Interpolate/extrapolate rotations
        time_diffs = vec_times - self.times[rot_idx]
        interp_av = avs[av_idx]
        interp_rots = Rotation.from_rotvec(interp_av * time_diffs[:, None]) * self._rots[rot_idx]

        # If actual angular velocities are available, linearly interpolate them
        if self.av is not None:
            av_diff = np.zeros((len(self.times), 3))
            if len(self.times) > 1:
                av_diff[:-1] = self.av[1:] - self.av[:-1]
                av_diff[-1] = av_diff[-2]
            interp_av = self.av[rot_idx] + (av_diff[rot_idx] * time_diffs[:, None])

        return interp_rots, interp_av


    def reinterpolate(self, times):
        """
        Reinterpolate the rotation at a given set of times.

        Parameters
        ----------
        times : 1darray or float
                The new times to interpolate at.

        Returns
        -------
         : TimeDependentRotation
           The new rotation at the input times
        """
        new_rots, av = self._slerp(times)
        return TimeDependentRotation(new_rots.as_quat(), times, self.source, self.dest, av=av)

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
            if self.av is not None:
                other_inverse = other._rot.inv()
                new_av = np.asarray([other_inverse.apply(av) for av in self.av])
            else:
                new_av = None
            return TimeDependentRotation((self._rots * other._rot).as_quat(), self.times, other.source, self.dest, av=new_av)
        elif isinstance(other, TimeDependentRotation):
            merged_times = np.union1d(np.asarray(self.times), np.asarray(other.times))
            reinterp_self = self.reinterpolate(merged_times)
            reinterp_other = other.reinterpolate(merged_times)
            new_quats = (reinterp_self._rots * reinterp_other._rots).as_quat()
            new_av = reinterp_other._rots.inv().apply(reinterp_self.av) + reinterp_other.av
            return TimeDependentRotation(new_quats, merged_times, other.source, self.dest, av=new_av)
        else:
            raise TypeError("Rotations can only be composed with other rotations.")

    def apply_at(self, vec, et):
        """
        Apply the rotation to a position at a specific time
        """
        return self.reinterpolate(et)._rots.apply(vec)

    def rotate_velocity_at(self, pos, vel, et):
        """
        Apply the rotation to a velocity at a specific time

        See:
        https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/rotation.html#State%20transformations
        For an explanation of why a separate method is required to rotate velocities.
        """
        vec_pos = np.asarray(pos)
        vec_vel = np.asarray(vel)
        if vec_pos.ndim < 1:
            vec_pos = np.asarray([pos])
        if vec_vel.ndim < 1:
            vec_vel = np.asarray([vel])
        if vec_pos.shape != vec_vel.shape:
            raise ValueError('Input velocities and positions must have the same shape')

        rots, avs = self._slerp(et)
        rotated_vel = np.zeros(vec_vel.shape)
        for indx in range(vec_pos.shape[0]):
            skew = np.array([[0, -avs[indx, 2], avs[indx, 1]],
                             [avs[indx, 2], 0, -avs[indx, 0]],
                             [-avs[indx, 1], avs[indx, 0], 0]])
            rot_deriv = np.dot(skew, rots[indx].as_matrix().T).T
            rotated_vel[indx] = rots[indx].apply(vec_vel[indx])
            rotated_vel[indx] += np.dot(rot_deriv, vec_pos[indx])

        return rotated_vel
