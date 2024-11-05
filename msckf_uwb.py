import numpy as np
from scipy.stats import chi2

from utils import *
from feature import Feature
import pyceres
from scipy.optimize import least_squares
import time
from collections import namedtuple


class UWBWindow(object):
    def __init__(self, anchor_num):
        self.window_size = 800
        self.anchor_num = anchor_num
        self.imu_orientation = []
        self.imu_position = []
        self.imu_cov = []
        self.timestamp = []
        self.distance = []


class IMUState(object):
    # id for next IMU state
    next_id = 0

    # Gravity vector in the world frame
    gravity = np.array([0., 0., -9.81])

    # Transformation offset from the IMU frame to the body frame.
    # The transformation takes a vector from the IMU frame to the
    # body frame. The z axis of the body frame should point upwards.
    # Normally, this transform should be identity.
    T_imu_body = Isometry3d(np.identity(3), np.zeros(3))

    def __init__(self, new_id=None):
        # An unique identifier for the IMU state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the IMU (body) frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the IMU (body) frame in the world frame.
        self.position = np.zeros(3)
        # Velocity of the IMU (body) frame in the world frame.
        self.velocity = np.zeros(3)

        # Bias for measured angular velocity and acceleration.
        self.gyro_bias = np.zeros(3)
        self.acc_bias = np.zeros(3)

        # These three variables should have the same physical
        # interpretation with `orientation`, `position`, and
        # `velocity`. There three variables are used to modify
        # the transition matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)
        self.velocity_null = np.zeros(3)

        # Transformation between the IMU and the left camera (cam0)
        self.R_imu_cam0 = np.identity(3)
        self.t_cam0_imu = np.zeros(3)

        # Transformation between the IMU and UWB tag
        self.R_imu_uwb = np.identity(3)
        self.t_uwb_imu = np.zeros(3)


class UWBState(object):
    def __init__(self):
        self.uwb_positions = np.array([
            [-6.0, -6.0, 0.5],
            [6.0, -5.0, 0.5],
            [-6.0, 9.0, 3.0],
            [6.0, 9.0, 3.0]
        ])


class CAMState(object):
    # Takes a vector from the cam0 frame to the cam1 frame.
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=None):
        # An unique identifier for the CAM state.
        self.id = new_id
        # Time when the state is recorded
        self.timestamp = None

        # Orientation
        # Take a vector from the world frame to the camera frame.
        self.orientation = np.array([0., 0., 0., 1.])

        # Position of the camera frame in the world frame.
        self.position = np.zeros(3)

        # These two variables should have the same physical
        # interpretation with `orientation` and `position`.
        # There two variables are used to modify the measurement
        # Jacobian matrices to make the observability matrix
        # have proper null space.
        self.orientation_null = np.array([0., 0., 0., 1.])
        self.position_null = np.zeros(3)


class StateServer(object):
    """
    Store one IMU states and several camera states for constructing
    measurement model.
    """

    def __init__(self):
        self.imu_state = IMUState()
        self.uwb_state = UWBState()
        self.cam_states = dict()  # <CAMStateID, CAMState>, ordered dict

        # State covariance matrix
        self.state_cov = np.zeros((33, 33))
        self.continuous_noise_cov = np.zeros((12, 12))


class UWBOptimization(object):
    def __init__(self, T_pose, state_cov, distance, init_guess):
        self.T_pose = np.array(T_pose)
        self.distance = np.array(distance)
        self.T_cov = state_cov
        self.init_guess = init_guess
        self.uwb_init = self.optimize_uwb_position()

    def residuals(self, uwb_position):
        residuals = []
        for T, dist in zip(self.T_pose, self.distance):
            predicted_dist = np.linalg.norm(T - uwb_position)
            residuals.append(predicted_dist - dist)
        return np.array(residuals)

    def optimize_uwb_position(self):
        result = least_squares(self.residuals, self.init_guess, method='lm')
        return result.x


class MSCKF(object):
    def __init__(self, config):
        self.config = config
        self.optimization_config = config.optimization_config

        # IMU data buffer
        # This is buffer is used to handle the unsynchronization or
        # transfer delay between IMU and Image messages.
        self.imu_msg_buffer = []
        self.uwb_msg_buffer = []

        # State vector
        self.state_server = StateServer()
        # Features used
        self.map_server = dict()  # <FeatureID, Feature>

        # Chi squared test table.
        # Initialize the chi squared test table with confidence level 0.95.
        self.chi_squared_test_table = dict()
        for i in range(1, 100):
            self.chi_squared_test_table[i] = chi2.ppf(0.05, i)

        # Set the initial IMU state.
        # The intial orientation and position will be set to the origin implicitly.
        # But the initial velocity and bias can be set by parameters.
        # TODO: is it reasonable to set the initial bias to 0?
        self.state_server.imu_state.velocity = config.velocity
        self.reset_state_cov()

        continuous_noise_cov = np.identity(12)
        continuous_noise_cov[:3, :3] *= self.config.gyro_noise
        continuous_noise_cov[3:6, 3:6] *= self.config.gyro_bias_noise
        continuous_noise_cov[6:9, 6:9] *= self.config.acc_noise
        continuous_noise_cov[9:, 9:] *= self.config.acc_bias_noise
        self.state_server.continuous_noise_cov = continuous_noise_cov

        # set uwb noise
        self.t_uwb_imu = self.config.t_uwb_imu

        # self.uwb_init_guess = np.array([[-6.0, -6.0, 0.5],
        #                            [6.0, -5.0, 0.5], [-6.0, 9.0, 3.0], [6.0, 9.0, 3.0]])

        self.uwb_init_guess = np.array([[-9.0, -5.0, 0.5],
                                        [7.0, -5.0, 0.5], [-9.0, 12.0, 3.0], [7.0, 11.0, 3.0]])

        # Gravity vector in the world frame
        IMUState.gravity = config.gravity

        # Transformation between the IMU and the left camera (cam0)
        T_cam0_imu = np.linalg.inv(config.T_imu_cam0)
        self.state_server.imu_state.R_imu_cam0 = T_cam0_imu[:3, :3].T
        self.state_server.imu_state.t_cam0_imu = T_cam0_imu[:3, 3]

        # Extrinsic parameters of camera and IMU.
        T_cam0_cam1 = config.T_cn_cnm1
        CAMState.R_cam0_cam1 = T_cam0_cam1[:3, :3]
        CAMState.t_cam0_cam1 = T_cam0_cam1[:3, 3]
        Feature.R_cam0_cam1 = CAMState.R_cam0_cam1
        Feature.t_cam0_cam1 = CAMState.t_cam0_cam1
        IMUState.T_imu_body = Isometry3d(
            config.T_imu_body[:3, :3],
            config.T_imu_body[:3, 3])

        # Tracking rate.
        self.tracking_rate = None

        # Indicate if the gravity vector is set.
        self.is_gravity_set = False
        # Indicate if the received image is the first one. The system will
        # start after receiving the first image.
        self.is_first_img = True

        # uwb anchor initialization
        self.anchor_num = 4

        self.is_uwb_init = False
        self.uwb_window = UWBWindow(self.anchor_num)

    def imu_callback(self, imu_msg):
        """
        Callback function for the imu message.
        """
        # IMU msgs are pushed backed into a buffer instead of being processed
        # immediately. The IMU msgs are processed when the next image is
        # available, in which way, we can easily handle the transfer delay.
        self.imu_msg_buffer.append(imu_msg)

        if not self.is_gravity_set:
            if len(self.imu_msg_buffer) >= 200:
                self.initialize_gravity_and_bias()
                self.is_gravity_set = True

    def uwb_callback(self, uwb_msg):
        self.uwb_msg_buffer.append(uwb_msg)

    def feature_callback(self, feature_msg):
        """
        Callback function for feature measurements.
        """
        if not self.is_gravity_set:
            return
        start = time.time()

        # Start the system if the first image is received.
        # The frame where the first image is received will be the origin.
        if self.is_first_img:
            self.is_first_img = False
            self.state_server.imu_state.timestamp = feature_msg.timestamp

        t = time.time()

        if not self.is_uwb_init:
            # Append window size
            self.uwb_window.imu_position.append(self.state_server.imu_state.position)
            self.uwb_window.imu_orientation.append(self.state_server.imu_state.orientation)
            self.uwb_window.imu_cov.append(self.state_server.state_cov[:21, :21])
            self.uwb_window.timestamp.append(self.state_server.imu_state.timestamp)

            # Initialize uwb
            if len(self.uwb_msg_buffer) >= self.uwb_window.window_size:
                self.initialize_uwb(feature_msg.timestamp, self.uwb_init_guess)
                self.is_uwb_init = True

            # Propogate the IMU state.
            # that are received before the image msg.
            self.batch_imu_processing(feature_msg.timestamp)
        else:
            self.batch_imu_uwb_processing(feature_msg.timestamp)


        print('---batch_imu_processing    ', time.time() - t)
        t = time.time()

        # Augment the state vector.
        self.state_augmentation(feature_msg.timestamp)

        print('---state_augmentation      ', time.time() - t)
        t = time.time()

        # Add new observations for existing features or new features
        # in the map server.
        self.add_feature_observations(feature_msg)

        print('---add_feature_observations', time.time() - t)
        t = time.time()

        # Perform measurement update if necessary.
        # And prune features and camera states.
        self.remove_lost_features()

        print('---remove_lost_features    ', time.time() - t)
        t = time.time()

        self.prune_cam_state_buffer()

        print('---prune_cam_state_buffer  ', time.time() - t)
        print('---msckf elapsed:          ', time.time() - start, f'({feature_msg.timestamp})')

        try:
            # Publish the odometry.
            return self.publish(feature_msg.timestamp)
        finally:
            # Reset the system if necessary.
            self.online_reset()

        # UWB anchor initialization

    def initialize_uwb(self, time_bound, uwb_init_value):

        # used uwb_msg_count for initialization
        used_uwb_msg_count = 0
        used_uwb_distance = []
        used_uwb_timestamp = []
        for uwb_msg in self.uwb_msg_buffer:
            if uwb_msg.timestamp > time_bound:
                break
            used_uwb_msg_count += 1
            used_uwb_timestamp.append(uwb_msg.timestamp)
            used_uwb_distance.append(uwb_msg.distance)

        # filter the uwb_msg
        assert (used_uwb_msg_count != 0)
        matched_uwb_timestamps, matched_uwb_distances = self.filter_uwb_timestamps(
            self.uwb_window.timestamp, used_uwb_timestamp, used_uwb_distance, 5e-2)

        self.uwb_msg_buffer = self.uwb_msg_buffer[used_uwb_msg_count:]

        # Solve the anchor position
        uwb_init_guess = uwb_init_value
        uwb_init = []

        for i, init_guess in enumerate(uwb_init_guess):
            uwb_init.append(UWBOptimization(self.uwb_window.imu_position, self.uwb_window.imu_cov,
                                            matched_uwb_distances[i], init_guess).uwb_init)
        self.state_server.uwb_state.uwb_positions = np.array(uwb_init)

        # Position initialization error
        # self.uwb_gt = np.array(
        #     [[-5, -5, 0], [5, -4, 0], [-5, 10, 2.5], [5, 10, 2.5]])
        self.uwb_gt = np.array(
            [[-10, -5, 0], [7, -4, 0], [-10, 12, 2.5], [7, 12, 2.5]])
        err_init = np.linalg.norm(self.uwb_gt - uwb_init)

        self.state_server.uwb_state.uwb_positions = self.uwb_gt * 1.01

        # state augmentation
        if not self.is_uwb_init:
            old_size = self.state_server.state_cov.shape[0]
        else:
            old_size = self.state_server.state_cov.shape[0] - 12

        state_cov = self.uwb_state_augmentation(old_size)

        # fix the covariance to be symmetric
        self.state_server.state_cov = (state_cov + state_cov.T) / 2.

        # clear the uwb window
        self.uwb_window.imu_orientation = []
        self.uwb_window.imu_position = []
        self.uwb_window.imu_cov = []
        self.uwb_window.timestamp = []
        self.uwb_window.distance = []

    def uwb_state_augmentation(self, old_size):
        state_cov = np.zeros((old_size + 12, old_size + 12))
        state_cov[:old_size, :old_size] = self.state_server.state_cov

        uwb_cov = self.config.uwb_noise ** 2

        for uwb_id, anchor_pos in enumerate(self.state_server.uwb_state.uwb_positions):
            H_I_stack = []
            P_aa_1 = np.zeros((3, 3))
            P_aa_2 = np.zeros((3, 21))
            P_aa_3 = np.zeros((21, 21))
            P_Ia = np.zeros((3, 21))
            H_a_stack = []
            for imu_position, imu_orientation, imu_cov in zip(
                    self.uwb_window.imu_position, self.uwb_window.imu_orientation, self.uwb_window.imu_cov):
                R_imu = to_rotation(imu_orientation)
                tag_position = imu_position + R_imu.T @ self.t_uwb_imu
                H_I = np.zeros((21,))
                H_a = np.zeros((3,))
                H_p = (tag_position - anchor_pos) / np.linalg.norm(tag_position - anchor_pos)
                H_r = H_p @ (R_imu.T @ skew(self.t_uwb_imu))
                H_u = H_p @ R_imu.T
                H_I[:3] = H_r
                H_I[12:15] = H_p
                H_a = -H_p

                H_I_stack.append(H_I)
                H_a_stack.append(H_a)

                # reshape the matrix
                H_I = np.array([H_I])
                H_a = np.array([H_a])


            H_I_stack = np.array(H_I_stack)
            H_a_stack = np.array(H_a_stack)
            Q_A, H_a_1 = np.linalg.qr(H_a_stack)
            H_I_1 = Q_A.T @ H_I_stack
            H_I_1 = H_I_1[:3,]

            P_aa = np.linalg.inv(H_a_1) @ (H_I_1 @ imu_cov @ H_I_1.T + uwb_cov * np.eye(3)) @ np.linalg.inv(H_a_1.T)
            P_Ia = -imu_cov @ H_I_1.T @ np.linalg.inv(H_a_1.T)
            state_cov[old_size + (uwb_id) * 3:old_size + (uwb_id + 1) * 3,
            old_size + (uwb_id) * 3:old_size + (uwb_id + 1) * 3] = P_aa

            state_cov[:21, old_size + (uwb_id) * 3:old_size + (uwb_id + 1) * 3] = P_Ia
            state_cov[old_size + (uwb_id) * 3:old_size + (uwb_id + 1) * 3, :21] = P_Ia.T

            T =  is_pos_matrix(P_aa)

        return state_cov

    def filter_uwb_timestamps(self, window_timestamps, uwb_timestamps, uwb_distances, threshold):
        # Store the UWB timestamps that match the criteria
        matched_uwb_timestamps = []
        matched_uwb_distance = []

        # Iterate over all UWB timestamps
        for uwb_time, uwb_dist in zip(uwb_timestamps, uwb_distances):
            # Find the robot timestamp that is closest to the current UWB timestamp
            closest_robot_time = min(window_timestamps, key=lambda x: abs(x - uwb_time))
            # Check if the closest robot timestamp is within the allowed threshold
            if abs(closest_robot_time - uwb_time) <= threshold:
                matched_uwb_timestamps.append(uwb_time)
                matched_uwb_distance.append(uwb_dist)

        matched_uwb_distance = np.array(matched_uwb_distance).T
        return matched_uwb_timestamps, matched_uwb_distance

    def initialize_gravity_and_bias(self):
        """
        Initialize the IMU bias and initial orientation based on the
        first few IMU readings.
        """
        sum_angular_vel = np.zeros(3)
        sum_linear_acc = np.zeros(3)
        for msg in self.imu_msg_buffer:
            sum_angular_vel += msg.angular_velocity
            sum_linear_acc += msg.linear_acceleration

        gyro_bias = sum_angular_vel / len(self.imu_msg_buffer)
        self.state_server.imu_state.gyro_bias = gyro_bias

        # This is the gravity in the IMU frame.
        gravity_imu = sum_linear_acc / len(self.imu_msg_buffer)

        # Initialize the initial orientation, so that the estimation
        # is consistent with the inertial frame.
        gravity_norm = np.linalg.norm(gravity_imu)
        IMUState.gravity = np.array([0., 0., -gravity_norm])

        self.state_server.imu_state.orientation = from_two_vectors(
            -IMUState.gravity, gravity_imu)

    # Filter related functions
    # (batch_imu_processing, process_model, predict_new_state)
    def batch_imu_processing(self, time_bound):
        """
        Propogate the state
        """
        used_imu_msg_count = 0
        for msg in self.imu_msg_buffer:
            imu_time = msg.timestamp
            if imu_time < self.state_server.imu_state.timestamp:
                used_imu_msg_count += 1
                continue
            if imu_time > time_bound:
                break

            # Execute process model.
            self.process_model(
                imu_time, msg.angular_velocity, msg.linear_acceleration)
            used_imu_msg_count += 1

            # Update the state info
            self.state_server.imu_state.timestamp = imu_time

        self.state_server.imu_state.id = IMUState.next_id
        IMUState.next_id += 1

        # Remove all used IMU msgs.
        self.imu_msg_buffer = self.imu_msg_buffer[used_imu_msg_count:]

    def batch_imu_uwb_processing(self, time_bound):
        """
        Propogate the state
        """
        used_imu_msg_count = 0
        used_uwb_msg_count = 0

        for uwb_msg in self.uwb_msg_buffer:
            uwb_time = uwb_msg.timestamp
            if uwb_time < time_bound:
                used_uwb_msg_count += 1
            else:
                break

            # IMU propagation and update within current uwb window
            for imu_msg in self.imu_msg_buffer:
                imu_time = imu_msg.timestamp
                if imu_time < self.state_server.imu_state.timestamp:
                    used_imu_msg_count += 1
                    continue

                if imu_time > uwb_time:
                    # Update with uwb
                    self.uwb_upate(uwb_msg, Mode = 'EKF')

                    # Remove all used UWB msgs.
                    self.uwb_msg_buffer = self.uwb_msg_buffer[used_uwb_msg_count:]
                    break

                # IMU propagation
                self.process_model(imu_time, imu_msg.angular_velocity, imu_msg.linear_acceleration)
                # Update the state timestamp
                self.state_server.imu_state.timestamp = imu_time
                used_imu_msg_count += 1

            self.state_server.imu_state.id = IMUState.next_id
            IMUState.next_id += 1

            # Remove all used IMU msgs.
            self.imu_msg_buffer = self.imu_msg_buffer[used_imu_msg_count:]

    def uwb_upate(self, uwb_msg, Mode = 'EKF'):

        # Update
        self.state_server.imu_state.timestamp = uwb_msg.timestamp

        # UWB tag's position
        imu_state = self.state_server.imu_state
        state_cov = self.state_server.state_cov
        R_imu = to_rotation(self.state_server.imu_state.orientation)
        tag_position = imu_state.position + R_imu.T @ self.t_uwb_imu
        uwb_cov = self.config.uwb_noise ** 2

        # UWB anchor position
        uwb_position = self.state_server.uwb_state.uwb_positions
        state_size = self.state_server.state_cov.shape[0]

        # Compute measurement jacobian
        H_stack = []
        Y = []
        residual = []
        R = []

        for uwb_id, distance in enumerate(uwb_msg.distance):
            H = np.zeros((state_size,))
            H_p = (tag_position - uwb_position[uwb_id]) / np.linalg.norm(tag_position - uwb_position[uwb_id])
            H_r = H_p @ (R_imu.T @ skew(self.t_uwb_imu))
            H_u = H_p @ R_imu.T
            H[:3] = H_r
            H[12:15] = H_p
            H[-12 + uwb_id * 3:-9 + uwb_id * 3 or None] = -H_p
            residual.append(distance - np.linalg.norm(tag_position - uwb_position[uwb_id]))

            R.append(uwb_cov)
            H_stack.append(H)

        # Update
        H_stack = np.vstack(H_stack)
        R = np.diag(R)
        S = H_stack @ state_cov @ H_stack.T + R
        K_gain = state_cov @ H_stack.T @ np.linalg.inv(S)
        X_corr = K_gain @ residual

        # State update
        X_corr_imu = X_corr[:21]
        X_corr_uwb = X_corr[-12:]

        if (np.linalg.norm(X_corr_imu[6:9]) > 0.5 or
                np.linalg.norm(X_corr_imu[12:15]) > 1.0):
            print('[Warning] Update change is too large')

        if Mode == 'EKF':
            # Covariance update
            state_cov = (np.eye(state_size) - K_gain @ H_stack) @ state_cov

            # update IMU state
            dq_imu = small_angle_quaternion(X_corr_imu[:3])
            imu_state.orientation = quaternion_multiplication(
                dq_imu, imu_state.orientation)
            imu_state.gyro_bias += X_corr_imu[3:6]
            imu_state.velocity += X_corr_imu[6:9]
            imu_state.acc_bias += X_corr_imu[9:12]
            imu_state.position += X_corr_imu[12:15]

            dq_extrinsic = small_angle_quaternion(X_corr_imu[15:18])
            imu_state.R_imu_cam0 = to_rotation(dq_extrinsic) @ imu_state.R_imu_cam0
            imu_state.t_cam0_imu += X_corr_imu[18:21]

            # update the UWB state
            for i in range(len(uwb_position)):
                uwb_position[i] += X_corr_uwb[i * 3:i * 3 + 3]
            self.state_server.uwb_state.uwb_positions = uwb_position

            # Update the camera states.
            for i, (cam_id, cam_state) in enumerate(
                    self.state_server.cam_states.items()):
                delta_x_cam = X_corr[21 + i * 6:27 + i * 6]
                dq_cam = small_angle_quaternion(delta_x_cam[:3])
                cam_state.orientation = quaternion_multiplication(
                    dq_cam, cam_state.orientation)
                cam_state.position += delta_x_cam[3:]

            self.state_server.state_cov = (state_cov + state_cov.T) / 2.

        elif Mode == 'SKF':
            # Covariance update
            state_cov_full = (np.eye(state_size) - K_gain @ H_stack) @ state_cov
            state_cov[-12:,-12:] = state_cov_full[-12:,-12:]
            # state_cov[:-12,-12:] = state_cov_full[:-12,-12:]
            # state_cov[-12:, :-12] = state_cov_full[-12:, :-12]
            self.state_server.state_cov = (state_cov + state_cov.T) / 2.
            # update the UWB state
            for i in range(len(uwb_position)):
                uwb_position[i] += X_corr_uwb[i * 3:i * 3 + 3]
            self.state_server.uwb_state.uwb_positions = uwb_position

            self.state_server.state_cov = (state_cov + state_cov.T) / 2.


    def process_model(self, time, m_gyro, m_acc):
        imu_state = self.state_server.imu_state
        dt = time - imu_state.timestamp

        gyro = m_gyro - imu_state.gyro_bias
        acc = m_acc - imu_state.acc_bias

        # Compute discrete transition and noise covariance matrix
        F = np.zeros((21, 21))
        G = np.zeros((21, 12))

        R_w_i = to_rotation(imu_state.orientation)

        F[:3, :3] = -skew(gyro)
        F[:3, 3:6] = -np.identity(3)
        F[6:9, :3] = -R_w_i.T @ skew(acc)
        F[6:9, 9:12] = -R_w_i.T
        F[12:15, 6:9] = np.identity(3)

        G[:3, :3] = -np.identity(3)
        G[3:6, 3:6] = np.identity(3)
        G[6:9, 6:9] = -R_w_i.T
        G[9:12, 9:12] = np.identity(3)

        # Approximate matrix exponential to the 3rd order, which can be
        # considered to be accurate enough assuming dt is within 0.01s.
        Fdt = F * dt
        Fdt_square = Fdt @ Fdt
        Fdt_cube = Fdt_square @ Fdt
        Phi = np.identity(21) + Fdt + Fdt_square / 2. + Fdt_cube / 6.

        # Propogate the state using 4th order Runge-Kutta
        self.predict_new_state(dt, gyro, acc)

        # Modify the transition matrix
        R_kk_1 = to_rotation(imu_state.orientation_null)
        Phi[:3, :3] = to_rotation(imu_state.orientation) @ R_kk_1.T

        u = R_kk_1 @ IMUState.gravity
        # s = (u.T @ u).inverse() @ u.T
        # s = np.linalg.inv(u[:, None] * u) @ u
        s = u / (u @ u)

        A1 = Phi[6:9, :3]
        w1 = skew(imu_state.velocity_null - imu_state.velocity) @ IMUState.gravity
        Phi[6:9, :3] = A1 - (A1 @ u - w1)[:, None] * s

        A2 = Phi[12:15, :3]
        w2 = skew(dt * imu_state.velocity_null + imu_state.position_null -
                  imu_state.position) @ IMUState.gravity
        Phi[12:15, :3] = A2 - (A2 @ u - w2)[:, None] * s

        # Propogate the state covariance matrix.
        Q = Phi @ G @ (self.state_server.continuous_noise_cov) @ G.T @ Phi.T * dt
        self.state_server.state_cov[:21, :21] = (
                Phi @ self.state_server.state_cov[:21, :21] @ Phi.T + Q)

        # Cross-covariance propagation
        if len(self.state_server.cam_states) > 0:
            self.state_server.state_cov[:21, 21:] = (
                    Phi @ self.state_server.state_cov[:21, 21:])
            self.state_server.state_cov[21:, :21] = (
                    self.state_server.state_cov[21:, :21] @ Phi.T)

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (self.state_server.state_cov + self.state_server.state_cov.T) / 2.

        # Update the state correspondes to null space.
        self.state_server.imu_state.orientation_null = imu_state.orientation
        self.state_server.imu_state.position_null = imu_state.position
        self.state_server.imu_state.velocity_null = imu_state.velocity

    def predict_new_state(self, dt, gyro, acc):
        # TODO: Will performing the forward integration using
        # the inverse of the quaternion give better accuracy?
        gyro_norm = np.linalg.norm(gyro)
        Omega = np.zeros((4, 4))
        Omega[:3, :3] = -skew(gyro)
        Omega[:3, 3] = gyro
        Omega[3, :3] = -gyro

        q = self.state_server.imu_state.orientation
        v = self.state_server.imu_state.velocity
        p = self.state_server.imu_state.position

        if gyro_norm > 1e-5:
            dq_dt = (np.cos(gyro_norm * dt * 0.5) * np.identity(4) +
                     np.sin(gyro_norm * dt * 0.5) / gyro_norm * Omega) @ q
            dq_dt2 = (np.cos(gyro_norm * dt * 0.25) * np.identity(4) +
                      np.sin(gyro_norm * dt * 0.25) / gyro_norm * Omega) @ q
        else:
            dq_dt = np.cos(gyro_norm * dt * 0.5) * (np.identity(4) +
                                                    Omega * dt * 0.5) @ q
            dq_dt2 = np.cos(gyro_norm * dt * 0.25) * (np.identity(4) +
                                                      Omega * dt * 0.25) @ q

        dR_dt_transpose = to_rotation(dq_dt).T
        dR_dt2_transpose = to_rotation(dq_dt2).T

        # k1 = f(tn, yn)
        k1_p_dot = v
        k1_v_dot = to_rotation(q).T @ acc + IMUState.gravity

        # k2 = f(tn+dt/2, yn+k1*dt/2)
        k1_v = v + k1_v_dot * dt / 2.
        k2_p_dot = k1_v
        k2_v_dot = dR_dt2_transpose @ acc + IMUState.gravity

        # k3 = f(tn+dt/2, yn+k2*dt/2)
        k2_v = v + k2_v_dot * dt / 2
        k3_p_dot = k2_v
        k3_v_dot = dR_dt2_transpose @ acc + IMUState.gravity

        # k4 = f(tn+dt, yn+k3*dt)
        k3_v = v + k3_v_dot * dt
        k4_p_dot = k3_v
        k4_v_dot = dR_dt_transpose @ acc + IMUState.gravity

        # yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
        q = dq_dt / np.linalg.norm(dq_dt)
        v = v + (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot) * dt / 6.
        p = p + (k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot) * dt / 6.

        self.state_server.imu_state.orientation = q
        self.state_server.imu_state.velocity = v
        self.state_server.imu_state.position = p

    # Measurement update
    # (state_augmentation, add_feature_observations)
    def state_augmentation(self, time):
        imu_state = self.state_server.imu_state
        R_i_c = imu_state.R_imu_cam0
        t_c_i = imu_state.t_cam0_imu

        # Add a new camera state to the state server.
        R_w_i = to_rotation(imu_state.orientation)
        R_w_c = R_i_c @ R_w_i
        t_c_w = imu_state.position + R_w_i.T @ t_c_i

        cam_state = CAMState(imu_state.id)
        cam_state.timestamp = time
        cam_state.orientation = to_quaternion(R_w_c)
        cam_state.position = t_c_w
        cam_state.orientation_null = cam_state.orientation
        cam_state.position_null = cam_state.position
        self.state_server.cam_states[imu_state.id] = cam_state

        # Update the covariance matrix of the state.
        # To simplify computation, the matrix J below is the nontrivial block
        # in Equation (16) of "MSCKF" paper.
        J = np.zeros((6, 21))
        J[:3, :3] = R_i_c
        J[:3, 15:18] = np.identity(3)
        J[3:6, :3] = skew(R_w_i.T @ t_c_i)
        J[3:6, 12:15] = np.identity(3)
        J[3:6, 18:21] = np.identity(3)

        # Resize the state covariance matrix.
        # old_rows, old_cols = self.state_server.state_cov.shape
        old_size = self.state_server.state_cov.shape[0]  # symmetric
        state_cov = np.zeros((old_size + 6, old_size + 6))
        state_cov[:old_size, :old_size] = self.state_server.state_cov

        # Fill in the augmented state covariance.
        state_cov[old_size:, :old_size] = J @ state_cov[:21, :old_size]
        state_cov[:old_size, old_size:] = state_cov[old_size:, :old_size].T
        state_cov[old_size:, old_size:] = J @ state_cov[:21, :21] @ J.T

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (state_cov + state_cov.T) / 2.

    def add_feature_observations(self, feature_msg):
        state_id = self.state_server.imu_state.id
        curr_feature_num = len(self.map_server)
        tracked_feature_num = 0

        for feature in feature_msg.features:
            if feature.id not in self.map_server:
                # This is a new feature.
                map_feature = Feature(feature.id, self.optimization_config)
                # add a new feature in the map_feature
                map_feature.observations[state_id] = np.array([
                    feature.u0, feature.v0, feature.u1, feature.v1])
                self.map_server[feature.id] = map_feature
            else:
                # This is an old feature.
                self.map_server[feature.id].observations[state_id] = np.array([
                    feature.u0, feature.v0, feature.u1, feature.v1])
                tracked_feature_num += 1

        self.tracking_rate = tracked_feature_num / (curr_feature_num + 1e-5)

    def measurement_jacobian(self, cam_state_id, feature_id):
        """
        This function is used to compute the measurement Jacobian
        for a single feature observed at a single camera frame.
        """
        # Prepare all the required data.
        cam_state = self.state_server.cam_states[cam_state_id]
        feature = self.map_server[feature_id]

        # Cam0 pose.
        R_w_c0 = to_rotation(cam_state.orientation)
        t_c0_w = cam_state.position

        # Cam1 pose.
        R_w_c1 = CAMState.R_cam0_cam1 @ R_w_c0
        t_c1_w = t_c0_w - R_w_c1.T @ CAMState.t_cam0_cam1

        # 3d feature position in the world frame.
        # And its observation with the stereo cameras.
        p_w = feature.position
        z = feature.observations[cam_state_id]

        # Convert the feature position from the world frame to
        # the cam0 and cam1 frame.
        p_c0 = R_w_c0 @ (p_w - t_c0_w)
        p_c1 = R_w_c1 @ (p_w - t_c1_w)

        # Compute the Jacobians.
        dz_dpc0 = np.zeros((4, 3))
        dz_dpc0[0, 0] = 1 / p_c0[2]
        dz_dpc0[1, 1] = 1 / p_c0[2]
        dz_dpc0[0, 2] = -p_c0[0] / (p_c0[2] * p_c0[2])
        dz_dpc0[1, 2] = -p_c0[1] / (p_c0[2] * p_c0[2])

        dz_dpc1 = np.zeros((4, 3))
        dz_dpc1[2, 0] = 1 / p_c1[2]
        dz_dpc1[3, 1] = 1 / p_c1[2]
        dz_dpc1[2, 2] = -p_c1[0] / (p_c1[2] * p_c1[2])
        dz_dpc1[3, 2] = -p_c1[1] / (p_c1[2] * p_c1[2])

        dpc0_dxc = np.zeros((3, 6))
        dpc0_dxc[:, :3] = skew(p_c0)
        dpc0_dxc[:, 3:] = -R_w_c0

        dpc1_dxc = np.zeros((3, 6))
        dpc1_dxc[:, :3] = CAMState.R_cam0_cam1 @ skew(p_c0)
        dpc1_dxc[:, 3:] = -R_w_c1

        dpc0_dpg = R_w_c0
        dpc1_dpg = R_w_c1

        H_x = dz_dpc0 @ dpc0_dxc + dz_dpc1 @ dpc1_dxc  # shape: (4, 6)
        H_f = dz_dpc0 @ dpc0_dpg + dz_dpc1 @ dpc1_dpg  # shape: (4, 3)

        # Modifty the measurement Jacobian to ensure observability constrain.
        A = H_x  # shape: (4, 6)
        u = np.zeros(6)
        u[:3] = to_rotation(cam_state.orientation_null) @ IMUState.gravity
        u[3:] = skew(p_w - cam_state.position_null) @ IMUState.gravity

        H_x = A - (A @ u)[:, None] * u / (u @ u)
        H_f = -H_x[:4, 3:6]

        # Compute the residual.
        r = z - np.array([*p_c0[:2] / p_c0[2], *p_c1[:2] / p_c1[2]])

        # H_x: shape (4, 6)
        # H_f: shape (4, 3)
        # r  : shape (4,)
        return H_x, H_f, r

    def feature_jacobian(self, feature_id, cam_state_ids):
        """
        This function computes the Jacobian of all measurements viewed
        in the given camera states of this feature.
        """
        feature = self.map_server[feature_id]

        # Check how many camera states in the provided camera id
        # camera has actually seen this feature.
        valid_cam_state_ids = []
        for cam_id in cam_state_ids:
            if cam_id in feature.observations:
                valid_cam_state_ids.append(cam_id)

        jacobian_row_size = 4 * len(valid_cam_state_ids)

        cam_states = self.state_server.cam_states

        if not self.is_uwb_init:
            H_xj = np.zeros((jacobian_row_size, 21 + len(self.state_server.cam_states) * 6))
        else:
            H_xj = np.zeros((jacobian_row_size, 33 + len(self.state_server.cam_states) * 6))

        H_fj = np.zeros((jacobian_row_size, 3))
        r_j = np.zeros(jacobian_row_size)

        stack_count = 0
        for cam_id in valid_cam_state_ids:
            H_xi, H_fi, r_i = self.measurement_jacobian(cam_id, feature.id)

            # Stack the Jacobians.
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            H_xj[stack_count:stack_count + 4, 21 + 6 * idx:21 + 6 * (idx + 1)] = H_xi
            H_fj[stack_count:stack_count + 4, :3] = H_fi
            r_j[stack_count:stack_count + 4] = r_i
            stack_count += 4

        # Project the residual and Jacobians onto the nullspace of H_fj.
        # svd of H_fj
        U, _, _ = np.linalg.svd(H_fj)
        A = U[:, 3:]

        H_x = A.T @ H_xj
        r = A.T @ r_j

        return H_x, r

    def measurement_update(self, H, r):
        if len(H) == 0 or len(r) == 0:
            return

        # Decompose the final Jacobian matrix to reduce computational
        # complexity as in Equation (28), (29).
        if H.shape[0] > H.shape[1]:
            # QR decomposition
            Q, R = np.linalg.qr(H, mode='reduced')  # if M > N, return (M, N), (N, N)
            H_thin = R  # shape (N, N)
            r_thin = Q.T @ r  # shape (N,)
        else:
            H_thin = H  # shape (M, N)
            r_thin = r  # shape (M)

        # Compute the Kalman gain.
        P = self.state_server.state_cov
        S = H_thin @ P @ H_thin.T + (self.config.observation_noise * np.identity(len(H_thin)))
        K_transpose = np.linalg.solve(S, H_thin @ P)
        K = K_transpose.T  # shape (N, K)

        # Compute the error of the state.
        delta_x = K @ r_thin

        # Update the IMU state.
        delta_x_imu = delta_x[:21]

        if (np.linalg.norm(delta_x_imu[6:9]) > 0.5 or
                np.linalg.norm(delta_x_imu[12:15]) > 1.0):
            print('[Warning] Update change is too large')

        dq_imu = small_angle_quaternion(delta_x_imu[:3])
        imu_state = self.state_server.imu_state
        imu_state.orientation = quaternion_multiplication(
            dq_imu, imu_state.orientation)
        imu_state.gyro_bias += delta_x_imu[3:6]
        imu_state.velocity += delta_x_imu[6:9]
        imu_state.acc_bias += delta_x_imu[9:12]
        imu_state.position += delta_x_imu[12:15]

        dq_extrinsic = small_angle_quaternion(delta_x_imu[15:18])
        imu_state.R_imu_cam0 = to_rotation(dq_extrinsic) @ imu_state.R_imu_cam0
        imu_state.t_cam0_imu += delta_x_imu[18:21]

        # Update the camera states.
        for i, (cam_id, cam_state) in enumerate(
                self.state_server.cam_states.items()):
            delta_x_cam = delta_x[21 + i * 6:27 + i * 6]
            dq_cam = small_angle_quaternion(delta_x_cam[:3])
            cam_state.orientation = quaternion_multiplication(
                dq_cam, cam_state.orientation)
            cam_state.position += delta_x_cam[3:]

        # Update state covariance.
        I_KH = np.identity(len(K)) - K @ H_thin
        # state_cov = I_KH @ self.state_server.state_cov @ I_KH.T + (
        #     K @ K.T * self.config.observation_noise)
        state_cov = I_KH @ self.state_server.state_cov  # ?

        # Fix the covariance to be symmetric
        self.state_server.state_cov = (state_cov + state_cov.T) / 2.

    def gating_test(self, H, r, dof):
        if not self.is_uwb_init:
            P1 = H @ self.state_server.state_cov @ H.T
        else:
            P1 = H @ self.state_server.state_cov @ H.T
        P2 = self.config.observation_noise * np.identity(len(H))
        gamma = r @ np.linalg.solve(P1 + P2, r)

        if (gamma < self.chi_squared_test_table[dof]):
            return True
        else:
            return False

    def remove_lost_features(self):
        # Remove the features that lost track.
        # BTW, find the size the final Jacobian matrix and residual vector.
        jacobian_row_size = 0
        invalid_feature_ids = []
        processed_feature_ids = []

        for feature in self.map_server.values():
            # Pass the features that are still being tracked.
            if self.state_server.imu_state.id in feature.observations:
                continue
            if len(feature.observations) < 3:
                invalid_feature_ids.append(feature.id)
                continue

            # Check if the feature can be initialized if it has not been.
            if not feature.is_initialized:
                # Ensure there is enough translation to triangulate the feature
                if not feature.check_motion(self.state_server.cam_states):
                    invalid_feature_ids.append(feature.id)
                    continue

                # Intialize the feature position based on all current available
                # measurements.
                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    invalid_feature_ids.append(feature.id)
                    continue

            jacobian_row_size += (4 * len(feature.observations) - 3)
            processed_feature_ids.append(feature.id)

        # Remove the features that do not have enough measurements.
        for feature_id in invalid_feature_ids:
            del self.map_server[feature_id]

        # Return if there is no lost feature to be processed.
        if len(processed_feature_ids) == 0:
            return

        if not self.is_uwb_init:
            H_x = np.zeros((jacobian_row_size,
                        21 + 6 * len(self.state_server.cam_states)))
        else:
            H_x = np.zeros((jacobian_row_size,
                            33 + 6 * len(self.state_server.cam_states)))

        r = np.zeros(jacobian_row_size)
        stack_count = 0

        # Process the features which lose track.
        for feature_id in processed_feature_ids:
            feature = self.map_server[feature_id]

            cam_state_ids = []
            for cam_id, measurement in feature.observations.items():
                cam_state_ids.append(cam_id)

            H_xj, r_j = self.feature_jacobian(feature.id, cam_state_ids)

            if self.gating_test(H_xj, r_j, len(cam_state_ids) - 1):
                H_x[stack_count:stack_count + H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count + len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            # Put an upper bound on the row size of measurement Jacobian,
            # which helps guarantee the executation time.
            if stack_count > 1500:
                break

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform the measurement update step.
        self.measurement_update(H_x, r)

        # Remove all processed features from the map.
        for feature_id in processed_feature_ids:
            del self.map_server[feature_id]

    def find_redundant_cam_states(self):
        # Move the iterator to the key position.
        cam_state_pairs = list(self.state_server.cam_states.items())

        key_cam_state_idx = len(cam_state_pairs) - 4
        cam_state_idx = key_cam_state_idx + 1
        first_cam_state_idx = 0

        # Pose of the key camera state.
        key_position = cam_state_pairs[key_cam_state_idx][1].position
        key_rotation = to_rotation(
            cam_state_pairs[key_cam_state_idx][1].orientation)

        rm_cam_state_ids = []

        # Mark the camera states to be removed based on the
        # motion between states.
        for i in range(2):
            position = cam_state_pairs[cam_state_idx][1].position
            rotation = to_rotation(
                cam_state_pairs[cam_state_idx][1].orientation)

            distance = np.linalg.norm(position - key_position)
            angle = 2 * np.arccos(to_quaternion(
                rotation @ key_rotation.T)[-1])

            if angle < 0.2618 and distance < 0.4 and self.tracking_rate > 0.5:
                rm_cam_state_ids.append(cam_state_pairs[cam_state_idx][0])
                cam_state_idx += 1
            else:
                rm_cam_state_ids.append(cam_state_pairs[first_cam_state_idx][0])
                first_cam_state_idx += 1
                cam_state_idx += 1

        # Sort the elements in the output list.
        rm_cam_state_ids = sorted(rm_cam_state_ids)
        return rm_cam_state_ids

    def prune_cam_state_buffer(self):
        if len(self.state_server.cam_states) < self.config.max_cam_state_size:
            return

        # Find two camera states to be removed.
        rm_cam_state_ids = self.find_redundant_cam_states()

        # Find the size of the Jacobian matrix.
        jacobian_row_size = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue
            if len(involved_cam_state_ids) == 1:
                del feature.observations[involved_cam_state_ids[0]]
                continue

            if not feature.is_initialized:
                # Check if the feature can be initialize.
                if not feature.check_motion(self.state_server.cam_states):
                    # If the feature cannot be initialized, just remove
                    # the observations associated with the camera states
                    # to be removed.
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

                ret = feature.initialize_position(self.state_server.cam_states)
                if ret is False:
                    for cam_id in involved_cam_state_ids:
                        del feature.observations[cam_id]
                    continue

            jacobian_row_size += 4 * len(involved_cam_state_ids) - 3

        # Compute the Jacobian and residual.
        if not self.is_uwb_init:
            H_x = np.zeros((jacobian_row_size, 21 + 6 * len(self.state_server.cam_states)))
        else:
            H_x = np.zeros((jacobian_row_size, 33 + 6 * len(self.state_server.cam_states)))
        r = np.zeros(jacobian_row_size)

        stack_count = 0
        for feature in self.map_server.values():
            # Check how many camera states to be removed are associated
            # with this feature.
            involved_cam_state_ids = []
            for cam_id in rm_cam_state_ids:
                if cam_id in feature.observations:
                    involved_cam_state_ids.append(cam_id)

            if len(involved_cam_state_ids) == 0:
                continue

            H_xj, r_j = self.feature_jacobian(feature.id, involved_cam_state_ids)

            if self.gating_test(H_xj, r_j, len(involved_cam_state_ids)):
                H_x[stack_count:stack_count + H_xj.shape[0], :H_xj.shape[1]] = H_xj
                r[stack_count:stack_count + len(r_j)] = r_j
                stack_count += H_xj.shape[0]

            for cam_id in involved_cam_state_ids:
                del feature.observations[cam_id]

        H_x = H_x[:stack_count]
        r = r[:stack_count]

        # Perform measurement update.
        self.measurement_update(H_x, r)

        for cam_id in rm_cam_state_ids:
            idx = list(self.state_server.cam_states.keys()).index(cam_id)
            cam_state_start = 21 + 6 * idx
            cam_state_end = cam_state_start + 6

            # Remove the corresponding rows and columns in the state
            # covariance matrix.
            state_cov = self.state_server.state_cov.copy()
            if cam_state_end < state_cov.shape[0]:
                size = state_cov.shape[0]
                state_cov[cam_state_start:-6, :] = state_cov[cam_state_end:, :]
                state_cov[:, cam_state_start:-6] = state_cov[:, cam_state_end:]
            self.state_server.state_cov = state_cov[:-6, :-6]

            # Remove this camera state in the state vector.
            del self.state_server.cam_states[cam_id]

    def reset_state_cov(self):
        """
        Reset the state covariance.
        """
        state_cov = np.zeros((21, 21))
        state_cov[3: 6, 3: 6] = self.config.gyro_bias_cov * np.identity(3)
        state_cov[6: 9, 6: 9] = self.config.velocity_cov * np.identity(3)
        state_cov[9:12, 9:12] = self.config.acc_bias_cov * np.identity(3)
        state_cov[15:18, 15:18] = self.config.extrinsic_rotation_cov * np.identity(3)
        state_cov[18:21, 18:21] = self.config.extrinsic_translation_cov * np.identity(3)
        self.state_server.state_cov = state_cov

    def reset(self):
        """
        Reset the VIO to initial status.
        """
        # Reset the IMU state.
        imu_state = IMUState()
        imu_state.id = self.state_server.imu_state.id
        imu_state.R_imu_cam0 = self.state_server.imu_state.R_imu_cam0
        imu_state.t_cam0_imu = self.state_server.imu_state.t_cam0_imu
        self.state_server.imu_state = imu_state

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Reset the state covariance.
        self.reset_state_cov()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Clear the IMU msg buffer.
        self.imu_msg_buffer.clear()

        # Reset the starting flags.
        self.is_gravity_set = False
        self.is_first_img = True

    def online_reset(self):
        """
        Reset the system online if the uncertainty is too large.
        """
        # Never perform online reset if position std threshold is non-positive.
        if self.config.position_std_threshold <= 0:
            return

        # Check the uncertainty of positions to determine if
        # the system can be reset.
        position_x_std = np.sqrt(self.state_server.state_cov[12, 12])
        position_y_std = np.sqrt(self.state_server.state_cov[13, 13])
        position_z_std = np.sqrt(self.state_server.state_cov[14, 14])

        if max(position_x_std, position_y_std, position_z_std
               ) < self.config.position_std_threshold:
            return

        print('Start online reset...')

        # Remove all existing camera states.
        self.state_server.cam_states.clear()

        # Clear all exsiting features in the map.
        self.map_server.clear()

        # Reset the state covariance.
        self.reset_state_cov()

    def publish(self, time):
        imu_state = self.state_server.imu_state
        uwb_state = self.state_server.uwb_state
        print('+++publish:')
        print('   timestamp:', imu_state.timestamp)
        print('   orientation:', imu_state.orientation)
        print('   position:', imu_state.position)
        print('   velocity:', imu_state.velocity)
        print()

        T_i_w = Isometry3d(
            to_rotation(imu_state.orientation).T,
            imu_state.position)
        T_b_w = IMUState.T_imu_body * T_i_w * IMUState.T_imu_body.inverse()
        body_velocity = IMUState.T_imu_body.R @ imu_state.velocity

        R_w_c = imu_state.R_imu_cam0 @ T_i_w.R.T
        t_c_w = imu_state.position + T_i_w.R @ imu_state.t_cam0_imu
        T_c_w = Isometry3d(R_w_c.T, t_c_w)

        return namedtuple('vio_result', ['timestamp', 'pose', 'velocity', 'cam0_pose', 'anchor_pos'])(
            time, T_b_w, body_velocity, T_c_w, uwb_state)