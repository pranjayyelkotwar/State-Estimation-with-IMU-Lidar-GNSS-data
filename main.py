# Starter code for the Coursera SDC Course 2 final project.
#
# Author: Trevor Ablett and Jonathan Kelly
# University of Toronto Institute for Aerospace Studies

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

#### 1. Data Loading and Preparation ##########################################################

# Load data from pickle file
with open('data/pt3_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract relevant data from the loaded dictionary
gt = data['gt']        # Ground truth data
imu_f = data['imu_f']  # IMU specific force data
imu_w = data['imu_w']  # IMU rotational velocity data
gnss = data['gnss']    # GNSS position data
lidar = data['lidar']  # LIDAR position data

# Plot ground truth trajectory
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()

# Define transformation matrix from LIDAR to IMU frame
C_li = np.array([
   [ 0.99376, -0.09722,  0.05466],
   [ 0.09971,  0.99401, -0.04475],
   [-0.04998,  0.04992,  0.9975 ]
])

# Define translation vector from LIDAR to IMU frame
t_i_li = np.array([0.5, 0.1, 0.5])

# Transform LIDAR data to the IMU frame
lidar.data = (C_li @ lidar.data.T).T + t_i_li

# Define sensor measurement variances
var_imu_f = 0.01
var_imu_w = 0.01
var_gnss  = 10.
var_lidar = 1.

# Define gravity vector
g = np.array([0, 0, -9.81])

# Initialize Jacobian matrices
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

# Initialize state estimation arrays
p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values from ground truth
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # initial covariance

# Initialize sensor data indices
gnss_i  = 0
lidar_i = 0

#### 2. Kalman Filter Implementation ##########################################################

def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
    """
    Perform measurement update step of the Kalman filter.
    """
    # 3.1 Compute Kalman Gain
    I = np.identity(3)
    R = I * sensor_var
    K = p_cov_check.dot(h_jac.T).dot(np.linalg.inv(h_jac.dot(p_cov_check).dot(h_jac.T) + R))

    # 3.2 Compute error state
    error = K.dot(y_k - p_check)

    # 3.3 Correct predicted state
    p_del = error[:3]
    v_del = error[3:6]
    phi_del = error[6:]

    p_hat = p_check + p_del
    v_hat = v_check + v_del
    q_hat = Quaternion(euler=phi_del).quat_mult_right(q_check) 

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.identity(9) - K.dot(h_jac)).dot(p_cov_check)

    return p_hat, v_hat, q_hat, p_cov_hat

# Main loop for state estimation
for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    rotation_matrix = Quaternion(*q_est[k-1]).to_mat()

    # 1.1 Linearize the motion model and compute Jacobians
    # Update position estimate
    p_est[k] = p_est[k-1] + delta_t*v_est[k-1] + (delta_t**2 / 2)*(rotation_matrix.dot(imu_f.data[k-1]) + g)
    # Update velocity estimate
    v_est[k] = v_est[k-1] + delta_t*(rotation_matrix.dot(imu_f.data[k-1]) + g)
    # Update orientation estimate
    q_est[k] = Quaternion(axis_angle=imu_w.data[k-1] * delta_t).quat_mult_right(q_est[k-1])

    # 2. Propagate uncertainty
    F = np.identity(9)
    Q = np.identity(6)
    F[:3, 3:6] = delta_t * np.identity(3)
    F[3:6, 6:] = -(rotation_matrix.dot(skew_symmetric(imu_f.data[k-1].reshape((3,1)))))
    Q[:, :3] *= delta_t**2 * var_imu_f
    Q[:, -3:] *= delta_t**2 * var_imu_w
    p_cov[k] = F.dot(p_cov[k-1]).dot(F.T) + l_jac.dot(Q).dot(l_jac.T)

    # 3. Check for available measurements and perform updates
    if lidar_i < lidar.t.shape[0] and lidar.t[lidar_i] == imu_f.t[k-1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_lidar, p_cov[k], lidar.data[lidar_i].T, p_est[k], v_est[k], q_est[k])
        lidar_i += 1
    if gnss_i < gnss.t.shape[0] and gnss.t[gnss_i] == imu_f.t[k-1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(var_gnss, p_cov[k], gnss.data[gnss_i].T, p_est[k], v_est[k], q_est[k])
        gnss_i += 1

#### 3. Visualization ##########################################################################

# Plot estimated trajectory against ground truth
est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62,0.77))
ax.view_init(elev=45, azim=-50)
plt.show()


# Plot the error of the position
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt),  3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0,0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), \
        angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt),  3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i+3])
ax[1,0].set_ylabel('Radians')
plt.show()