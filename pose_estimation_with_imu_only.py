import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

with open('data/pt3_data.pkl', 'rb') as file:
    data = pickle.load(file)

gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

# # Plot the ground truth trajectory
# gt_fig = plt.figure()
# ax = gt_fig.add_subplot(111, projection='3d')
# ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.set_title('Ground Truth trajectory')
# ax.set_zlim(-1, 5)
# plt.show()

p_est = np.zeros((imu_f.data.shape[0], 3))
v_est = np.zeros((imu_f.data.shape[0], 3))
q_est = np.zeros((imu_f.data.shape[0], 4))

# Initial estimates
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()

g = np.array([0, 0, -9.81])  # gravity

for k in range(1, imu_f.data.shape[0]):  # Loop through the IMU data
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    rotation_matrix = Quaternion(*q_est[k-1]).to_mat()

    p_est[k] = p_est[k-1] + delta_t*v_est[k-1] + (delta_t**2 / 2)*(rotation_matrix.dot(imu_f.data[k-1]) + g)
    v_est[k] = v_est[k-1] + delta_t*(rotation_matrix.dot(imu_f.data[k-1]) + g)
    q_est[k] = Quaternion(axis_angle=imu_w.data[k-1] * delta_t).quat_mult_right(q_est[k-1])


# # Plot the estimated trajectory
# est_fig = plt.figure()
# ax = est_fig.add_subplot(111, projection='3d')
# ax.plot(p_est[:,0], p_est[:,1], p_est[:,2])
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# ax.set_title('Estimated trajectory')
# ax.set_zlim(-1, 5)
# plt.show()

# Plot the ground truth and estimated trajectory
gt_fig = plt.figure()
ax = gt_fig.add_subplot(111, projection='3d')
ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2], label='Ground Truth')
ax.plot(p_est[:,0], p_est[:,1], p_est[:,2], label='Estimated')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_title('Ground Truth and Estimated trajectory')
ax.set_zlim(-1, 5)
ax.legend()
plt.show()


