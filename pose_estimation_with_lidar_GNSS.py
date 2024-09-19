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

C_li = np.array([
    [0.99376, -0.09722, 0.05466],
    [0.09971, 0.99401, -0.04475],
    [-0.04998, 0.04992, 0.9975]
])

t_i_li_i = np.array([0.5, 0.1, 0.5])

lidar.data = (C_li @ lidar.data.T).T + t_i_li_i

# print(lidar.data)
# print(gnss.data)

# # Plot lidar and ground truth trajectory
lidar_fig = plt.figure()
ax = lidar_fig.add_subplot(111, projection='3d')

ax.plot(gt.p[:,0], gt.p[:,1], gt.p[:,2])
ax.plot(lidar.data[:,0], lidar.data[:,1], lidar.data[:,2])
ax.plot(gnss.data[:,0], gnss.data[:,1], gnss.data[:,2])

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

ax.set_title('Lidar and GNSS and Ground Truth trajectory')
ax.set_zlim(-1, 5)
plt.show()
