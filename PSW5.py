# https://gist.github.com/xehivs/786d9d1d3aabe3d6ae4a137e17f740a5
import numpy as np
import matplotlib.pyplot as plt
#
# fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
#
# points = [
#     [0,0],
#     [1,0],
#     [1,1],
#     [0,1],
#     [0,0]
# ]
#
# ones = np.ones((len(points), 1))
#
# A = np.concatenate([points, ones], axis=1)
#
# theta = np.pi/3
#
# T = {
#     'identity' : [
#         [1,0,0],
#         [0,1,0],
#         [0,0,1],
#     ],
#     'x-y translation' : [
#         [1,0, .5],
#         [0,1, .5],
#         [0,0,1  ]
#     ],
#     'x-reflection' : [
#         [-1,0,0],
#         [ 0,1,0],
#         [ 0,0,1]
#     ],
#     'y-reflection' : [
#         [1, 0,0],
#         [0,-1,0],
#         [0, 0,1]
#     ],
#     'x-scale' : [
#         [2,0,0],
#         [0,1,0],
#         [0,0,1]
#     ],
#     'y-scale' : [
#         [1,0,0],
#         [0,2,0],
#         [0,0,1]
#     ],
#     'angle-rotation' : [
#         [np.cos(theta), -np.sin(theta),0],
#         [np.sin(theta),  np.cos(theta),0],
#         [0         ,0           ,1]
#     ],
#     'x-inclination' : [
#         [1, .5,0],
#         [0,1  ,0],
#         [0,0  ,1]
#     ],
#     'y-inclination' : [
#         [1, 0,0],
#         [.5,1  ,0],
#         [0,0  ,1]
#     ]
#
# }
#
# for i, key in enumerate(T):
#     Z = A @ np.array(T[key]).T
#
#     axs[i//3,i%3].set_title(key)
#     axs[i//3,i%3].plot(Z[:,0], Z[:,1])
#     axs[i//3,i%3].plot(A[:,0], A[:,1])
#
# plt.show()
import scipy.interpolate
from scipy.interpolate import griddata
import skimage
fig, axs = plt.subplots(4, 1)

chelsea = skimage.data.chelsea()
chelsea_mean = np.mean(chelsea, axis=-1).astype(np.uint8)


skip = 10
chelsea_skip = chelsea_mean[::skip, ::skip]

axs[0].imshow(chelsea_skip, cmap='Blues_r')


source_values = chelsea_skip.reshape(-1)
aspect = chelsea_skip.shape[0]/chelsea_skip.shape[1]

x_source_space = np.linspace(0,1,chelsea_skip.shape[1])
y_source_space = np.linspace(0,aspect,chelsea_skip.shape[0])


A = np.array(np.meshgrid(x_source_space,y_source_space))
A = A.reshape((2,-1)).T

theta = np.pi/4

T = [
            [np.cos(theta), -np.sin(theta),0],
            [np.sin(theta),  np.cos(theta),0],
            [0         ,0           ,1]
        ]

ones = np.ones(len(A))
A = np.concatenate([A, np.expand_dims(ones, axis=1)], axis=1)

axs[1].scatter(A[:,0], -A[:,1], c=source_values, cmap='Blues_r')

Z = A @ np.array(T).T

axs[2].scatter(Z[:,0], -Z[:,1], c=source_values, cmap='Blues_r')
axs[2].scatter(A[:,0], -A[:,1], c=source_values, cmap='Blues_r')

XX, YY = np.mgrid[np.min(Z[:,0]):np.max(Z[:,0]):512j,
         np.min(Z[:,1]):np.max(Z[:,1]):512j]

D = griddata((Z[:,0], Z[:,1]), source_values, (XX,YY), method='linear', fill_value=0)

axs[3].imshow(D,  cmap='Reds_r')


plt.show()
