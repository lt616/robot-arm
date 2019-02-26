import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open("jacobianTranspose_pos.txt", "r") as file:
	xs = []
	ys = []
	zs = []

	i = 0
	for line in file:
		if i > 80:
			break

		i += 1

		data = line.split(",")
		xs.append(float(data[0]))
		ys.append(float(data[1]))
		zs.append(float(data[2]))

# Build plot chart
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(xs, ys, zs, 'red')
ax.scatter(xs, ys, zs)

plt.savefig('/Users/cherryzhao/Desktop/19Spring/COMS4733/assignment02/5_jacobianTranspose.png')
plt.show()

