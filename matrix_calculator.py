from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt

print(3/2)

# Q3
# for c) and d)
# J = np.matrix([[-2 - 3 * math.sqrt(3) / 2, math.sqrt(3) / 2, - math.sqrt(3)], [2 * math.sqrt(3) + 0.5, 0.5, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1]])
# for e) and f)
# J = np.matrix([[-2 - 3 * math.sqrt(3) / 2, math.sqrt(3) / 2, - math.sqrt(3)], [2 * math.sqrt(3) + 0.5, 0.5, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 1]])

# v = np.matrix([[-1, 2]]).transpose()

# v2 = np.matrix([[-1, 2, 1, -3, 0, -2]]).transpose()

# c)
# res = J.transpose() * (J * J.transpose()).I * v
# print(res)

# d)
# resp = (np.identity(3) - J.transpose() * (J * J.transpose()).I * J)
# print(resp)

# e)
# res2 = (J.transpose() * J).I * J.transpose() * v2
# print(res2)

# f)
# ve = J * res2
# print(ve)


# Q4
qi = 0
qf = 2
vf = 1
tf = 2

# Q4 a
# ac = 2
# tf += vf / ac
# qf += ac * (vf / ac) * (vf / ac) / 2
# tc = tf / 2 - math.sqrt((tf * tf * ac - 4 * (qf - qi)) / ac) / 2

# Q4 b
vc = 1.5
tc = 1.2
ac = vc / tc
tf = 2.8
qf = 2.4

print("tc = " + str(tc)) 

t1 = np.arange(0.0, tc, 0.02)
t2 = np.arange(tc, tf - tc, 0.02)
t3 = np.arange(tf - tc, tf, 0.02)

# Position t
plt.xlabel('Time t')
plt.ylabel('Position q(t)')
plt.title('Position trajectory')
plt.grid(True)
plt.plot(t1, qi + ac * t1 * t1 / 2, t2, qi + ac * tc * (t2 - tc / 2), t3, qf - ac * (tf - t3) * (tf - t3) / 2)
plt.show()

# Velocity t
# plt.xlabel('Time t')
# plt.ylabel('Velocity q\'(t)')
# plt.title('Velocity trajectory')
# plt.grid(True)
# plt.plot(t1, ac * t1, t2, ac * tc + 0 * t2, t3, ac * tf - ac * t3)
# plt.show()

#Acceleration t
# plt.xlabel('Time t')
# plt.ylabel('Acceleration q\'\'(t)')
# plt.title('Acceleration trajectory')
# plt.grid(True)
# plt.plot(t1, ac + 0 * t1, t2, 0 + 0 * t2, t3, -ac + 0 * t3)
# plt.show()

