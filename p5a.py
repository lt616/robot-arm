from __future__ import division
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv

#DH = [a, alpha, d]

PI = 3.141592657
BIAS = 1e-2

def computeHomo(joint, DH):
	temp = np.zeros((4, 4))

	temp[0][0] = math.cos(joint)
	temp[0][1] = (-1) * math.sin(joint) * math.cos(DH[1])
	temp[0][2] = math.sin(joint) * math.sin(DH[1])
	temp[0][3] = DH[0] * math.cos(joint)

	temp[1][0] = math.sin(joint)
	temp[1][1] = math.cos(joint) * math.cos(DH[1])
	temp[1][2] = (-1) * math.cos(joint) * math.sin(DH[1])
	temp[1][3] = DH[0] * math.sin(joint)

	temp[2][0] = 0
	temp[2][1] = math.sin(DH[1])
	temp[2][2] = math.cos(DH[1])
	temp[2][3] = DH[2]

	temp[3][0] = 0
	temp[3][1] = 0
	temp[3][2] = 0
	temp[3][3] = 1

	return temp


def computeAllHomo(joints, DH):
	As = {}
	Homos = {}
	Zs = {}
	Ps = {}

	Zs[0] = np.matrix([[0], [0], [1]])
	Ps[0] = np.matrix([[0], [0], [0]])

	for i in np.arange(0, 7):
		As[i] = computeHomo(joints[i], DH[i])

		if i != 0:
			Homos[i] = np.matmul(Homos[i - 1], As[i])
		else:
			Homos[0] = As[0]

		Zs[i + 1] = Homos[i][:3, [2]]
		Ps[i + 1] = Homos[i][:3, [3]]

		# print("\n" + str(i) + " homoT:")
		# print(Homos[i])
		# print(Ps[i + 1])

	return Zs, Ps

def computeJacobian(joints, DH):
	Zs, Ps = computeAllHomo(joints, DH)
	J = np.zeros((6, 7))

	for i in np.arange(0, 7):
		Jpi = np.cross(Zs[i].transpose(), np.subtract(Ps[7], Ps[i]).transpose())
		J[0, i] = Jpi[0][0]
		J[1, i] = Jpi[0][1]
		J[2, i] = Jpi[0][2]

		J[3, i] = Zs[i][0][0]
		J[4, i] = Zs[i][1][0]
		J[5, i] = Zs[i][2][0]

		# print(Zs[i])
		# print(J)

	return J


def computeInitError(Ti, Td):
	e = np.zeros((6, 1))
	pi = Ti[:3, [3]]
	pd = Td[:3, [3]]

	e[0, 0] = pd[0][0] - pi[0][0]
	e[1, 0] = pd[1][0] - pi[1][0]
	e[2, 0] = pd[2][0] - pi[2][0]

	xi = Ti[:3, [0]].transpose()
	yi = Ti[:3, [1]].transpose()
	zi = Ti[:3, [2]].transpose()

	xd = Td[:3, [0]].transpose()
	yd = Td[:3, [1]].transpose()
	zd = Td[:3, [2]].transpose()

	print(xd)
	print(yd)
	print(zd)

	e0 = np.cross(xi, xd) + np.cross(yi, yd) + np.cross(zi, zd)

	e[3, 0] = e0[0, 0]
	e[4, 0] = e0[0, 1]
	e[5, 0] = e0[0, 2]

	return e


def computeAlpha(J, e):
	JJTe = np.matmul(np.matmul(J, J.transpose()), e).transpose()[0]
	return np.dot(JJTe, e.transpose()[0]) / np.dot(JJTe, JJTe)


def computeRightPseudo(J):
	return np.dot(J.transpose(), inv(np.matmul(J, J.transpose())))


def computeDampedLS(J):
	return np.matmul(J.transpose(), inv(np.matmul(J, J.transpose()) + np.identity(6)))


def updateJacoTranspose(e, qi, DH):

	deltax = np.matrix([[1], [1], [1], [1], [1], [1]])
	i = 0
	while (not checkTiny(deltax)):
		J = computeJacobian(qi, DH)

		alpha = computeAlpha(J, e)
		print(alpha)

		deltaq = np.matmul(alpha * J.transpose(), e)
		deltax = np.matmul(J, deltaq)
		print(deltax)

		qi += deltaq.transpose()[0]
		e = deltax

	print("all done!")


def updateJacoInverse(e, qi, DH):

	deltax = np.matrix([[1], [1], [1], [1], [1], [1]])

	while (not checkTiny(deltax)):
		J = computeJacobian(qi, DH)
		Jinv = computeRightPseudo(J)

		deltaq = np.matmul(Jinv, e)
		deltax = np.matmul(J, deltaq)

		# print(deltax)

		qi += deltaq.transpose()[0]
		# print(deltaq)
		print(qi)

		e = deltax

	print("all done")


def updateDampedLS(e, qi, DH):

	deltax = np.matrix([[1], [1], [1], [1], [1], [1]])

	while (not checkTiny(deltax)):
		J = computeJacobian(qi, DH)
		Jstar = computeDampedLS(J)

		deltaq = np.matmul(Jstar, e)
		deltax = np.matmul(J, deltaq)

		print(deltax)

		qi += deltaq.transpose()[0]
		e = deltax

	print("all done")


def checkTiny(x):
	for i in np.arange(0, 6):
		if x[i, 0] > BIAS:
			return False
	return True


def initTs():
	Ti = np.matrix([[0.515, 0.481, 0.709, -0.0815], [0.476, -0.849, 0.230, -0.0409], [0.713, 0.219, -0.667, 0.399], [0, 0, 0, 1]])
	Td = np.matrix([[-0.781, -0.474, 0.407, -0.0206], [-0.220, 0.818, 0.531, -0.147], [-0.585, 0.325, -0.743, 0.620], [0, 0, 0, 1]])

	return Ti, Td


def initDH(): 
	DHrow0 = [0, 0, 0.333]
	DHrow1 = [0, -PI / 2, 0]
	DHrow2 = [0, PI / 2, 0.316]
	DHrow3 = [0.0825, PI / 2, 0]
	DHrow4 = [-0.0825, -PI / 2, 0.384]
	DHrow5 = [0, PI / 2, 0]
	DHrow6 = [0.088, PI / 2, 0]

	return [DHrow0, DHrow1, DHrow2, DHrow3, DHrow4, DHrow5, DHrow6]

def initJoints():
	return [0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7]


DH = initDH()
joints = initJoints()
Zs, Ps = computeAllHomo(joints, DH)
J = computeJacobian(joints, DH)

# P5a
print(J)

Ti, Td = initTs()
e = computeInitError(Ti, Td)

print(e)

# updateJacoTranspose(e, joints, DH)
updateJacoInverse(e, joints, DH)
# updateDampedLS(e, joints, DH)






