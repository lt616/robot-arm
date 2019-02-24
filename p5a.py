from __future__ import division
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import os

#DH = [a, alpha, d]

PI = 3.141592657

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

def computeJacobian(Zs, Ps):

	J = np.zeros((6, 7))

	for i in np.arange(0, 7):
		Jpi = np.cross(Zs[i].transpose(), np.subtract(Ps[7], Ps[i]).transpose())
		J[0, i] = Jpi[0][0]
		J[1, i] = Jpi[0][1]
		J[2, i] = Jpi[0][2]

		J[3, i] = Zs[i][0][0]
		J[4, i] = Zs[i][1][0]
		J[5, i] = Zs[i][2][0]

		print(Zs[i])
		print(J)




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
computeJacobian(Zs, Ps)




