# Copyright (C) 2019 Arc676/Alessandro Vinciguerra <alesvinciguerra@gmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation (version 3)

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import argparse

parser = argparse.ArgumentParser("Four Ant Problem simulation, generalized to any number of ants.")
parser.add_argument("--ants", "-a", nargs=1, default=[4], type=int, help="Number of ants", dest="ants")
parser.add_argument("-o", "--output", nargs=1, help="Output file", dest="output")
parser.add_argument("-t", "--duration", nargs=1, default=[3550], type=int, help="Number of seconds to simulate", dest="tf")

args = parser.parse_args()
ants = args.ants[0]
tf = args.tf[0]
output = args.output

import numpy as np
from numpy.linalg import norm

import matplotlib
if output is not None:
	output = output[0]
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ode_solvers import implicit_mid_point
import math

def normalize(v):
	return v/norm(v)

def velocity(t, q):
	velocities = [
		normalize(q[0:2] - q[-2:]) if i == ants * 2 else
		normalize(q[i:i + 2] - q[i - 2:i])
		for i in range(2, ants * 2 + 2, 2)
	]
	v = np.concatenate(tuple(velocities), axis=None)
	return v

radius = 3600 / (2 * math.sin(math.pi / ants))
y0 = np.array([
	radius * math.cos(math.radians(360 * (n // 2) / ants)) if n % 2 == 0 else
	radius * math.sin(math.radians(360 * (n // 2) / ants))
	for n in range(ants * 2)
])
T, y = implicit_mid_point(velocity, y0, tf, tf)

fig = plt.figure(0)
for i in range(ants):
	plt.plot(y[:,(i * 2)], y[:,(i * 2 + 1)], label=chr(65 + i))
plt.legend()
plt.axis("equal")
if output is None:
	plt.show()
else:
	fig.savefig(output)
