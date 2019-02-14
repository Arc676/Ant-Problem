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

import numpy as np
from scipy import optimize

def integrate(method, rhs, y0, T, N):
    y = np.empty((N+1,) + y0.shape)

    t0, dt = 0.0, T/N
    y[0,...] = y0
    for i in range(0, N):
        y[i+1,...] = method(rhs, y[i,...], t0 + i*dt, dt)

    t = np.arange(N+1)*dt
    return t, y

def explicit_euler_step(rhs, y0, t0, dt):
    return y0 + dt*rhs(t0, y0)

def explicit_euler(rhs, y0, T, N):
    return integrate(explicit_euler_step, rhs, y0, T, N)

def implicit_euler_step(rhs, y0, t0, dt):
    # Das implizite Eulerverfahren ist
    #     y1 = y0 + dt * rhs(t+dt, y1)
    # Wir müssen diese gleichung nach y1 auflösen.
    F = lambda y1 : y1 - (y0 + dt * rhs(t0 + dt, y1))

    initial_guess = explicit_euler_step(rhs, y0, t0, dt)
    return optimize.fsolve(F, initial_guess)

def implicit_euler(rhs, y0, T, N):
    return integrate(implicit_euler_step, rhs, y0, T, N)

def implicit_mid_point_step(rhs, y0, t0, dt):
    # Die implizite Mittelpunktsregel ist
    #    y1 = y0 + dt*rhs(t+0.5*dt, 0.5*(y0 + y1))
    F = lambda y1 : y1 - (y0 + dt*rhs(t0 + 0.5*dt, 0.5*(y0 + y1)))

    initial_guess = explicit_euler_step(rhs, y0, t0, dt)
    return optimize.fsolve(F, initial_guess)

def implicit_mid_point(rhs, y0, T, N):
    return integrate(implicit_mid_point_step, rhs, y0, T, N)

def velocity_verlet_step(rhs, xv0, t0, dt):
    xv0 = xv0.reshape((2, -1))
    xv1 = np.empty_like(xv0)
    x0, x1 = xv0[0,:], xv1[0,:]
    v0, v1 = xv0[1,:], xv1[1,:]

    x1[:] = x0 + dt*v0 + 0.5*dt**2 * rhs(t0, x0)
    v1[:] = v0 + 0.5*dt*(rhs(t0, x0) + rhs(t0+dt, x1))

    return xv1.reshape(-1)

def velocity_verlet(rhs, y0, T, N):
    return integrate(velocity_verlet_step, rhs, y0, T, N)
