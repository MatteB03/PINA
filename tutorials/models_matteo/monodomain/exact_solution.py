# this file contains the exact (FDM) solution exactly as done in the thesis
# see: https://gitlab.com/ADeGobbis/pinn-electrophysiology/-/blob/master/old/alternate_pde_loss/coll_points.py

import numpy as np
import matplotlib.pyplot as plt


# Parameters of the problem
w = h = 100  # Plate size (mm)
dx = dy = 0.5  # Grid resolution (mm)
D_true = 12.9 * 0.25  # Diffusivity (mm/ms)
time_max = 2.0  # Time horizon (ms)
dt = 0.01  # Time step (ms)

nx, ny, nt = int(w/dx), int(h/dy), int(time_max/dt)
dx2, dy2 = dx*dx, dy*dy

Tcool = 0.0
Thot = 1.0

# Initialize the FDM grid and initial condition
u_fdm = np.full((nx + 1, ny + 1, nt + 1), Tcool)

r, cx, cy = 25, 50, 50  # Circular hot spot radius and center
r2 = r**2
for i in range(nx + 1):
    for j in range(ny + 1):
        p2 = (i*dx - cx)**2 + (j*dy - cy)**2
        if p2 < r2:
            u_fdm[i, j, 0] = Thot

# Function to perform the timestep propagation (FDM)
def do_timestep(u0):
    u = np.full((nx + 1, ny + 1), Tcool)
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D_true * dt * (
        (u0[2:, 1:-1] - 2.0*u0[1:-1, 1:-1] + u0[:-2, 1:-1]) / dx2
        + (u0[1:-1, 2:] - 2.0*u0[1:-1, 1:-1] + u0[1:-1, :-2]) / dy2)
    return u

# Propagate the solution in time
for k in range(nt):
    u_fdm[:, :, k+1] = do_timestep(u_fdm[:, :, k])

# Plot at certain times
k_time = [10, 50, 100, 150]
u_exact = u_fdm[:, :, k_time]

# Prepare to plot the comparison in a 4x1 subplot
fig, axs = plt.subplots(4, 1, figsize=(10, 12))  # 4 rows, 1 column
for idx, k in enumerate(k_time):
    axs[idx].imshow(u_exact[:, :, idx], cmap='hot', vmin=Tcool - 0.5, vmax=Thot + 0.5)
    axs[idx].set_axis_off()
    axs[idx].set_title(f'Time step {k * dt:.2f} ms')

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
plt.colorbar(axs[0].imshow(u_exact[:, :, 0], cmap='hot', vmin=Tcool - 0.5, vmax=Thot + 0.5), cax=cbar_ax)

plt.suptitle('Exact Solution Comparison')
plt.show() 
