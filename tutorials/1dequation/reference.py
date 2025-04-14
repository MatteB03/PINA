# this is a file with very few modifications from the original in the thesis gitlab
# the main addition is a command at the end to save the PINN solution of the thesis 
# and the FDM solution, so as to compare them in the plots.py file

#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
w = 1.0
T = 1.0
num_epochs = 15000
learning_rate = 10e-3
num_boundary_samples = 1000
num_collocation_points = 5000
mu_exact = 2.0
mu_guess = 1.0
l = w / 3

# Boundary function
def boundary(x):
    y = x * (w + 2*T)
    if y < T:
        return 0.0, T - y
    elif y < w + T:
        return y - T, 0.0
    else:
        return w, y - T - w

# Initial condition function
def sol_bound(a):
    if a[1] != 0.0:
        return 0.0
    elif (a[0] - w/2) < l**2/4:
        return 1.0
    else:
        return 0.0

# Set random seed for reproducibility
tf.random.set_seed(100)

# Generate random uniform samples for the boundary conditions
unif = tf.random.uniform(shape=[num_boundary_samples], maxval=1.0)

# Generate boundary points and training data
b_points = tf.stack(tf.map_fn(fn=lambda x: boundary(x), elems=unif, dtype=(tf.float32, tf.float32)), axis=1)
u_train = tf.map_fn(fn=lambda x: sol_bound(x), elems=b_points, dtype=(tf.float32))

# Collocation points
x_grid = tf.random.uniform(shape=[num_collocation_points], maxval=w).numpy()
t_grid = tf.random.uniform(shape=[num_collocation_points], maxval=T).numpy()

# Define the PINN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])

# Define loss functions
mu = tf.constant(mu_exact)

def PDE(x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        u = model(tf.stack((x, t), axis=1))
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
    return u_t - mu*(u_xx) + tf.reshape(u**3 - u, (num_collocation_points,))

last_loss_fit = tf.constant([0.0])
def loss_fit():
    global last_loss_fit
    last_loss_fit = tf.reduce_mean(tf.square(model(b_points) - u_train[:, None]))
    return last_loss_fit

last_loss_PDE = tf.constant([0.0])
def loss_PDE():
    global last_loss_PDE
    last_loss_PDE = tf.reduce_mean(tf.square(PDE(tf.constant(x_grid), tf.constant(t_grid))))
    return last_loss_PDE

# Train the model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for i in range(num_epochs):
    with tf.GradientTape() as tape:
        loss = loss_fit() + loss_PDE()
    
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    if i % 100 == 0:
        print(f'iter = {i}, mu = {mu.numpy():.2f}, loss_fit = {last_loss_fit.numpy():.6f}, loss_PDE = {last_loss_PDE.numpy():.6f}')

# Post-processing FDM
nt = 10000
nx = 49
dx = w / nx
dt = T / nt
dx2 = dx**2

def do_timestep(u0, u):
    u[1:-1] = u0[1:-1] + mu_exact * dt * (u0[2:] - 2.0 * u0[1:-1] + u0[:-2]) / dx2 - dt * (np.power(u0[1:-1], 3) - u0[1:-1])
    u0 = u.copy()
    return u0, u

u0 = np.zeros(nx + 1)
u = u0.copy()

# Initialize the FDM solution
for i in range(nx + 1):
    if (i * dx - w / 2)**2 < l**2 / 4:
        u0[i] = 1.0

# Time points for testing
t_test = [100, 500, 2000, 5000]
xx = np.linspace(0, w, num=nx + 1, dtype=np.float32)
t_vals = np.linspace(0, T, num=50)  # Full 50 time steps

# Compute full PINN solution (all 50 time steps)
pinn_solutions_full = []
for t in t_vals:
    test_points = np.stack((xx, np.full_like(xx, t)), axis=1)
    pinn_solutions_full.append(model.predict(test_points))

pinn_solutions_full = np.array(pinn_solutions_full)  # Shape: (50, nx+1)

# Compute full FDM solution (all 50 time steps)
fdm_solutions_full = np.zeros((50, nx + 1))
for i in range(50):
    for _ in range(int(t_vals[i] / dt)):
        u0, u = do_timestep(u0, u)
    fdm_solutions_full[i, :] = u.copy()

# Save both solutions
np.savez("saved_full_solution.npz", x=xx, t=t_vals, u_pinn=pinn_solutions_full, u_fdm=fdm_solutions_full)