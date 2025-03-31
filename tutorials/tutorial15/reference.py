#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

##########################################################
#
# Finds and compares the solution of the nonlinear PDE to the one obtained with FDM
#
# u_t - mu*u_xx + u^3 - u = 0 in R x [0,T]
#
# with data only on the border of the domain
#
#################################################

w = 1.0
T = 1.0

num_epochs = 15000
learning_rate = 10e-3

num_boundary_samples = 1000
num_collocation_points = 5000
mu_exact = 2.0
mu_guess = 1.0

l = w/3



def boundary(x):
    y = x * (w + 2*T)
    if y < T:
        return 0.0, T - y
    elif y < w + T:
        return y - T, 0.0
    else:
        return w, y - T - w


tf.random.set_seed(100)

unif = tf.random.uniform(shape = [num_boundary_samples], maxval = 1.0)

def sol_bound(a):
    if a[1] != 0.0:
        return 0.0
    elif (a[0] - w/2) < l**2/4:
        return 1.0
    else:
        return 0.0

b_points = tf.stack(tf.map_fn(fn = lambda x: boundary(x), elems=unif, dtype = (tf.float32, tf.float32)), axis=1)
u_train = tf.map_fn(fn = lambda x: sol_bound(x), elems=b_points, dtype = (tf.float32))


#Generate the exact solution with FDM

#print(u_train)
#print(b_points)


#collocation points
x_grid  = tf.random.uniform(shape=[num_collocation_points], maxval = w).numpy()
t_grid  = tf.random.uniform(shape=[num_collocation_points], maxval = T).numpy()


# model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1)
])

mu = tf.constant(mu_exact)

trainable_variables = model.trainable_variables  

# loss functions
def PDE(x, t):
    with tf.GradientTape(persistent = True) as tape:
        tape.watch(x)
        tape.watch(t)
        u = model(tf.stack((x,t),axis=1))
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        u_xx = tape.gradient(u_x, x)
    return u_t - mu*(u_xx) + tf.reshape(u**3 - u,(num_collocation_points,))

last_loss_fit = tf.constant([0.0])
def loss_fit():
    global last_loss_fit
    last_loss_fit = tf.reduce_mean(tf.square(model(b_points) - u_train[:,None]))
    return last_loss_fit
last_loss_PDE = tf.constant([0.0])
def loss_PDE():
    global last_loss_PDE
    last_loss_PDE = tf.reduce_mean(tf.square(PDE(tf.constant(x_grid), tf.constant(t_grid))))
    return last_loss_PDE


#%%%%%%%%%%%%%%%%%% Training
# initialize optimizer
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)

# optimization loop
for i in range(num_epochs):
    with tf.GradientTape() as tape:
        loss = loss_fit() + loss_PDE()
        grads = tape.gradient(loss, trainable_variables)
        opt.apply_gradients(zip(grads, trainable_variables))

    if i % 100 == 0:
        print('iter = %d, mu = %f, loss_fit = %f, loss_PDE = %f' %
              (i, mu, last_loss_fit.numpy(), last_loss_PDE.numpy()))


# Sample the solution at 501 x 501 points
x_test = np.linspace(0, w, 501)
t_test = np.linspace(0, T, 501)
xx, tt = np.meshgrid(x_test, t_test)
test_points = np.stack([xx.ravel(), tt.ravel()], axis=1)

# Predict using the trained model
u_pred = model.predict(test_points).reshape(501, 501)

# Save as a compressed file
np.savez("saved_solution.npz", x=x_test, t=t_test, u=u_pred)
print("Save successful!")


#%%%%%%%%%%%%%%%%%% Post-processing
nt = 10000
nx = 49
dx = w / nx
dt = T / nt

dx2 = dx**2

def do_timestep(u0, u):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1] = u0[1:-1] + mu_exact* dt *(u0[2:] - 2.0*u0[1:-1] + u0[:-2])/dx2 - dt * (np.power(u0[1:-1],3) + u0[1:-1])
    u0 = u.copy()
    return u0, u

u0 = np.zeros(nx + 1)
u = u0.copy()

for i in range(nx + 1):
    if (i*dx - w/2)**2 < l**2/4:
        u0[i] = 1.0



#t_test = [100, 500, 2000, 5000]
#fig = plt.figure()
#fignum = 1
#xx = np.linspace(0,w, num=nx + 1, dtype=np.float32)
#for i in range(nt + 1): #To complete
#    u0, u = do_timestep(u0, u)
#    if i in t_test:
#        test_points = tf.stack((xx, np.full((nx + 1), i*dt)), axis = 1)
#        ax = fig.add_subplot(220 + fignum)
#        ax.plot(xx,  model(test_points).numpy(), 'r-')
#        ax.plot(xx, u.copy(), 'k-')
#        ax.set_title('{:.1f} ms'.format(i*1000*dt))
#        ax.set_ylim([-0.1, 1.5])
#        fignum += 1
#plt.tight_layout(h_pad=2)
#plt.show()
#plt.show()
#plt.savefig("../../figure/forward_1D/parabolic_nonlinear.png")


# Space-time grid
nx_plot, nt_plot = 501, 501  # 501x501 points
x_vals = np.linspace(0, w, nx_plot)
t_vals = np.linspace(0, T, nt_plot)

# Create meshgrid of (x, t)
X, T_mesh = np.meshgrid(x_vals, t_vals)
X_flat, T_flat = X.flatten(), T_mesh.flatten()

# Evaluate the model at these (x, t) points
test_points = tf.stack((X_flat, T_flat), axis=1)
U_pred = model(test_points).numpy().reshape(nt_plot, nx_plot)  # Reshape back to 2D

# Plot results
'''plt.figure(figsize=(8, 6))
plt.imshow(U_pred, extent=[0, w, 0, T], origin="lower", aspect="auto")
plt.colorbar(label="u(x, t)")
plt.xlabel("Space (x)")
plt.ylabel("Time (t)")
plt.title("Solution u(x, t) in Space-Time")
plt.show()'''
import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# Match PINN resolution
nx, nt = 501, 501  
x_vals = np.linspace(0, w, nx)  
t_vals = np.linspace(0, T, nt)  
dx = x_vals[1] - x_vals[0]
dx2 = dx**2
dt = min(0.5 * dx2 / (2 * mu_exact), T / nt)  # More stability

# Initialize solution array (501 × 501)
u_fd = np.zeros((nt, nx))

# Initial condition: Step function
for i in range(nx):
    if (x_vals[i] - w/2) ** 2 < l**2 / 4:
        u_fd[0, i] = 1.0

# Finite Difference Time Evolution (Explicit)
for n in range(nt - 1):
    u_next = u_fd[n].copy()
    
    # Explicit scheme for the diffusion and reaction term
    u_next[1:-1] = (
        u_fd[n, 1:-1] + mu_exact * dt * (u_fd[n, 2:] - 2.0 * u_fd[n, 1:-1] + u_fd[n, :-2]) / dx2
        - dt * (u_fd[n, 1:-1]**3 - u_fd[n, 1:-1])
    )
    
    # Enforce Boundary Conditions
    u_next[0] = u_next[1]  # Neumann (no flux at x=0)
    u_next[-1] = u_next[-2]  # Neumann (no flux at x=w)
    
    u_fd[n + 1] = u_next

# Save FDM solution
np.savez("fdm_solution.npz", x=x_vals, t=t_vals, u=u_fd)

print("FDM solution saved with shape:", u_fd.shape)
