# here we try using the data as is done in the thesis to see how it affects the result
# imports
import sys
import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

import numpy as np
import tensorflow as tf
import time
import warnings
import torch

warnings.filterwarnings('ignore')

from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operator import grad, laplacian
from pina import Condition, LabelTensor
from pina.domain import CartesianDomain, EllipsoidDomain, Difference
from pina.equation import Equation, FixedValue, FixedGradient
import matplotlib.pyplot as plt

from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward

from pina.optim import TorchOptimizer
from pytorch_lightning import seed_everything
seed_everything(42, workers=True)

############################################################################

# Here we create the custom geometries we'll be using as initial condition
# The sampling method in the thesis consists of taking more points in a region 
# very near the activation front, we do something analogue by sampling a lot of points on the 
# border at t=[0,0] (not t=0 because later domain discretization would drop the "t" variable)
circle_border = EllipsoidDomain({"x":[25,75], "y":[25,75], "t":[0,0]}, sample_surface=True)
circle = EllipsoidDomain({"x":[25,75], "y":[25,75], "t":[0,0]})
outside_circle = Difference([CartesianDomain({"x": [0, 100],"y": [0, 100], "t": [0,0]}),circle])

if len(sys.argv) >= 2:
    rho = float(sys.argv[1])
else:
    rho = 1.0

#Parameters of the model
num_training_samples = 1000
num_testing_samples = 10000
num_epochs              = 2000
learning_rate           = 1e-3
num_collocation_points  = 5000

#Collocation band width
coll_width = 5.0

#Parameters of the problem
# plate size, mm
w = h = 100
# intervals in x-, y- directions, mm
dx = dy = 0.5
# diffusivity,
D_true = 12.9*0.25  #mm/ms  12.9 -> AP model
#Time horizon
time_max = 2.0
dt = 0.01
nx, ny, nt = int(w/dx), int(h/dy), int(time_max/dt)
dx2, dy2 = dx*dx, dy*dy
Tcool = 0.0
Thot = 1.0
K_true = 12.9 * 2.0    # 12.9 -> AP model

#Initialization for the FDM
u_train = np.zeros((num_training_samples))
u_fdm   = np.full((nx + 1, ny + 1, nt + 1), Tcool)
#print(u_fdm.shape)

#Initial condition
r, cx, cy = 25, 50, 50
r2 = r**2
for i in range(nx + 1):
    for j in range(ny + 1):
        p2 = (i*dx-cx)**2 + (j*dy-cy)**2
        if p2 < r2:
            u_fdm[i,j,0] = Thot

def do_timestep(u0):
    # Propagate with forward-difference in time, central-difference in space
    u = np.full((nx + 1, ny + 1), Tcool)
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D_true * dt * (
          (u0[2:, 1:-1] - 2.0*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
          + (u0[1:-1, 2:] - 2.0*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 ) + dt * K_true * np.multiply(np.multiply(u0[1:-1, 1:-1],(u0[1:-1, 1:-1]-0.1)),(1.0 - u0[1:-1, 1:-1]))
    return u

x_train = tf.random.uniform(shape=[num_training_samples], maxval=w).numpy()
y_train = tf.random.uniform(shape=[num_training_samples], maxval=h).numpy()
t_train = np.sort(tf.random.uniform(shape=[num_training_samples], maxval=time_max).numpy())

def lin_interpolate(f, index, increment, point):
    #Linearly interpolate the 3D function goven the values in a grid around the point
    #print(f.shape)
    coord   = (point - np.multiply(index, increment))/increment
    i_coord = np.ones(3) - coord

    return f[1,1,1]*coord[0]*coord[1]*coord[2] \
        + f[0,1,1]*i_coord[0]*coord[1]*coord[2] \
        + f[1,0,1]*coord[0]*i_coord[1]*coord[2] \
        + f[1,1,0]*coord[0]*coord[1]*i_coord[2] \
        + f[0,0,1]*i_coord[0]*i_coord[1]*coord[2] \
        + f[0,1,0]*i_coord[0]*coord[1]*i_coord[2] \
        + f[1,0,0]*coord[0]*i_coord[1]*i_coord[2] \
        + f[0,0,0]*i_coord[0]*i_coord[1]*i_coord[2]

for k in range(nt):
    u_fdm[:,:,k+1] = do_timestep(u_fdm[:,:,k])
    # Find the t_train values in here and approximate
    for m in range(0, num_training_samples):
        if t_train[m] < (k+1)*dt and t_train[m] >= k*dt:
            i, j = int(x_train[m]/dx), int(y_train[m]/dy)
            u_train[m] = lin_interpolate(u_fdm[i:(i+2), j:(j+2), k:(k+2)], (i,j,k), (dx,dy,dt), np.array([x_train[m], y_train[m], t_train[m]]))

#Train on the grid
xi = np.random.choice(nx + 1, size=num_testing_samples)
yi = np.random.choice(ny + 1, size=num_testing_samples)
ti = np.random.choice(nt + 1, size=num_testing_samples)

u_test = u_fdm[xi,yi,ti]

xyt_test = tf.stack((xi*dx, yi*dy, ti*dt), axis=1)

print(u_test.shape)
print(xyt_test.shape)


xyt_test_np = xyt_test.numpy()  # Only works in eager mode (which you’re in)
input_points = LabelTensor(xyt_test_np, labels=['x', 'y', 't'])

u_test = u_test.reshape(-1, 1)  # (10000, 1)
output_points = LabelTensor(u_test, labels=['u'])

class MonodomainProblem(TimeDependentProblem, SpatialProblem):
    
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, L],"y": [0, L]})
    temporal_domain = CartesianDomain({"t": [0, T]})
    
    # defining the ode equation
    def monodomain_equation(input_, output_):
        # computing the derivative
        u_t = grad(output_, input_, components=["u"], d=["t"])
        u_xx = laplacian(output_, input_, components=["u"], d=["x"])
        u_yy = laplacian(output_, input_, components=["u"], d=["y"])
        u = output_.extract(["u"])
        D = 3.225 ##should be correct, 12.9 * 0.25
        K = 25.8
        alpha = 0.1 #investigate further ##should be correct
        res = u_t - D * (u_xx + u_yy) - K* u * (1-u) * (u - alpha)
        return res
    
    # conditions to hold
    conditions = {
        "L_u": Condition(
            domain=CartesianDomain({"x":[0,100],"y":[0,100],"t":[0,2]}),
            equation=Equation(monodomain_equation)),
        "data": Condition(
            input=input_points,
            target=output_points),
        "u_1": Condition(
            domain= circle,
            equation= FixedValue(1)),
        "u_1_border": Condition(
            domain= circle_border,
            equation= FixedValue(1)),
        "u_0": Condition(
            domain= outside_circle,
            equation= FixedValue(0)),
        }

problem = MonodomainProblem()
problem.discretise_domain(mode="grid", domains=["L_u"], 
                          sample_rules={'x':{'n': 100, 'mode':'grid'},
                                        'y':{'n': 100, 'mode':'grid'},
                                        't':{'n': 20,'mode':'grid'}})
problem.discretise_domain(500, "random", domains=["u_1"])
problem.discretise_domain(500, "random", domains=["u_0"])
problem.discretise_domain(2000, "random", domains=["u_1_border"])



model = FeedForward(
    layers=[10,20,20,10],
    func=torch.nn.Tanh, 
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables),
    )

from pina.loss import ScalarWeighting
K_weight = 8
rho1 = 10/K_weight
rho2 = 10/(K_weight)**2
print(rho1,rho2)
pinn = PINN(
    problem,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=1e-3, weight_decay=0),
    weighting=ScalarWeighting({"x_bound_0_loss":rho1, "x_bound_1_loss":rho1,
                               "y_bound_0_loss":rho1, "y_bound_1_loss":rho1,
                               "L_u_loss":rho2, "u_0_loss":rho2,
                               "u_1_loss":rho2, "u_1_border_loss": rho2,
                               "u_0_2_loss":rho2,
                               "u_1_2_loss":rho2, "u_1_2_border_loss": rho2
                                }),
    loss=torch.nn.MSELoss(),
)  

#from lightning.pytorch.loggers import TensorBoardLogger # if you want to log losses; to see how to visualize them look tutorial1, 2 or 3
trainer = Trainer(
    solver=pinn,
    max_epochs= 2000,
    accelerator="cpu", ##gpu not available in SISSA work stations (at least not on mine :( )
    #logger=TensorBoardLogger(save_dir="training_logs"),
    enable_model_summary=False,
    gradient_clip_val=0.7,
    train_size=0.9,
    val_size=0.0,
    test_size=0.1
)  # , accelerator='cpu', enable_model_summary=False) # we train on CPU and avoid model summary at beginning of training (optional)

# train
trainer.train()
trainer.test()

import numpy as np

time_points = [0.1, 0.5, 1.0, 1.5]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, t_val in enumerate(time_points):
    
    input_test = CartesianDomain({"x": [0, L], "y": [0, L], "t": t_val}).sample(200, mode="grid")
    output_test = pinn(input_test)
    u = output_test.extract(["u"]).detach().numpy()
    x = input_test.extract(["x"]).detach().numpy()
    y = input_test.extract(["y"]).detach().numpy()

    sc = axes[idx].scatter(x, y, c=u, s=5)
    axes[idx].set_title(f"u(x, y, t={t_val})")
    axes[idx].set_xlabel("x")
    axes[idx].set_ylabel("y")
    axes[idx].set_aspect('equal')
    fig.colorbar(sc, ax=axes[idx])

plt.tight_layout()
plt.show()
