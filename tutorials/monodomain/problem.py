import warnings
import torch

warnings.filterwarnings('ignore')

from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operator import grad, laplacian
from pina import Condition
from pina.domain import CartesianDomain, EllipsoidDomain, Difference
from pina.equation import Equation, FixedValue
import matplotlib.pyplot as plt

from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.callback import MetricTracker

from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

L = 100
T = 2.0

class MonodomainProblem(TimeDependentProblem, SpatialProblem):
    
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, L],"y": [0, L]})
    temporal_domain = CartesianDomain({"t": [0, T]})
    circle = EllipsoidDomain({"x":[L/4,3*L/4], "y":[L/4,3*L/4], "t":[0,0]})
    outside_circle = Difference([CartesianDomain({"x": [0, L],"y": [0, L], "t": [0,0]}),circle])
    
    # defining the ode equation
    def monodomain_equation(input_, output_):
        # computing the derivative
        u_t = grad(output_, input_, components=["u"], d=["t"])
        nabla_u = laplacian(output_, input_, components=["u"], d=["x"])
        u = output_.extract(["u"])
        D = 3.225
        K = 25.8
        alpha = 0.1 #investigate further
        res = u_t - D * nabla_u - K* u * (1-u) * (u - alpha)
        return res
    
    def bound_cond_1(input_,output_):
        D = 3.225
        u_x = grad(output_, input_, components=["u"], d=["x"])
        return D * u_x
    
    def bound_cond_2(input_,output_):
        D = 3.225
        u_y = grad(output_, input_, components=["u"], d=["y"])
        return D * u_y
    
    # conditions to hold
    # specify the fixed gradient direction (ex x,y...)
    conditions = {
        "L_u": Condition(
            domain=CartesianDomain({"x":[0,L],"y":[0,L],"t":[0,T]}),
            equation=Equation(monodomain_equation)),
        "y_bound_0":Condition(
            domain=CartesianDomain({"x":0,"y":[0,L],"t":[0,T]}),
            equation= Equation(bound_cond_1)),
        "y_bound_1":Condition(
            domain=CartesianDomain({"x":1,"y":[0,L],"t":[0,T]}),
            equation= Equation(bound_cond_1)),
        "x_bound_0":Condition(
            domain=CartesianDomain({"x":[0,L],"y":0,"t":[0,T]}),
            equation= Equation(bound_cond_2)),
        "x_bound_1":Condition(
            domain=CartesianDomain({"x":[0,L],"y":0,"t":[0,T]}),
            equation= Equation(bound_cond_2)),
        "u_1": Condition(
            domain= circle,
            equation= FixedValue(1)),
        "u_0": Condition(
            domain= outside_circle,
            equation= FixedValue(0)
        )
        }


problem = MonodomainProblem()

problem.discretise_domain(mode= "grid", domains=["x_bound_0", "x_bound_1"],
                          sample_rules={'x':{'n': 200, 'mode':'grid'},
                                        'y':{'n': 1, 'mode':'grid'},
                                        't':{'n':20, 'mode':'grid'}})
problem.discretise_domain(mode= "grid", domains=["y_bound_0", "y_bound_1"],
                          sample_rules={'x':{'n': 1, 'mode':'grid'},
                                        'y':{'n': 200, 'mode':'grid'},
                                        't':{'n': 20, 'mode':'grid'}})
problem.discretise_domain(200, "random", domains=["u_0"])
problem.discretise_domain(600, "random", domains=["u_1"])
problem.discretise_domain(mode="grid", domains=["L_u"], 
                          sample_rules={'x':{'n': 100, 'mode':'grid'},
                                        'y':{'n': 100, 'mode':'grid'},
                                        't':{'n': 20,'mode':'grid'}})

# setting the seed for reproducibility
seed_everything(42, workers=True)

# build the model
model = FeedForward(
    layers=[16,16,16,16],
    func=torch.nn.Tanh, 
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables),
)
pinn = PINN(
    problem,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=1e-4, weight_decay=0),
    loss=torch.nn.MSELoss()
)  

from lightning.pytorch.loggers import TensorBoardLogger
trainer = Trainer(
    solver=pinn,
    max_epochs= 5000,
    accelerator="cpu",
    logger=TensorBoardLogger(save_dir="training_logs"),
    enable_model_summary=False,
    gradient_clip_val=0.7,
    callbacks=[MetricTracker()],
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
