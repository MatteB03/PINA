# imports
import warnings
import torch

warnings.filterwarnings('ignore')

from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operator import grad, laplacian
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
import matplotlib.pyplot as plt

from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
from pina.callback import MetricTracker

from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

############################################################################

# defining the problem
class TimeSpaceProblem(TimeDependentProblem, SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    # defining the ode equation
    def equation_4p5(input_, output_):
        # computing the derivative
        u_t = grad(output_, input_, components=["u"], d=["t"])
        # computing the laplacian
        nabla_u = laplacian(output_, input_, components=["u"], d=["x"])
        u = output_.extract(["u"])
        gamma = 2

        res = u_t - gamma * nabla_u + u**3 - u
        return res
    
    # conditions to hold
    # specify the fixed gradient direction (ex x,y...)
    conditions = {
        #we implement the initial condition (step function) as 3 separate FixedValue equations
        "step1": Condition(
            domain=CartesianDomain({"x": [0,0.249], "t": 0}),
            equation=FixedValue(0)),
        "step2": Condition(
            domain=CartesianDomain({"x": [0.25,0.75], "t": 0}),
            equation=FixedValue(1)),
        "step3": Condition(
            domain=CartesianDomain({"x": [0.751,1], "t": 0}),
            equation=FixedValue(0)),
        "L_u": Condition(
            domain=CartesianDomain({"x":[0,1],"t":[0,1]}),
            equation=Equation(equation_4p5)),
        "u0":Condition(
            domain=CartesianDomain({"x":0,"t":[0,1]}),
            equation= FixedValue(0)),
        "u1":Condition(
            domain=CartesianDomain({"x":1,"t":[0,1]}),
            equation= FixedValue(0))
        }

# we now instantiate the problem and discretise its domains 
problem = TimeSpaceProblem()

problem.discretise_domain(4000, "random", domains=["L_u"])
problem.discretise_domain(250, "grid", domains=["step1", "step3"])
problem.discretise_domain(501, "grid", domains=["step2"])
problem.discretise_domain(1001, "grid", domains=["u0","u1"])

############################################################################

# building and training the model

# setting the seed for reproducibility
seed_everything(42, workers=True)

# build the model
model = FeedForward(
    layers=[16,16,16,16],
    func=torch.nn.Tanh, 
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables),
)

# build the solver
pinn = PINN(
    problem,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=5e-5, weight_decay=0), ##try varying lr
    loss=torch.nn.MSELoss()
)  # when using optimizers import them from pina.optim, the only mandatory argument is an instance of torch.optim (ex. Adam, AdamW)

trainer = Trainer(
    solver=pinn,
    max_epochs= 20000,
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

############################################################################

# saving plot samples

import torch
import numpy as np
import matplotlib.pyplot as plt

input_test = CartesianDomain({"x": [0, 1], "t": [0, 1]}).sample(50, mode="grid")
output_test = pinn(input_test)
torch.save(output_test, "pinn_output_post_tuning.pt")
