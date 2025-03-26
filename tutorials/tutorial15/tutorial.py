import warnings
import torch
import scipy.io as io
import numpy as np
from scipy.io import loadmat

warnings.filterwarnings('ignore')

from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operator import grad, laplacian
from pina import Condition, LabelTensor
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue, FixedGradient
import matplotlib.pyplot as plt

from pina import Trainer
from pina.solver import PINN, RBAPINN, SelfAdaptivePINN as SAPINN
from pina.model import FeedForward
from pina.callback import MetricTracker, PINAProgressBar, SwitchOptimizer
from pina.loss import LpLoss

from tsproblem import TimeSpaceProblem
from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

problem = TimeSpaceProblem()

problem.discretise_domain(4000, "random", domains=["L_u"])
problem.discretise_domain(250, "grid", domains=["step1", "step3"])
problem.discretise_domain(251, "grid", domains=["step2"])
problem.discretise_domain(101, "grid", domains=["u0","u1"])

#print("Input points:", problem.input_pts)

# from pina import Plotter

# pl = Plotter()
# pl.plot_samples(problem=problem)#,variables='spatial')


# setting the seed for reproducibility
seed_everything(42, workers=True)

# build the model
model = FeedForward(
    layers=[16, 16, 16, 16],
    func=torch.nn.Tanh,  # Tanh,
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
    max_epochs=10000,#logger=TensorBoardLogger('tutorial_logs'),
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

input_test = CartesianDomain({"x": [0, 1], "t": [0,1]}).sample(101, mode='grid')
#input_test = CartesianDomain({"x": [0.1, 20], "t": [1, 70]}).sample(100, mode='grid')
output_test = pinn(input_test)
u = output_test.extract(["u"])

plt.figure(figsize=(12, 10))
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=u.detach().numpy(), cmap='coolwarm')
plt.colorbar()
plt.title('u')
plt.grid()
plt.tight_layout() 
plt.show()