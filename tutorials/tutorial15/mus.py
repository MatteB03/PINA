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

from lightning.pytorch.loggers import TensorBoardLogger
from pina import Trainer
from pina.solver import PINN, RBAPINN, SelfAdaptivePINN as SAPINN
from pina.model import FeedForward
from pina.callback import MetricTracker, PINAProgressBar, SwitchOptimizer
from pina.loss import LpLoss

from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

import matplotlib.pyplot as plt

# List to store results for each mu
results = []

mus=[0.1,0.2,0.3,0.5,0.8,1,1.2,1.5,2,4,5,10] 

for value in mus:
    class TimeSpaceProblem(TimeDependentProblem, SpatialProblem):
        output_variables = ["u"]
        spatial_domain = CartesianDomain({"x": [0, 1]})
        temporal_domain = CartesianDomain({"t": [0, 1]})

        # Defining the PDE equation
        def equation_4p5(input_, output_):
            u_t = grad(output_, input_, components=["u"], d=["t"])
            nabla_u = laplacian(output_, input_, components=["u"], d=["x"])
            u = output_.extract(["u"])
            mu = value

            res = u_t - mu * nabla_u + u**3 - u
            return res
    
        # Conditions
        conditions = {
            "step1": Condition(
                domain=CartesianDomain({"x": [0, 0.249], "t": 0}),
                equation=FixedValue(0)),
            "step2": Condition(
                domain=CartesianDomain({"x": [0.25, 0.75], "t": 0}),
                equation=FixedValue(1)),
            "step3": Condition(
                domain=CartesianDomain({"x": [0.751, 1], "t": 0}),
                equation=FixedValue(0)),
            "L_u": Condition(
                domain=CartesianDomain({"x": [0, 1], "t": [0, 1]}),
                equation=Equation(equation_4p5)),
            "u0": Condition(
                domain=CartesianDomain({"x": 0, "t": [0, 1]}),
                equation=FixedValue(0)),
            "u1": Condition(
                domain=CartesianDomain({"x": 1, "t": [0, 1]}),
                equation=FixedValue(0))
        }

    problem = TimeSpaceProblem()
    problem.discretise_domain(4000, "random", domains=["L_u"])
    problem.discretise_domain(250, "grid", domains=["step1", "step3"])
    problem.discretise_domain(501, "grid", domains=["step2"])
    problem.discretise_domain(1001, "grid", domains=["u0", "u1"])

    # Setting the seed for reproducibility
    seed_everything(42, workers=True)

    # Build the model
    model = FeedForward(
        layers=[16, 16, 16, 16],
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

    trainer = Trainer(
        solver=pinn,
        max_epochs=10000,
        accelerator="cpu",
        #logger=TensorBoardLogger(save_dir="training_logs"),
        enable_model_summary=False,
        callbacks=[MetricTracker()],
        train_size=0.9,
        val_size=0.0,
        test_size=0.1
    )

    # Train
    trainer.train()
    trainer.test()

    # Test the trained model
    input_test = CartesianDomain({"x": [0, 1], "t": [0, 1]}).sample(501, mode='grid')
    output_test = pinn(input_test)
    u = output_test.extract(["u"])

    # Store the results
    results.append((value, input_test, u))

# Create subplots
fig, axes = plt.subplots(3, 4, figsize=(24, 16))
axes = axes.flatten()

# Plot results
for i, (mu, input_test, u) in enumerate(results):
    ax = axes[i]
    scatter = ax.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=u.detach().numpy())
    ax.set_title(f"mu = {mu}")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    fig.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()
