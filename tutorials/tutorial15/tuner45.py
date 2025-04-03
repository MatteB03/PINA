import warnings
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

warnings.filterwarnings('ignore')
from lightning.pytorch.loggers import TensorBoardLogger
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

from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

class TimeSpaceProblem(TimeDependentProblem, SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    # defining the ode equation
    def equation_4p5(input_, output_):
        # computing the derivative
        u_t = grad(output_, input_, components=["u"], d=["t"])
        nabla_u = laplacian(output_, input_, components=["u"], d=["x"])
        u = output_.extract(["u"])
        gamma = 2

        res = u_t - gamma * nabla_u + u**3 - u
        return res
    
    # conditions to hold
    # specify the fixed gradient direction (ex x,y...)
    conditions = {
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

def Train_PINN(config):
    problem = TimeSpaceProblem()
    problem.discretise_domain(4000, "random", domains=["L_u"])
    problem.discretise_domain(250, "grid", domains=["step1", "step3"])
    problem.discretise_domain(501, "grid", domains=["step2"])
    problem.discretise_domain(1001, "grid", domains=["u0","u1"])
    seed_everything(42, workers=True)
    model = FeedForward(
    layers=[16,16,16,16],
    func=torch.nn.Tanh,  # Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables))
    solver = PINN(
    problem,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=config["lr"], weight_decay=0),
    loss=torch.nn.MSELoss()
)  
    trainer= Trainer(solver, max_epochs=config["epochs"],accelerator="cpu",
    logger=TensorBoardLogger(save_dir="training_logs"),
    enable_progress_bar=False,
    enable_model_summary=False,
    gradient_clip_val=0.7,
    callbacks=[TuneReportCallback(metrics={"loss": "test_loss"}, on="test_end")],
    train_size=0.9,
    val_size=0.0,
    test_size=0.1
)  
    trainer.train()
    trainer.test()
config = {
    "lr":tune.choice([1e-4,2.5e-4,5e-4,1e-3,2.5e-3,5e-3]),
    "epochs":tune.choice([5000,7500,10000,12500,15000])
    }

tune_analysis = tune.run(
    tune.with_parameters(Train_PINN),
    resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust resources as needed
    metric="loss",
    mode="min",
    config=config,
    num_samples=30,  # Number of trials to run
    scheduler=ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)  # ASHA for early stopping
)
best_config = tune_analysis.get_best_config(metric="loss", mode="min")
print("Best hyperparameters found were: ", best_config)