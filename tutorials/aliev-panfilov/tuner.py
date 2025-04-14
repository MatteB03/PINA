# imports
import torch
import warnings
import torch

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from pina import Trainer
from pina.model import FeedForward
from pina.solver import PINN, RBAPINN
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operator import grad, laplacian
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue, FixedGradient
import matplotlib.pyplot as plt

from pina import Trainer
from pina.solver import PINN, RBAPINN, SelfAdaptivePINN as SAPINN
from pina.model import FeedForward
from pina.callback import MetricTracker, PINAProgressBar, SwitchOptimizer
from pina.loss import LpLoss

from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

warnings.filterwarnings('ignore')

############################################################################

# defining the problem to use in the tuner
class TimeSpaceProblem(TimeDependentProblem, SpatialProblem):
    output_variables = ["V", "W"]
    spatial_domain = CartesianDomain({"x": [0.1, 20]})
    Tf = 70
    temporal_domain = CartesianDomain({"t": [1, Tf]})

    # defining the ode equation
    def aliev_panfilov_equation1(input_, output_):
        # computing the derivative and laplacian using functions from pina.operator
        V_t = grad(output_, input_, components=["V"], d=["t"])
        nabla_V = laplacian(output_, input_, components=["V"], d=["x"])

        # extracting the u input variable
        V = output_.extract(["V"])
        W = output_.extract(["W"])
        
        # paper data, reference here: https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2021.768419/full#supplementary-material
        k = 8
        a = 0.01 
        D = 0.1  

        # extract is the standard procedure to use variables in ODE's/PDE's
        t = input_.extract(["t"])
        x = input_.extract(["x"])

        res_V = (
            (-V_t) + D * nabla_V - k * V * (V - a) * (V - 1) - V * W
        )  
        return res_V

    def aliev_panfilov_equation2(input_, output_):
        # computing the derivative
        W_t = grad(output_, input_, components=["W"], d=["t"])

        # extracting the u input variable
        V = output_.extract(["V"])
        W = output_.extract(["W"])

        # paper data
        k = 8
        eps = 0.002
        mu1 = 0.2  
        mu2 = 0.3
        b = 0.15  
        res_W = W_t - ((eps + mu1 * W / (V + mu2)) * (-W - k * V * (V - b - 1)))
        return res_W

    # defining initial conditions
    def initial_cond_V(input_, output_):
        x = input_.extract(["x"])
        return output_.extract(["V"]) - torch.exp(-0.8 * x) 
    
    def initial_cond_W(input_, output_):
        x = input_.extract(["x"])
        return output_.extract(["W"]) - 2.5 * torch.exp(-0.8 * x)  # we rescale the equation for the PINN to correctly learn the intensity of the predicted W
    
    # conditions to hold
    # remember to specify the fixed gradient direction (ex x,y...)
    conditions = {
        "gamma1": Condition(
            domain=CartesianDomain({"x": 0.1, "t": [1, Tf]}),
            equation=FixedGradient(0.0, components=["V"], d=["x"]),
        ),
        "gamma2": Condition(
            domain=CartesianDomain({"x": 20, "t": [1, Tf]}),
            equation=FixedGradient(0.0, components=["V"], d=["x"]),
        ),
        "t0_V": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": 1}),
            equation=Equation(initial_cond_V),
        ),
        "t0_W": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": 1}),
            equation=Equation(initial_cond_W),
        ),
        "D_V": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": [1, Tf]}),
            equation=Equation(aliev_panfilov_equation1),
        ),
        "D_W": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": [1, Tf]}),
            equation=Equation(aliev_panfilov_equation2))
        }

# we now assign the problem and discretise its domains 
problem = TimeSpaceProblem()
# discretise according to the importance and extension of each condition, feel free to change the values here
problem.discretise_domain(4000, "random", domains=["D_V", "D_W"])
problem.discretise_domain(140, "grid", domains=["t0_V", "t0_W"])
problem.discretise_domain(200, "grid", domains=["gamma2", "gamma1"])

############################################################################

# we now define the function we'll be using for the tuning
# you can see it's almost identical to what we already do in the tutorial.py,
# the difference is that here you pass the function the parameters you intend to test 
def train_pinn(config):
    print(f"Running train_pinn with config: {config}")  # Debugging output
    problem = TimeSpaceProblem()
    problem.discretise_domain(4000, "random", domains=["D_V", "D_W"])
    problem.discretise_domain(140, "grid", domains=["t0_V", "t0_W"])
    problem.discretise_domain(200, "grid", domains=["gamma2", "gamma1"])

    class M2TUNE(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.space_w = FeedForward(
                n_layers=config["n_layers"], inner_size=32, #you can see that we use a dict as function input
                func=torch.nn.Tanh, input_dimensions=1, output_dimensions=1
            )
            self.space_v = FeedForward(
                n_layers=config["n_layers"], inner_size=32,
                func=torch.nn.Tanh, input_dimensions=1, output_dimensions=1
            )
            self.coeff_time = torch.nn.Parameter(torch.tensor(config["coeff_time"], dtype=torch.float32))
            self.v_shift = config["v_shift"]
            self.w_shift = config["w_shift"]

        def forward(self, x):
            x_ = x.extract(["x"])
            v = self.space_v(x_ - self.coeff_time * (x.extract(["t"]) - self.v_shift))
            w = self.space_w(x_ - self.coeff_time * (x.extract(["t"]) - self.w_shift))
            v.labels = ["V"]
            w.labels = ["W"]
            return torch.cat([v, w], dim=1)

    print("Initializing model...")
    model = M2TUNE(config)  # FIX: Pass config properly

    print("Initializing solver...")
    solver = config['solver'](problem, model)

    print("Starting training...")
    trainer = Trainer(
        solver,
        max_epochs=5000,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=0.7,
        log_every_n_steps=0,
        callbacks=[TuneReportCallback(metrics={"loss": "val_loss"},
                                      on="train_end")],# when using test_loss remember to do it on test_end, been there before
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )
    trainer.train()
    print("Training complete.")

############################################################################

# when using ray write the parameters you want to test in a config dict,
# where the keys are the params and the items are lists of possible values    
config = {
    "coeff_time": tune.choice([0.5,1]),
    "n_layers": tune.choice([3,4]),
    "solver" : tune.choice([PINN, RBAPINN]),
    "v_shift":tune.choice([0,10,20]),
    "w_shift":tune.choice([0,10,20]),
    }

# this is the crucial part, where you set how you want the run to be executed
tune_analysis = tune.run(
    tune.with_parameters(train_pinn),
    resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust resources as needed
    metric="loss",
    mode="min",
    config=config,
    num_samples=72,  # Number of trials to run
    scheduler=ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)  # ASHA for early stopping
)
best_config = tune_analysis.get_best_config(metric="loss", mode="min")
print("Best hyperparameters found were: ", best_config)