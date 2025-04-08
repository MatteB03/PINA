import warnings
import torch
import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

warnings.filterwarnings('ignore')

from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operator import grad, laplacian
from pina import Condition
from pina.domain import CartesianDomain, EllipsoidDomain, Difference
from pina.equation import Equation, FixedValue, FixedGradient
from pina.loss import ScalarWeighting
import matplotlib.pyplot as plt

from pina import Trainer
from pina.solver import PINN
from pina.model import FeedForward
import random
from pina.optim import TorchOptimizer

from lightning.pytorch.loggers import TensorBoardLogger

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
        u_xx = laplacian(output_, input_, components=["u"], d=["x"])
        u_yy = laplacian(output_, input_, components=["u"], d=["y"])
        u = output_.extract(["u"])
        D = 3.225 ##should be correct, 12.9 * 0.25
        K = 25.8
        alpha = 0.1 #investigate further ##should be correct
        res = u_t - D * (u_xx + u_yy) - K* u * (1-u) * (u - alpha)
        return res
    
    '''def bound_cond_1(input_,output_):
        D = 3.225
        u_x = grad(output_, input_, components=["u"], d=["x"])
        return D * u_x
    
    def bound_cond_2(input_,output_):
        D = 3.225
        u_y = grad(output_, input_, components=["u"], d=["y"])
        return D * u_y'''
    
    # conditions to hold
    # specify the fixed gradient direction (ex x,y...)
    conditions = {
        "L_u": Condition(
            domain=CartesianDomain({"x":[0,L],"y":[0,L],"t":[0,T]}),
            equation=Equation(monodomain_equation)),
        "y_bound_0":Condition(
            domain=CartesianDomain({"x":0,"y":[0,L],"t":[0,T]}),
            equation= FixedGradient(0.0, components=["u"], d=["x"])),
        "y_bound_1":Condition(
            domain=CartesianDomain({"x":1,"y":[0,L],"t":[0,T]}),
            equation= FixedGradient(0.0, components=["u"], d=["x"])),
        "x_bound_0":Condition(
            domain=CartesianDomain({"x":[0,L],"y":0,"t":[0,T]}),
            equation= FixedGradient(0.0, components=["u"], d=["y"])),
        "x_bound_1":Condition(
            domain=CartesianDomain({"x":[0,L],"y":1,"t":[0,T]}),
            equation= FixedGradient(0.0, components=["u"], d=["y"])),
        "u_1": Condition(
            domain= circle,
            equation= FixedValue(1)),
        "u_0": Condition(
            domain= outside_circle,
            equation= FixedValue(0)
        )
        }
def Train_PINN(config):

    problem = MonodomainProblem()

    problem.discretise_domain(mode= "grid", domains=["x_bound_0", "x_bound_1"],
                          sample_rules={'x':{'n': 100, 'mode':'grid'},
                                        'y':{'n': 1, 'mode':'grid'},
                                        't':{'n':20, 'mode':'grid'}})
    problem.discretise_domain(mode= "grid", domains=["y_bound_0", "y_bound_1"],
                          sample_rules={'x':{'n': 1, 'mode':'grid'},
                                        'y':{'n': 100, 'mode':'grid'},
                                        't':{'n': 20, 'mode':'grid'}})
    problem.discretise_domain(200, "random", domains=["u_0"])
    problem.discretise_domain(600, "random", domains=["u_1"])
    problem.discretise_domain(mode="grid", domains=["L_u"], 
                          sample_rules={'x':{'n': 100, 'mode':'grid'},
                                        'y':{'n': 100, 'mode':'grid'},
                                        't':{'n': 20,'mode':'grid'}})
    model = FeedForward(
    layers=config["layers"],
    func=torch.nn.Tanh,  # Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables))
    solver = PINN(
    problem,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr = config["lr"], weight_decay=0),
    loss=torch.nn.MSELoss(),
    weighting=ScalarWeighting({"x_bound_0_loss":1,
                               "x_bound_1_loss":1,
                               "y_bound_0_loss":1,
                               "y_bound_1_loss":1,
                               "L_u_loss":1,
                               "u_1_loss":config["weight"],
                               "u_0_loss":1
                                })
)  
    trainer= Trainer(solver, max_epochs=5000,accelerator="cpu",
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
    time_points = [0.1, 0.5, 1.0, 1.5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, t_val in enumerate(time_points):
        input_test = CartesianDomain({"x": [0, L], "y": [0, L], "t": t_val}).sample(200, mode="grid")
        output_test = solver(input_test)
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
    # Prepare a title that contains configuration info
    config_title = f"Layers: {config['layers']} | lr: {config['lr']}"
    fig.suptitle(config_title, fontsize=16)
    
    # Directory where the plot will be saved
# Directory where the plot will be saved
    save_dir = "training_plots"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# Get the absolute path of the training_plots directory
    absolute_path = os.path.abspath(save_dir)
    print(f"Absolute path of the plot directory: {absolute_path}")

# Generate a unique filename for the plot
    filename = os.path.join(save_dir, f"plot_{random.randint(3000, 18000)}.png")
    
# Save the plot
    plt.savefig(filename)
    plt.close(fig)

# Print the path where the image was saved
    print(f"Saved plot at: {filename}")

    
config = {
    "layers":tune.choice([[16,16,16,16],[10,20,20,10],[32,32,32,32]]),
    "lr":tune.choice([1e-4, 1e-3]),
    "weight":tune.choice([1,4])
    }

tune_analysis = tune.run(
    tune.with_parameters(Train_PINN),
    resources_per_trial={"cpu": 1, "gpu": 0},  # Adjust resources as needed
    metric="loss",
    mode="min",
    config=config,
    num_samples=24,  # Number of trials to run
    scheduler=ASHAScheduler(max_t=100, grace_period=10, reduction_factor=2)  # ASHA for early stopping
)
best_config = tune_analysis.get_best_config(metric="loss", mode="min")
print("Best hyperparameters found were: ", best_config)