import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from pina import Trainer
from pina.model import FeedForward
from pina.solver import PINN, RBAPINN, SelfAdaptivePINN as SAPINN
from pina.problem.zoo import Poisson2DSquareProblem
from tsproblem import TimeSpaceProblem
from ray.tune.integration.pytorch_lightning import TuneReportCallback

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
                n_layers=config["n_layers"], inner_size=32,
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
        callbacks=[TuneReportCallback(metrics={"loss": "val_loss"}, on="train_end")],
        train_size=0.8,
        val_size=0.1,
        test_size=0.1
    )
    trainer.train()
    print("Training complete.")
config = {
    "coeff_time": tune.choice([0.5,1]),
    "n_layers": tune.choice([3,4]),
    "solver" : tune.choice([PINN, RBAPINN]),
    "v_shift":tune.choice([0,10,20]),
    "w_shift":tune.choice([0,10,20]),
    }

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