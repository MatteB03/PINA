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


problem = TimeSpaceProblem()

problem.discretise_domain(4000, "random", domains=["L_u"])
problem.discretise_domain(250, "grid", domains=["step1", "step3"])
problem.discretise_domain(501, "grid", domains=["step2"])
problem.discretise_domain(1001, "grid", domains=["u0","u1"])

# setting the seed for reproducibility
seed_everything(42, workers=True)

# build the model
model = FeedForward(
    layers=[16,16,16,16],
    func=torch.nn.Tanh,  # Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables),
)
pinn = PINN(
    problem,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=5e-5, weight_decay=0), ##try varying lr
    loss=torch.nn.MSELoss()
)  

from lightning.pytorch.loggers import TensorBoardLogger
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

'''input_test = CartesianDomain({"x": [0, 1], "t": [0, 1]}).sample(501, mode='grid')
output_test = pinn(input_test)
u = output_test.extract(["u"])

plt.figure(figsize=(6, 5))
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=u.detach().numpy())
plt.colorbar()
plt.title('u')
plt.grid()
plt.tight_layout() 
plt.show()''' #PRE-PAPER-SOLUTION


'''print("Shape of u_saved:", u_saved.shape)
print("Shape of u_fdm:", u_fdm.shape)
print("Shape of u_pinn:", u_pinn.shape)

print("Max value of u_saved:", np.max(u_saved))
print("Min value of u_saved:", np.min(u_saved))
print("Max value of u_fdm:", np.max(u_fdm))
print("Min value of u_fdm:", np.min(u_fdm))
print("Max value of u_pinn:", np.max(u_pinn))
print("Min value of u_pinn:", np.min(u_pinn))'''



'''diff1 = u_pinn - u_saved
diff2 = u_pinn - u_fdm

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(diff1, extent=[0, 1, 0, 1], origin="lower", cmap="bwr", aspect="auto")
plt.colorbar(label="Difference")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Difference Between PINN and Paper Solution")

plt.subplot(1, 2, 2)
plt.imshow(diff2, extent=[0, 1, 0, 1], origin="lower", cmap="bwr", aspect="auto")
plt.colorbar(label="Difference")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Difference Between PINN and FDM Solution")

plt.tight_layout()
plt.show()'''


import numpy as np
import matplotlib.pyplot as plt

input_test = CartesianDomain({"x": [0, 1], "t": [0, 1]}).sample(50, mode="grid")
output_test = pinn(input_test)
import torch

torch.save(output_test, "pinn_output_post_tuning.pt")

data_saved = np.load("saved_solution.npz")
#data_fdm = np.load("fdm_solution.npz")

x_saved, t_saved, u_saved = data_saved["x"], data_saved["t"], data_saved["u"]
#x_fdm, t_fdm, u_fdm = data_fdm["x"], data_fdm["t"], data_fdm["u"]

u_pinn = output_test.extract(["u"]).detach().numpy().reshape(501, 501).T


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(u_saved, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
plt.colorbar()
plt.title("Saved Solution")

plt.subplot(1, 2, 2)
plt.imshow(u_pinn, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
plt.colorbar()
plt.title("PINN Solution")


plt.tight_layout()
plt.show()
'''
# Load saved PINN solution
data_saved = np.load("saved_solution.npz")
x_pinn = data_saved["x"]  # Spatial points for old PINN
t_pinn = data_saved["t"]  # Time points for old PINN
u_pinn = data_saved["u"]  # Old PINN solution values

# Time steps to plot
t_test = [100, 500, 2000, 5000]
fig = plt.figure(figsize=(10, 8))
fignum = 1
xx = np.linspace(0, w, num=nx + 1, dtype=np.float32)

# Compute and plot solutions
for i in range(nt + 1):
    u0, u = do_timestep(u0, u)  # FDM step
    
    if i in t_test:  # Only plot at specific time steps
        test_points = np.stack((xx, np.full((nx + 1), i * dt)), axis=1)
        pinn_new = pinn(torch.tensor(test_points, dtype=torch.float32)).detach().numpy()  # New PINN solution

        ax = fig.add_subplot(220 + fignum)
        ax.plot(xx, u, 'k-', label="FDM")
        ax.plot(xx, pinn_new, 'b--', label="New PINN")

        # Find closest time index in old PINN data
        time_idx = np.abs(t_pinn - i * dt).argmin()
        ax.plot(x_pinn, u_pinn[:, time_idx], 'r-.', label="Old PINN")

        ax.set_title(f'Time step {i}')
        ax.set_ylim([-0.1, 1.5])
        ax.legend()
        fignum += 1

plt.tight_layout(h_pad=2)
plt.show()
'''
