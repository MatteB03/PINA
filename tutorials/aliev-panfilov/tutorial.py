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

from tutorials.tutorial15.tuner import TimeSpaceProblem
from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

snap=1

domain_ = np.linspace(0.1, 20, 200)
domain_lol = torch.tensor([[value] for value in domain_], dtype=torch.float32)  
time = torch.ones(domain_lol.shape[0], 1, dtype=torch.float32)
time=time*snap
domain_with_time = torch.cat((domain_lol, time), dim=1) 
labels = ['x', 't']
input_domain = LabelTensor(domain_with_time, labels=labels)

data = io.loadmat("../../../EP-PINNs/data_files/Aliev_Panfilov_Model/_1D/data_1d_left.mat")
Vin = torch.tensor(data['Vsav'][snap-1,:],dtype=torch.float32).unsqueeze(1)
Win = torch.tensor(data['Wsav'][snap-1,:],dtype=torch.float32).unsqueeze(1)
VinT=torch.cat((Vin, time), dim=1)
WinT=torch.cat((Win, time), dim=1)
V_0 = LabelTensor(VinT, labels=['V','t'])
W_0 = LabelTensor(WinT, labels=['W','t'])

problem = TimeSpaceProblem()

problem.discretise_domain(4000, "random", domains=["D_V", "D_W"])
problem.discretise_domain(140, "grid", domains=["t0_V", "t0_W"])
problem.discretise_domain(200, "grid", domains=["gamma2", "gamma1"])

#print("Input points:", problem.input_pts)

# from pina import Plotter

# pl = Plotter()
# pl.plot_samples(problem=problem)#,variables='spatial')


# setting the seed for reproducibility
seed_everything(42, workers=True)

# build the model
'''model = FeedForward(
    layers=[32, 32, 32, 32],
    func=torch.nn.Tanh,  # Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables),
)

class MyModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.space = FeedForward(
            layers=[32, 32, 32], func=torch.nn.Softplus, input_dimensions=1, output_dimensions=50)

        #self.coeff_time = torch.nn.Parameter(torch.rand(1))
        self.time = FeedForward(
            layers=[32, 32, 32], func=torch.nn.Softplus, input_dimensions=1, output_dimensions=50)
        
        self.reduction = torch.nn.Linear(50, 2)

    def forward(self, x):
        t = self.time(x.extract(["t"]))
        x = self.space(x.extract(["x"]))
        return self.reduction(t * x)
        # return tmp'''



class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.space_w = FeedForward(
            layers=[32]*3, func=torch.nn.Tanh, input_dimensions=1, output_dimensions=1)
        self.space_v = FeedForward(
            layers=[32]*3, func=torch.nn.Tanh, input_dimensions=1, output_dimensions=1)
        self.coeff_time_v = torch.nn.Parameter(torch.Tensor([1]))
        self.coeff_time_w = torch.nn.Parameter(torch.Tensor([1]))
        self.known_term_v = torch.nn.Parameter(torch.Tensor([0]))
        self.known_term_w = torch.nn.Parameter(torch.Tensor([10]))
        self.v_shift = torch.nn.Parameter(torch.Tensor([0]))
        self.w_shift = torch.nn.Parameter(torch.Tensor([10]))

    def forward(self, x):
        x_ = x.extract(["x"])
        v = self.space_v(x_ + self.known_term_v - self.coeff_time_v * (x.extract(["t"]) - self.v_shift))
        w = self.space_w(x_ + self.known_term_w - self.coeff_time_w * (x.extract(["t"]) - self.w_shift))




        v.labels = ["V"]
        w.labels=["W"]
        return torch.cat([v, w], dim=1)




model = MyModel()
problem_initial = problem

pinn = PINN(
    problem_initial,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=1e-4, weight_decay=0),
    loss=torch.nn.MSELoss()
)  
pinn2 = SAPINN(problem_initial,
    model,
    optimizer_model=TorchOptimizer(torch.optim.AdamW, lr=1e-3, weight_decay=0),
    loss=torch.nn.MSELoss()
)  
#from lightning.pytorch.loggers import TensorBoardLogger

trainer = Trainer(
    solver=pinn,
    max_epochs=5000,#logger=TensorBoardLogger('tutorial_logs'),
    accelerator="cpu",
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

input_test = CartesianDomain({"x": [0.1, 20], "t": [1, 70]}).sample(70, mode='grid')
#input_test = CartesianDomain({"x": [0.1, 20], "t": [1, 70]}).sample(100, mode='grid')
output_test = pinn(input_test)
V = output_test.extract(["V"])
W = output_test.extract(["W"])

from equation import AlievPanfilov1D_RK_Istim
import matplotlib.pyplot as plt

Vsav, Wsav = AlievPanfilov1D_RK_Istim()


#print(W.detach().numpy().shape)
NewW=Wsav.flatten()
#print(len(NewW))


plt.figure(figsize=(12, 10))

# V plot
plt.subplot(3, 2, 1) 
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=V.detach().numpy(), cmap='coolwarm')
plt.colorbar()
plt.title('V')

# W plot
plt.subplot(3, 2, 2)
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=W.detach().numpy(), cmap='coolwarm')
plt.colorbar()
plt.title('W')

# Vsav plot
plt.subplot(3, 2, 3)
plt.title('V (AU)')
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=Vsav.flatten(), cmap='coolwarm')
plt.ylabel('Time (AU)')
plt.xlabel('x (cells)')
plt.colorbar()

# Wsav plot
plt.subplot(3, 2, 4)
plt.title('W (AU)')
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=Wsav.flatten(), cmap='coolwarm')
plt.ylabel('Time (AU)')
plt.xlabel('x (cells)')
plt.colorbar()

# V - Vsav difference
plt.subplot(3, 2, 5)
plt.title('V - Vsav')
V_diff = V.detach().numpy().flatten() - Vsav.flatten()
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=V_diff, cmap='coolwarm')
plt.ylabel('Time (AU)')
plt.xlabel('x (cells)')
plt.colorbar()

# W - Wsav difference
plt.subplot(3, 2, 6)
plt.title('W - Wsav')
W_diff = W.detach().numpy().flatten() - Wsav.flatten()
plt.scatter(input_test.extract(["x"]), input_test.extract(["t"]), c=W_diff, cmap='coolwarm')
plt.ylabel('Time (AU)')
plt.xlabel('x (cells)')
plt.colorbar()

plt.tight_layout() 
plt.show()