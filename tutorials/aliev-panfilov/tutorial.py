# imports
import warnings
import torch
import scipy.io as io
import numpy as np
from scipy.io import loadmat

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

from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

warnings.filterwarnings('ignore')

############################################################################

# this part imports the first time snapshot data to use in similar fashion to the successful analysis by Perfrancesco,I avoided
# using it since we'd be using data and the result doesn't seem to improve a lot with only a single snapshot but feel free to do as you please

domain_ = np.linspace(0.1, 20, 200)
domain_lol = torch.tensor([[value] for value in domain_], dtype=torch.float32)  
time = torch.ones(domain_lol.shape[0], 1, dtype=torch.float32)
domain_with_time = torch.cat((domain_lol, time), dim=1) 
labels = ['x', 't']
input_domain = LabelTensor(domain_with_time, labels=labels)

# to use the data you must have installed the repo at this link: https://github.com/martavarela/EP-PINNs
data = io.loadmat("../../../EP-PINNs/data_files/Aliev_Panfilov_Model/_1D/data_1d_left.mat") 
Vin = torch.tensor(data['Vsav'][0,:],dtype=torch.float32).unsqueeze(1)
Win = torch.tensor(data['Wsav'][0,:],dtype=torch.float32).unsqueeze(1)
VinT=torch.cat((Vin, time), dim=1)
WinT=torch.cat((Win, time), dim=1)
V_0 = LabelTensor(VinT, labels=['V','t'])
W_0 = LabelTensor(WinT, labels=['W','t'])

############################################################################

# now we define the problem as you'll probably have seen plenty of times in the tutorials
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
            equation=Equation(aliev_panfilov_equation2),
        ),
        #"dataV": Condition(input=input_domain, target=V_0),
        #"dataW": Condition(input=input_domain, target=W_0) #these are the additional conditions if you want to train using the first snapshot
    }

# we now assign the problem and discretise its domains 
problem = TimeSpaceProblem()
# discretise according to the importance and extension of each condition, feel free to change the values here
problem.discretise_domain(4000, "random", domains=["D_V", "D_W"])
problem.discretise_domain(140, "grid", domains=["t0_V", "t0_W"])
problem.discretise_domain(200, "grid", domains=["gamma2", "gamma1"])
#problem.discretise_domain(2000, "random", domains=["dataV", "dataW"])

############################################################################

# setting the seed for reproducibility
seed_everything(42, workers=True)


# build the model and set up the training

# I'll leave here these models if you think it would be interesting to experiment with simpler structures as well
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


# the current model has two separate spaces for v and w and 3 separate learnable Parameters for each variable
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.space_w = FeedForward(
            layers=[32]*3, func=torch.nn.Tanh, input_dimensions=1, output_dimensions=1)
        self.space_v = FeedForward(
            layers=[32]*3, func=torch.nn.Tanh, input_dimensions=1, output_dimensions=1)
        
        # Feel free to modify the parameters at will, even removing them or using the same parameter for both parts of the forward method
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


# assign the model
model = MyModel()

# assign the solver, feel free to pick other activation functions or learning rates (these come from tuning)
pinn = PINN(
    problem,
    model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=1e-4, weight_decay=0),
    loss=torch.nn.MSELoss()
)  
pinn2 = SAPINN(problem,
    model,
    optimizer_model=TorchOptimizer(torch.optim.AdamW, lr=1e-3, weight_decay=0),
    loss=torch.nn.MSELoss()
)  

#from lightning.pytorch.loggers import TensorBoardLogger #use TensorBoard to save the training losses, to launch a TB session look up tutorials 1,2 or 3

# set up the trainer
trainer = Trainer(
    solver=pinn,
    max_epochs=5000,#logger=TensorBoardLogger('tutorial_logs'),
    accelerator="cpu",
    enable_model_summary=False,
    gradient_clip_val=0.7,
    train_size=0.9,
    val_size=0.0,
    test_size=0.1
)  # , accelerator='cpu', enable_model_summary=False) # we train on CPU and avoid model summary at beginning of training (optional)

# train and test
trainer.train()
trainer.test()

############################################################################

# plot the solution

input_test = CartesianDomain({"x": [0.1, 20], "t": [1, 70]}).sample(70, mode='grid')
output_test = pinn(input_test)
V = output_test.extract(["V"])
W = output_test.extract(["W"])

# take the real solution to visualise and compute the difference 
from equation import AlievPanfilov1D_RK_Istim
import matplotlib.pyplot as plt
Vsav, Wsav = AlievPanfilov1D_RK_Istim()

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