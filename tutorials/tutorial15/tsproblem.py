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

from pytorch_lightning import seed_everything
from pina.optim import TorchOptimizer

class TimeSpaceProblem(TimeDependentProblem, SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    temporal_domain = CartesianDomain({"t": [0,1]})

    # defining the ode equation
    def equation_4p5(input_, output_):
        # computing the derivative
        u_t = grad(output_, input_, components=["u"], d=["t"])
        nabla_u = laplacian(output_, input_, components=["u"], d=["x"])
        
        u = output_.extract(["u"])
        gamma = 1

        res = u_t - gamma * nabla_u + u**3 - u
        return res
        

    def initial_cond_V(input_, output_):
        x = input_.extract(["x"])
        #return output_.extract(["V"]) - torch.exp(-0.8 * x)  # V_0 #original
        return output_.extract(["V"])-torch.exp(-0.8 * x)
    def initial_cond_W(input_, output_):
        x = input_.extract(["x"])
        return output_.extract(["W"]) - 2.5 * torch.exp(-0.8 * x)  # W_0
        #return output_.extract(["W"]) - 2* torch.exp(-0.8 * x)  # W_0 #original
    

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
