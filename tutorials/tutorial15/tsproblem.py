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
    output_variables = ["V", "W"]
    spatial_domain = CartesianDomain({"x": [0.1, 20]})
    Tf = 70
    temporal_domain = CartesianDomain({"t": [1, Tf]})

    # defining the ode equation
    def aliev_panfilov_equation1(input_, output_):
        # computing the derivative
        V_t = grad(output_, input_, components=["V"], d=["t"])
        nabla_V = laplacian(output_, input_, components=["V"], d=["x"])

        # extracting the u input variable
        V = output_.extract(["V"])
        W = output_.extract(["W"])
        # print(V.shape)
        # data
        k = 8
        a = 0.01
        #a=0.002 #additional
        D = 0.1  # 7.692e-5#0.1
        #D=0.02 #additional
        t = input_.extract(["t"])
        x = input_.extract(["x"])

        res_V = (
            (-V_t) + D * nabla_V - k * V * (V - a) * (V - 1) - V * W
        )  # + I_stim #+ 0.1

        return res_V

    def aliev_panfilov_equation2(input_, output_):
        # computing the derivative
        W_t = grad(output_, input_, components=["W"], d=["t"])

        # extracting the u input variable
        V = output_.extract(["V"])
        W = output_.extract(["W"])

        # data
        k = 8
        eps = 0.002
        mu1 = 0.2  # 0.2
        mu2 = 0.3
        b = 0.15  # 0.15
        #b=0.075 #additional
        res_W = W_t - ((eps + mu1 * W / (V + mu2)) * (-W - k * V * (V - b - 1)))
        return res_W

    def initial_cond_V(input_, output_):
        x = input_.extract(["x"])
        #return output_.extract(["V"]) - torch.exp(-0.8 * x)  # V_0 #original
        return output_.extract(["V"])-torch.exp(-0.05*x)
    def initial_cond_W(input_, output_):
        x = input_.extract(["x"])
        return output_.extract(["W"]) - 2.5 * torch.exp(-0.08 * x)  # W_0
        #return output_.extract(["W"]) - 2* torch.exp(-0.8 * x)  # W_0 #original
    

    # conditions to hold
    # specify the fixed gradient direction (ex x,y...)
    conditions = {
        "gamma1": Condition(
            domain=CartesianDomain({"x": 0.1, "t": [1, Tf]}),
            equation=FixedGradient(0.0, components=["V"], d=["x"]),
        ),
        "gamma2": Condition(
            domain=CartesianDomain({"x": 20, "t": [1, Tf]}),
            equation=FixedGradient(0.0, components=["V"], d=["x"]),
        ),
    #conditions={
        "t0_V": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": 1}),
            equation=Equation(initial_cond_V),
        ),  # FixedValue(0.01, components=['V']),data_weight=1),
        "t0_W": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": 1}),
            equation=Equation(initial_cond_W),
        ),  # FixedValue(0.01, components=['W']),data_weight=1),
        "D_V": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": [1, Tf]}),
            equation=Equation(aliev_panfilov_equation1),
        ),
        "D_W": Condition(
            domain=CartesianDomain({"x": [0.1, 20], "t": [1, Tf]}),
            equation=Equation(aliev_panfilov_equation2),
        ),
        #"dataV": Condition(input=input_domain, target=V_0),
        #"dataW": Condition(input=input_domain, target=W_0)
        #'data1': Condition(input_points=TX, output_points=Vsav_Wsav_labeled)#,data_weight=1),
    }
