import torch
import pytest

from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina import LabelTensor, Condition
from pina.domain import CartesianDomain
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue


def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                  torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term


my_laplace = Equation(laplace_equation)
in_ = LabelTensor(torch.tensor([[0., 1.]], requires_grad=True), ['x', 'y'])
out_ = LabelTensor(torch.tensor([[0.]], requires_grad=True), ['u'])


class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'gamma1':
        Condition(domain=CartesianDomain({
            'x': [0, 1],
            'y': 1
        }),
                  equation=FixedValue(0.0)),
        'gamma2':
        Condition(domain=CartesianDomain({
            'x': [0, 1],
            'y': 0
        }),
                  equation=FixedValue(0.0)),
        'gamma3':
        Condition(domain=CartesianDomain({
            'x': 1,
            'y': [0, 1]
        }),
                  equation=FixedValue(0.0)),
        'gamma4':
        Condition(domain=CartesianDomain({
            'x': 0,
            'y': [0, 1]
        }),
                  equation=FixedValue(0.0)),
        'D':
        Condition(domain=CartesianDomain({
            'x': [0, 1],
            'y': [0, 1]
        }),
                  equation=my_laplace),
        'data':
        Condition(input_points=in_, output_points=out_)
    }

    def poisson_sol(self, pts):
        return -(torch.sin(pts.extract(['x']) * torch.pi) *
                 torch.sin(pts.extract(['y']) * torch.pi)) / (2 * torch.pi**2)

    truth_solution = poisson_sol


def test_discretise_domain():
    n = 10
    poisson_problem = Poisson()
    boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
    poisson_problem.discretise_domain(n, 'grid', domains=boundaries)
    for b in boundaries:
        assert poisson_problem.discretised_domains[b].shape[0] == n
    poisson_problem.discretise_domain(n, 'random', domains=boundaries)
    for b in boundaries:
        assert poisson_problem.discretised_domains[b].shape[0] == n

    poisson_problem.discretise_domain(n, 'grid', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n**2
    poisson_problem.discretise_domain(n, 'random', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n

    poisson_problem.discretise_domain(n, 'latin', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n

    poisson_problem.discretise_domain(n, 'lh', domains=['D'])
    assert poisson_problem.discretised_domains['D'].shape[0] == n

    poisson_problem.discretise_domain(n)


def test_sampling_few_variables():
    n = 10
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n,
                                      'grid',
                                      domains=['D'],
                                      variables=['x'])
    assert poisson_problem.discretised_domains['D'].shape[1] == 1


def test_variables_correct_order_sampling():
    n = 10
    poisson_problem = Poisson()
    poisson_problem.discretise_domain(n,
                                      'grid',
                                      domains=['D'])
    assert poisson_problem.discretised_domains['D'].labels == sorted(
        poisson_problem.input_variables)

    poisson_problem.discretise_domain(n, 'grid', domains=['D'])
    assert poisson_problem.discretised_domains['D'].labels == sorted(
        poisson_problem.input_variables)


# def test_add_points():
#     poisson_problem = Poisson()
#     poisson_problem.discretise_domain(0,
#                                       'random',
#                                       domains=['D'],
#                                       variables=['x', 'y'])
#     new_pts = LabelTensor(torch.tensor([[0.5, -0.5]]), labels=['x', 'y'])
#     poisson_problem.add_points({'D': new_pts})
#     assert torch.isclose(poisson_problem.discretised_domain['D'].extract('x'),
#                          new_pts.extract('x'))
#     assert torch.isclose(poisson_problem.discretised_domain['D'].extract('y'),
#                          new_pts.extract('y'))