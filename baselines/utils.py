import torch.nn as nn


def create_net(n_inputs, n_outputs, n_layers=1, n_units=100, nonlinear=nn.Tanh):
    if n_layers == 0:
        return nn.Linear(n_inputs, n_outputs)
    layers = [nn.Linear(n_inputs, n_units)]
    for _ in range(n_layers - 1):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)
