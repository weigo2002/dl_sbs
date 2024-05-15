import torch
from torch import nn


def build_model(lr=0.1):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(1, 1))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='mean')
    return model, loss_fn, optimizer
