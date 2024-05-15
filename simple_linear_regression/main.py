import torch
from torch import nn

from simple_linear_regression.model import build_model
from train import Trainer
from data_generation import generate_data
from data_preparation import prepare_data


if __name__ == '__main__':
    model, loss_fn, optimizer = build_model()

    trainer = Trainer(model, loss_fn, optimizer)
    print('Start generating data...')
    x, y = generate_data()
    train_loader, val_loader = prepare_data(x, y)
    trainer.set_loaders(train_loader, val_loader)
    trainer.set_tensorboard('classy')

    print('Start training...')
    trainer.train(n_epochs=200)
    print('Finish training.')
    print(model.state_dict())
