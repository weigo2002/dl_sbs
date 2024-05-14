import torch
from torch import nn

from train import Trainer
from data_generation import generate_data
from data_preparation import prepare_data

def configure_model():
    lr = 0.1

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(1, 1))

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='mean')
    return model, loss_fn, optimizer

if __name__ == '__main__':

    model, loss_fn, optimizer = configure_model()

    trainer = Trainer(model, loss_fn, optimizer)
    print('Start generating data...')
    x, y = generate_data()
    train_loader, val_loader = prepare_data(x, y)
    trainer.set_loaders(train_loader, val_loader)
    trainer.set_tensorboard('classy')

    print('Start training...')
    trainer.train(n_epochs=200)
    print('Finish training.')
