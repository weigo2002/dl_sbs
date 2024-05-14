import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def prepare_data(x, y):
    torch.manual_seed(13)

    x_tensor = torch.as_tensor(x).float()
    y_tensor = torch.as_tensor(y).float()

    dataset = TensorDataset(x_tensor, y_tensor)

    ratio = .8
    n_total = len(dataset)
    n_train = int(ratio * n_total)
    n_val = n_total - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, )
    val_loader = DataLoader(dataset=val_data, batch_size=16, )
    return train_loader, val_loader