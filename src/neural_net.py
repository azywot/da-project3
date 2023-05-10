from time import sleep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm as tqdm


class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets.astype(int))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


def data_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    dataset = NumpyDataset(X, y)
    return DataLoader(dataset, batch_size=16)


class SimpleNN(nn.Module):
    def __init__(self, criteria_nr: int = 5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(criteria_nr, 8)
        self.fc2 = nn.Linear(8, 16)
        # self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.sigmoid(x)
        return x


def train(
    model: SimpleNN, train_dataloader: DataLoader, lr: float = 0.01, epoch_nr: int = 10
) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    criterion = nn.BCELoss()
    for epoch in range(epoch_nr):
        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for data in tepoch:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                accuracy = (outputs.round() == labels).float().mean()

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item(), accuracy=100 * accuracy)
                tepoch.update()
                sleep(0.01)
