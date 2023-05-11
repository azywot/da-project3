from time import sleep
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm as tqdm
from torcheval.metrics import BinaryAccuracy
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


class SimpleNN(nn.Module):
    def __init__(self, criteria_nr: int = 5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(criteria_nr, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


def Train_NN(
    model: SimpleNN,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    path: str,
    lr: float = 0.01,
    epoch_nr: int = 200,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    loss_function = nn.BCELoss()
    metric = BinaryAccuracy()
    best_acc = 0.0
    best_auc = 0.0
    sig = nn.Sigmoid()
    for epoch in tqdm(range(epoch_nr)):
        for _, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(sig(np.squeeze(outputs)), labels.float())
            loss.backward()
            optimizer.step()
            metric.update(torch.flatten(outputs), labels)
            acc = metric.compute().item()
            auc = roc_auc_score(labels, outputs.detach().numpy()[:, 0])
            _, _, f1, _ = precision_recall_fscore_support(
                labels,
                np.squeeze(sig(outputs).detach().numpy().round()),
                average="binary",
            )

        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            best_f1 = f1
            with torch.no_grad():
                for i, data in enumerate(test_dataloader, 0):
                    inputs, labels = data
                    outputs = model(inputs)
                    loss_test = loss_function(sig(np.squeeze(outputs)), labels.float())
                    metric.update(torch.flatten(outputs), labels)
                    acc_test = metric.compute().item()
                    auc_test = roc_auc_score(labels, outputs.detach().numpy()[:, 0])
                    _, _, f1_test, _ = precision_recall_fscore_support(
                        labels,
                        np.squeeze(sig(outputs).detach().numpy().round()),
                        average="binary",
                    )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss,
                    "loss_test": loss_test,
                    "accuracy_train": acc,
                    "accuracy_test": acc_test,
                    "auc_train": auc,
                    "auc_test": auc_test,
                    "f1_train": f1,
                    "f1_test": f1_test,
                },
                path,
            )

    return best_acc, acc_test, best_auc, auc_test, best_f1, f1_test
