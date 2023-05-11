# SOURCE: Professor Krzysztof Martyn (PUT)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import BinaryAccuracy
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import numpy as np

from functools import partial
from src.neural_net import SimpleNN


class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.Tensor(data)
        self.targets = torch.LongTensor(targets.astype(int))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)

def BCELoss(x, target):
    return torch.mean(-target * torch.log(x) - (1 - target) * torch.log(1 - x))

def Regret(x, target):
    return torch.mean(
        torch.relu(-(target >= 1).float() * x) + torch.relu((target < 1).float() * x)
    )


def Accuracy(x, target):
    return (target == (x[:, 0] > 0) * 1).detach().numpy().mean()

def AUC(x, target):
    return roc_auc_score(target.detach().numpy(), x.detach().numpy()[:, 0])


def F1_score(x, target):
    y_pred = (x[:, 0] > 0) * 1
    _, _, f1_score, _ = precision_recall_fscore_support(
        target.detach().numpy(), y_pred, average="binary"
    )
    return f1_score

def CreateDataLoader(X, y):
    dataset = NumpyDataset(X, y)
    return DataLoader(dataset, batch_size=len(dataset))


def Train(
    model,
    train_dataloader,
    test_dataloader,
    path,
    lr=0.01,
    epoch_nr=200,
    loss_function=Regret,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    best_acc = 0.0
    best_auc = 0.0
    for epoch in tqdm(range(epoch_nr)):
        for _, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            acc = Accuracy(outputs, labels)
            auc = AUC(outputs, labels)
            f1 = F1_score(outputs, labels)

        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            best_f1 = f1
            with torch.no_grad():
                for i, data in enumerate(test_dataloader, 0):
                    inputs, labels = data
                    outputs = model(inputs)
                    loss_test = loss_function(outputs, labels)
                    acc_test = Accuracy(outputs, labels)
                    auc_test = AUC(outputs, labels)
                    f1_test = F1_score(outputs, labels)

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


def Train_NN(
    model: SimpleNN,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    path: str,
    lr: float=0.01,
    epoch_nr: int=200,
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
            # print(np.squeeze(outputs).shape, labels.shape, type(np.squeeze(outputs)), type(labels))
            # print(sig(np.squeeze(outputs)), labels)
            loss = loss_function(sig(np.squeeze(outputs)), labels.float())
            loss.backward()
            optimizer.step()
            metric.update(torch.flatten(outputs), labels)
            acc = metric.compute()
            auc = roc_auc_score(labels, outputs.detach().numpy()[:, 0])
            # print(np.squeeze(sig(outputs).detach().numpy().round()))
            _, _, f1, _ = precision_recall_fscore_support(
                labels, np.squeeze(sig(outputs).detach().numpy().round()), average="binary"
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
                    acc_test = (outputs.round() == labels).float().mean()
                    auc_test = roc_auc_score(labels, outputs.detach().numpy()[:, 0])
                    _, _, f1_test, _ = precision_recall_fscore_support(
                        labels, np.squeeze(sig(outputs).detach().numpy().round()), average="binary"
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

    return best_acc.numpy(), acc_test.numpy(), best_auc, auc_test, best_f1, f1_test


class Hook:
    def __init__(self, m, f):
        self.hook = m.register_forward_hook(partial(f, self))

    def remove(self):
        self.hook.remove()

    def __del__(self):
        self.remove()


def append_output(hook, mod, inp, outp):
    if not hasattr(hook, "stats"):
        hook.stats = []
    if not hasattr(hook, "name"):
        hook.name = mod.__class__.__name__
    data = hook.stats
    data.append(outp.data)
