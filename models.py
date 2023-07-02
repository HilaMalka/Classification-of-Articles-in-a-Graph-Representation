from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from data_loader import load_dataset
from torch_geometric.nn import GATConv
from torch_geometric.nn import ChebConv
import pandas as pd
import numpy as np


criterion = torch.nn.CrossEntropyLoss()
# dataset = torch.load('/home/student/HW3/data/hw3/raw/data.pt')
device = torch.device("cpu")
data = load_dataset()


class Cheb(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = ChebConv(128, hidden_channels, 5)
        self.conv2 = ChebConv(hidden_channels, hidden_channels//4, 5)
        # self.conv3 = ChebConv(hidden_channels // 2, hidden_channels // 4, 2)
        # self.conv4 = ChebConv(hidden_channels // 4, hidden_channels // 8, 2)
        self.conv5 = ChebConv(hidden_channels // 4, 40, 5)

    def forward(self, x, edge_index):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = (x - mean) / std
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.25, training=self.training)
        # x = self.conv4(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv5(x, edge_index)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(128, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels//2)
        self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        self.conv4 = GCNConv(hidden_channels // 4, hidden_channels // 8)
        self.conv5 = GCNConv(hidden_channels // 8, 40)

    def forward(self, x, edge_index):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = (x - mean) / std
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv4(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.25, training=self.training)
        x = self.conv5(x, edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(128, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels*heads, 40, heads=heads)
        # self.conv3 = GATConv(heads*hidden_channels//2, 40,  heads=heads)

    def forward(self, x, edge_index):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        x = (x - mean) / std
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        # x = x.tanh()
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv3(x, edge_index)
        return x


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x.to(device), data.edge_index.to(device))  # Perform a single forward pass.
    loss = criterion(out[data.train_mask_bool],
                     data.y[data.train_mask_bool])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def train_batched():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    batch_size = 16  # Number of samples per mini-batch
    num_batches = len(data.train_mask_bool) // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        eyal = data.x[start_idx:end_idx].to(device)
        out = model(data.x[start_idx:end_idx].to(device), data.edge_index.to(device))
        # loss = criterion(out[data.train_mask_bool[start_idx:end_idx]],
        #                  data.y[data.train_mask_bool[start_idx:end_idx]])

        # loss.backward()

        # Perform gradient accumulation every `accumulation_steps` mini-batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Update parameters based on remaining gradients
    optimizer.step()
    optimizer.zero_grad()

    return loss




def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.val_mask_bool] == data.y[data.val_mask_bool]  # Check against ground-truth labels.
    dict_pred = {'y_pred': pred[data.val_mask_bool], 'y_true':data.y[data.val_mask_bool] }
    df = pd.DataFrame.from_dict(dict_pred)
    df.to_csv('preds.csv')
    test_acc = int(test_correct.sum()) / int(data.val_mask_bool.sum())  # Derive ratio of correct predictions.
    return test_acc


if __name__ == '__main__':
    model = GAT(hidden_channels=256, heads=4)
    #model = GCN(hidden_channels=512)
    #model = Cheb(hidden_channels=256)

    model = model.to(device)
    print(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    losses = []
    accs = []
    n = 1
    for epoch in range(1, 500):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-4)
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        losses.append(np.round(loss.item(),4))
        if epoch % 5 == 0:
            test_acc = test()
            accs.append(test_acc)
            print(f'Epoch: {epoch:03d}, Acc: {test_acc:.2f}')
        # if epoch % 100 == 0:
        #     n = n/2
    test_acc = test()
    zeros_to_add = [0] * (len(losses) - len(accs))
    accs.extend(zeros_to_add)

    dict_res = {'loss': losses, 'accuracy': accs}
    df = pd.DataFrame.from_dict(dict_res)
    df.to_csv('loss_acc_tmp_GAT.csv')
    print(f'Test Accuracy: {test_acc:.4f}')
    torch.save(model.state_dict(), "model_tmp_GAT.pth")
    
   