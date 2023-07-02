from models import *
from dataset import HW3Dataset
import torch
import pandas as pd

model = GAT(hidden_channels=256, heads=4)
model.load_state_dict(torch.load('comp.pth'))
model.eval()

dataset = HW3Dataset(root='data/hw3/')
data = dataset[0]
out = model.forward(data.x, data.edge_index)
pred = out.argmax(dim=1).tolist()
idiocies = list(range(len(pred)))

df = pd.DataFrame.to_csv(pd.DataFrame.from_dict({'idx': idiocies,
                                                 'prediction': pred}),
                         'prediction.csv', index=False)
