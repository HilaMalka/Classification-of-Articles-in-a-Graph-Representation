import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import numpy as np
from models import *
from sklearn.metrics import classification_report
import torch

res = pd.read_csv('loss_acc_tmp_GAT.csv')
acc = res['accuracy']
real_acc = [a for a in acc if a > 0]

plt.plot(real_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy for epoch')
plt.show()
plt.savefig('accuracy.png')
plt.plot(res['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss for epoch')
plt.show()
plt.savefig('loss.png')


model = GAT(hidden_channels=256, heads=4)
model.load_state_dict(torch.load('model_tmp_GAT.pth'))
model.eval()
print("####################################")

out = model.forward(data.x, data.edge_index)
pred = out[data.val_mask_bool].argmax(dim=1).tolist()
label = data.y[data.val_mask_bool].tolist()

print(classification_report(label, pred, digits=4))