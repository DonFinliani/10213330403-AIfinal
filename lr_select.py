from dataloader import trainset,valset
from model import *
from train import *
from val import *
from torch.utils import data

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mode='multi'
lr_list=[2e-4,9e-5,7e-5]
epochs=5
model=CombinedModel(mode).to(device)
best_acc=0.0
best_lr=0.0
for lr in lr_list:
    a,b=train(model,mode,epochs=epochs,lr=lr)
    val_acc=val(model,mode)
    print(f'lr={lr},train_acc={b[-1]:.04f},val_acc={val_acc:.04f}')
    if(val_acc>best_acc):
        best_acc=val_acc
        best_lr=lr
        torch.save(model,f'model/sm_{lr}.pth')