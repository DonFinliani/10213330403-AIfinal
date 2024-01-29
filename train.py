from dataloader import trainset
from model import *
from torch.utils import data
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,mode,epochs=5,lr=7e-5,batch_size=64):
    start=time.perf_counter()
    criterion=nn.CrossEntropyLoss()
    model.train()
    trainloader=data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    losslist=[]
    acclist=[]
    for epoch in range(epochs):
        total_loss=0
        correct_num=0
        for v_images,v_texts,attention_masks,labels in trainloader:
            v_images=v_images.to(device)
            v_texts=v_texts.to(device)
            attention_masks=attention_masks.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs=model(v_images,v_texts,attention_masks,mode)
            _,preds=torch.max(outputs,dim=1)
            loss=criterion(outputs,labels)
            loss=loss.to(device)
            correct_num+=torch.sum(preds==labels)
            loss.backward()
            optimizer.step()
            total_loss+=loss
        epoch_loss=total_loss/len(trainloader)
        epoch_acc=correct_num/len(trainset)
        end_epoch=time.perf_counter()
        time_used=sec2min(end_epoch-start)
        print(f'time used:{time_used} epoch_loss:{epoch_loss:.4f} epoch_acc:{epoch_acc:.4f}')
        losslist.append(epoch_loss)
        acclist.append(epoch_acc)
    return losslist,acclist

def sec2min(sec):
    m=sec//60
    s=sec%60
    return f'{m:.0f}m{s:.0f}s'

if __name__ == '__main__':
    model=CombinedModel('multi')
    model=model.to(device)
    a,b=train(model,'multi')
    torch.save(model,"model/multi_7e-05_dropout0.6_batch_full.pth")