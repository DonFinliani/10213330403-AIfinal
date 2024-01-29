from dataloader import valset
from model import *
from torch.utils import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def val(dataloader,model,mode):
    model.eval()
    correct_num=0
    valdataloader=dataloader
    for v_images,v_texts,attention_masks,labels in valdataloader:
        v_images=v_images.to(device)
        v_texts=v_texts.to(device)
        attention_masks=attention_masks.to(device)
        labels=labels.to(device)
        with torch.no_grad():
            outputs=model(v_images,v_texts,attention_masks,mode)
            _,preds=torch.max(outputs,1)
            correct_num+=torch.sum(labels==preds)
    return correct_num.cpu().numpy() /len(valset)

if __name__=='__main__':
    model=torch.load("model/multi_7e-05_dropout0.6_batch_full.pth")
    mode='multi'
    batch_size=64
    valdataloader=data.DataLoader(valset,batch_size,shuffle=False)
    val_acc=val(valdataloader,model,mode)
    print("val_acc:",val_acc)
            