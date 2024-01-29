from dataloader import trainset,valset
from model import *
from train import *
from val import *
from torch.utils import data
import argparse
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=64
valdataloader=data.DataLoader(valset,batch_size,shuffle=False)

parser=argparse.ArgumentParser(description="")
parser.add_argument('--mode',type=str,help='choose a mode',default='all')
parser.add_argument('--lr',type=float,help="choose learning_rate",default=7e-5)
parser.add_argument('--epochs',type=int,help="epochs num",default=5)
parser.add_argument('--dropout',type=float,help='set arg of dropout',default=0.6)
args=parser.parse_args()

if args.mode=='all':
    mode_list=['multi','text','image']
elif args.mode =='multi':
    mode_list=['multi']
elif args.mode =='text':
    mode_list=['text']
elif args.mode =='image':
    mode_list=['image']

lr=args.lr
epochs=args.epochs

for mode in mode_list:
    model=CombinedModel(mode,dropout=args.dropout).to(device)
    a,b=train(model,mode,epochs=epochs,lr=lr)
    val_acc=val(valdataloader,model,mode)
    print(f'mode={mode},train_acc={b[-1]:.04f},val_acc={val_acc:.04f}')
    torch.save(model,f'model/{mode}_{lr}_dropout0.6_batch_full.pth')  
  