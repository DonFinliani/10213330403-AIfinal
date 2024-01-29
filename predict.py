from dataloader import transform,multidata
from model import *
from torch.utils import data
import pandas as pd
from PIL import Image
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torch.load("model/multi_7e-05_dropout0.6_batch_full.pth")

data_t=pd.read_csv("Data/test_without_label.txt",index_col=0)
id_list=data_t.index.values 
labels=data_t.values
texts=[]
v_images=[]
for id in id_list:
    text_path=f'Data/data/{id}.txt'
    image_path=f'Data/data/{id}.jpg'
    with open(text_path,'r',encoding="GB18030") as f:
        texts.append(f.read())
    image=Image.open(image_path)
    v_images.append(transform(image))    
    
tok_texts=tokenizer(texts,padding='max_length',max_length=100,truncation=True,return_tensors='pt')

predictset=multidata(v_images,tok_texts['input_ids'],tok_texts['attention_mask'],labels)
predictdataloader=data.DataLoader(predictset,batch_size=64,shuffle=False)
mode='multi'
model.eval()
predict_labels=np.array([])
for v_images,v_texts,attention_masks,_ in predictdataloader:
    v_images=v_images.to(device)
    v_texts=v_texts.to(device)
    attention_masks=attention_masks.to(device)    
    with torch.no_grad():
        outputs=model(v_images,v_texts,attention_masks,mode)
        _,preds=torch.max(outputs,1)
        preds=preds.cpu().numpy()
        predict_labels=np.concatenate((predict_labels,preds),axis=0)


data_result=pd.read_csv("Data/test_without_label.txt",index_col=0)
data_result['tag']=predict_labels
num_label={0:'positive',1:'neutral',2:'negative'}
data_result['tag']=data_result['tag'].replace(num_label)
data_result.to_csv("Data/result.txt")
