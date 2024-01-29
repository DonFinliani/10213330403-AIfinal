from torch.utils import data
from transformers import BertModel, BertTokenizer
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

bertmodel=BertModel.from_pretrained('bert-base-uncased')
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

data1=pd.read_csv('Data/train.txt',index_col=0)
data1=data1
label_num={'positive':0,'neutral':1,'negative':2}
data2=data1['tag'].replace(label_num)
id_list=data2.index.values
labels=data2.values
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

class multidata(data.Dataset):
    def __init__(self,v_images,v_texts,attention_masks,labels):
        self.v_images=v_images
        self.v_texts=v_texts
        self.attention_masks=attention_masks
        self.labels=labels
    def __getitem__(self,index):
        v_image=self.v_images[index]
        v_text=self.v_texts[index]
        attention_mask=self.attention_masks[index]
        label=self.labels[index]
        return v_image,v_text,attention_mask,label
    def __len__(self):
        return len(self.labels)

images_train,images_val,texts_v_train,texts_v_val,texts_attn_train,texts_attn_val,labels_train,labels_val=train_test_split(v_images,tok_texts['input_ids'],tok_texts['attention_mask'],labels,test_size=0.2,random_state=42)       

trainset=multidata(images_train,texts_v_train,texts_attn_train,labels_train)
valset=multidata(images_val,texts_v_val,texts_attn_val,labels_val)


'''trainloader=data.Dataloader(trainset,batch_size=64,shuffle=True)
valloader=data.Dataloader(valset,batch_size=64,shuffle=False)'''