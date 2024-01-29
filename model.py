from dataloader import bertmodel,tokenizer
from torch import nn
import torch
from torchvision import models
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder,self).__init__()
        self.resnet=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    def forward(self,v_image):
        output=self.resnet(v_image)
        return output
    
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder,self).__init__()
        self.bert=bertmodel
        
    def forward(self,v_text,attention_mask):
        outputs=self.bert(v_text,attention_mask)
        return outputs[1]
    
class Classifier(nn.Module):
    def __init__(self,dropout,mode):
        super(Classifier,self).__init__()
        self.dropout=dropout
        if mode == 'image':
            self.input_size=1000
        elif mode =='text':
            self.input_size=768
        elif mode =='multi':
            self.input_size=1768
        
        self.dropout1=nn.Dropout(self.dropout)
        self.linear1=nn.Linear(self.input_size,256)
        self.batchnorm1=nn.BatchNorm1d(256)
        self.relu1=nn.ReLU()
        self.dropout2=nn.Dropout(self.dropout)
        self.linear2=nn.Linear(256,64)
        self.batchnorm2=nn.BatchNorm1d(64)
        self.relu2=nn.ReLU()
        self.linear3=nn.Linear(64,3)
    
    def forward(self,x):
        x=self.dropout1(x)
        x=self.linear1(x)
        x=self.batchnorm1(x)
        x=self.relu1(x)
        x=self.dropout2(x)
        x=self.linear2(x)
        x=self.batchnorm2(x)
        x=self.relu2(x)
        x=self.linear3(x)
        return x
    
class CombinedModel(nn.Module):
    def __init__(self,mode,dropout=0.5):
        super(CombinedModel,self).__init__()
        self.ImageModel=ImageEncoder()
        self.TextModel=TextEncoder()
        self.ClassifyModel=Classifier(dropout,mode) 

    def forward(self,v_image,v_text,attention_mask,mode):
        if mode == 'image':
            hidden_state=self.ImageModel(v_image)
        elif mode == 'text':
            hidden_state=self.TextModel(v_text,attention_mask)
        elif mode =='multi':
            hidden_state=torch.cat((self.ImageModel(v_image),self.TextModel(v_text,attention_mask)),dim=-1)
        output=self.ClassifyModel(hidden_state)
        return output