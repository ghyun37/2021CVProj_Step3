# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:27:58 2021

@author: Gahyun
"""

import os
import glob
import shutil
from collections import OrderedDict
# from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
# from torchsummary import summary
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils import Dataset
from mobilenet import MobileNetV2, Bottleneck


# 파라미터
nbatch = 16 # batch size
rvalid = 0.2 # 검증용 데이터 구성 비율
classes = ('burn_disease', 'healthy', 'leafspot') # 클래스 정보
## data augmentation
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224,224)),
                                transforms.RandomResizedCrop(size=(224,224), scale=(0.5, 1.5), ratio=(1,1)),
                                transforms.RandomRotation(85),
                                transforms.RandomAffine(translate=(0.2,0.2), degrees=0),
                                transforms.ToTensor()
                                ])
## bottleneck에 필요한 설정값
settings = {'c': [16, 24, 32, 64, 96, 160, 320], 
            't': [1, 6, 6, 6, 6, 6, 6],           
            's': [1, 2, 2, 2, 1, 2, 1],
            'n': [1, 2, 3, 4, 3, 3, 1]
            }
epochs = 100


# 데이터 생성
data_path = glob.glob('vision/training_set/*/*')
idx = torch.randperm(len(data_path))
nvl = int(rvalid * len(data_path))
vldata_path = data_path[0:nvl]
trdata_path = data_path[nvl:]
tedata_path = glob.glob('vision/test_set/*/*')
## 학습
tr_loader = DataLoader(Dataset(trdata_path, classes, transform=transform),
                       batch_size=nbatch,
                       shuffle=True)
## 검증
vl_loader = DataLoader(Dataset(vldata_path, classes, transform=transform),
                       batch_size=nbatch,
                       shuffle=False)
## 평가
te_loader = DataLoader(Dataset(tedata_path, classes, transform=transform),
                       batch_size=nbatch,
                       shuffle=False)

ntr = len(tr_loader)
nvl = len(vl_loader)


# 모델 정의
model = MobileNetV2(block=Bottleneck, settings=settings, mul=.5, nclasses=3)
device = torch.device("cuda")
model.to(device)
# 손실함수와 최적화 함수 정의
lossfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    tr_loss = 0
    tr_accuracy = 0
    
    model.train().cuda()
    
    # for inputs, labels in tr_loader:
    for inputs, labels in iter(tr_loader):
        print(type(inputs))
        print(type(labels))
        inputs, labels = inputs.to(device), labels.to(device)
        # 최적화 함수 gradient zero화
        optimizer.zero_grad()
        # forward
        preds = model(inputs)
        # loss 계산
        loss = lossfn(preds, labels)
        # backward
        loss.backward()
        # 최적화 수행
        optimizer.step()
        
        # accuracy 계산
        pred_labels = torch.argmax(preds, axis=1)
        tr_accuracy += sum(pred_labels == labels) / ntr
        tr_loss += (loss.data.item() * inputs.shape[0]) / ntr
        
    vl_loss = 0
    vl_accuracy = 0
    model.eval().cuda()
    with torch.no_grad():
        for inputs, labels in iter(vl_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # forward
            preds = model(inputs)
            # loss 계산
            loss = lossfn(preds, labels)
            # accuracy 계산
            pred_labels = torch.argmax(preds, axis=1)
            vl_accuracy += sum(pred_labels == labels) / nvl
            vl_loss += (loss.data.item() * inputs.shape[0]) / nvl
        
    print("Epoch {}: train_loss = {:0.6f} , train_accuracy = {:0.3f} , val_loss = {:0.6f} , val_accuracy = {:0.3f}".format(
                epoch+1, tr_loss, tr_accuracy, vl_loss, vl_accuracy))       
    torch.save(model.state_dict(), 'save_weights/checkpoint_gpu_{}'.format(epoch + 1))
            
    