#coding:utf-8
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import models

import torch    
import torch.nn as nn 
import torch.nn.functional as F 
from tensorboardX import SummaryWriter

class ResidualBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel//2,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(outchannel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel//2,outchannel,kernel_size=(1,stride),stride=(1,stride),padding=0,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel,outchannel,kernel_size=(1,stride),stride=(1,stride),padding=0,bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self,x):
        out = self.left(x)
        out_short= self.shortcut(x)
        out += out_short
        out = F.relu(out)
        return out

class net_text(nn.Module):
    def __init__(self,ResidualBlock=ResidualBlock,num_classes=10,mode='train'):
        super(net_text,self).__init__()
        self.inchannel = 300
        self.mode = mode

        self.layer1 = self.make_layer(ResidualBlock,256,1,inchannel=300)
        self.layer2 = self.make_layer(ResidualBlock,512,2,inchannel=256)
        self.layer3 = self.make_layer(ResidualBlock,1024,2,inchannel=512)
        self.layer4 = self.make_layer(ResidualBlock,2048,1,inchannel=1024)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(
            nn.Linear(2048,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True))

        self.dropout = nn.Dropout(0.75)
        
    def make_layer(self,block,channels,stride,inchannel):
        layers = []
        strides = [stride]
        for stride in strides:
            layers.append(block(inchannel,channels,stride))
            inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.dropout(out)

        return out

class net_image(nn.Module):
    def __init__(self,numclass,mode='train'):
        super(net_image,self).__init__()
        self.num_class = numclass
        self.mode = mode
        resnet = models.resnet50(pretrained=True)

        self.backbone = nn.Sequential(*(list(resnet.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if self.mode == 'train':
            for p in self.parameters():
                p.requires_grad = False

        self.fc1 = nn.Sequential(
            nn.Linear(resnet.fc.in_features,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(0.75)

    def forward(self,x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.dropout(x)

        return x

class Merge_image_text(nn.Module):
    def __init__(self,num_class,mode='train'):
        super(Merge_image_text,self).__init__()
        self.num_class = num_class
        self.mode = mode
        self.image_feature = net_image(self.num_class,self.mode)
        self.text_feature = net_text(ResidualBlock=ResidualBlock,num_classes=self.num_class,mode=self.mode)
        self.fc = nn.Linear(2048,self.num_class)
    
    def forward(self,x,y):
        """
        The first stage only train on the text
        """
        image_out = self.image_feature(x)
        text_out = self.text_feature(y)

        if self.mode == 'test':
            return image_out,text_out
        
        fc_img = self.fc(image_out)
        fc2 = self.fc(text_out)
        
        return fc_img,fc2

if __name__ == "__main__":
    input_data1 = Variable(torch.rand(64,3,224,224))
    input_data2 = Variable(torch.FloatTensor(64,300,1,32))

    net = Merge_image_text(10000,'test')
    # net = net_image(10000,'train')
    # net = net_text(ResidualBlock=ResidualBlock,num_classes=10000)
    print(net)
    img_get,text_get = net(input_data1,input_data2)
    print("text_get.shape:: ",text_get.shape)
    print('text_get.shape:: ',img_get.shape)