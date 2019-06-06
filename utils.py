#coding:utf-8
import os
import logging
import torch
from torchvision import datasets, transforms
from collections import OrderedDict
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import random

from torchvision import datasets

def save_network(network,path,epoch_label):
    file_path = os.path.join(path,'net_%s.pth' % epoch_label)
    torch.save(network.state_dict(),file_path)  ##Here delete moduel

def load_network(network,path,epoch_label):
    file_path = os.path.join(path,'net_%s.pth' % epoch_label)

    #Original saved file with DataParallel
    state_dict = torch.load(file_path,map_location=lambda storage,loc: storage)
    network.load_state_dict(state_dict,strict=False)

    return network

def getDataloader(dataset,batch_size,part,shuffle=True,augment=True):
    """
    Return the dataloader and imageset of the given dataset

    Returns:
    (torch.utils.data.Dataloader,torchvision.datasets.ImageFolder) --the data loader and the image set 
    """
    transforms_list = [
        transforms.Resize(size=(224,224),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]

    data_transfrom = transforms.Compose(transforms_list)
    image_dataset = datasets.ImageFolder(os.path.join(dataset,part),data_transfrom)
    dataloader = torch.utils.data.DataLoader(image_dataset,batch_size=batch_size,shuffle=shuffle,num_workers=16,drop_last=True)
    print("Done>>>>")

    return dataloader


class Logger(logging.Logger):
    """
    print logs to console and file
    add function to draw the training log curve
    """
    def __init__(self,dir_path):
        self.dir_path = dir_path
        os.makedirs(self.dir_path,exist_ok=True)
        super(Logger,self).__init__("Training logger")

        #print logs to consoles and file 
        file_handler = logging.FileHandler(os.path.join(
            self.dir_path,'train_log.txt'
        ))
        console_handler = logging.StreamHandler()
        log_format = logging.Formatter(
            '%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        self.addHandler(file_handler)
        self.addHandler(console_handler)

        ##Draw curve
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121,title='Training loss')
        self.ax1 = self.fig.add_subplot(122)
        self.x_epoch_loss = []
        self.x_epoch_test = []
        self.y_train_loss = []
        self.y_test = {}
        self.y_test['top1'] = []
        self.y_test['mAP'] = []

    def save_curve(self):
        self.ax0.plot(self.x_epoch_loss,self.y_train_loss,'bs-',markersize='2',label='test')
        self.ax0.set_ylabel("Training")
        self.ax0.set_xlabel('Epoch')
        self.ax0.legend()

        save_path = os.path.join(self.dir_path,'train_log.png')
        self.fig.savefig(save_path)
    
    def save_img(self,fig):
        plt.imsave(os.path.join(self.dir_path,'rank_list.png'),fig)