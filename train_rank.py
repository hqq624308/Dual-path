# -*- coding: utf-8 -*-
from __future__  import print_function,division

import os
import time
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets,transforms
from load_text import load_data
from rank_loss import ImageSelector,TextSelector
from loss import TripletLoss

from model_rank import Merge_image_text
from test_acc import test
import utils

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--save_path',type=str,default='./model2_56')
parser.add_argument('--datasets',type=str,default='./data_56')
parser.add_argument('--batch_size',type=int,default=32,help='batch_size') 
parser.add_argument('--learning_rate',type=float,default=0.001,help = 'FC parms learning rate')
parser.add_argument('--epochs',type=int,default=120,help='The number of epochs to train')
parser.add_argument('--stage',type=str,default='I',choices=['I','II'],help='which stage is on')

arg = parser.parse_args()

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

#Make saving directory
save_dir_path = os.path.join(arg.save_path,arg.datasets)
os.makedirs(save_dir_path,exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# -------------------------------------Train Function--------------------------------------
def train_rank(model,criterion,optimizer,scheduler,dataloder,text_loader,num_epochs,device,stage):
    start_time = time.time()
    # Logger instance
    logger = utils.Logger(save_dir_path)
    logger.info('-'*10)
    logger.info(vars(arg))
    logger.info('Stage: '+stage)
    print("############################ Train stage II #############################")
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1,num_epochs))
        model.train()
        scheduler.step()
        ##Training 
        batch_num = 0
        loss_avg = []

        get = list(zip(dataloder,text_loader))
        random.shuffle(get)
        img,txt = zip(*get)

        for (inputs,labels),(text_inputs,text_labels) in zip(img,txt):

            batch_num += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            text_inputs = text_inputs.to(device)
            text_labels = text_labels.to(device,dtype=torch.int64)

            outputs,text_outs = model(inputs,text_inputs)
            
            # print("output.shape:: ",outputs.shape)
            # print("text_out.shape:: ",text_outs.shape)
            # print("label.shape: ",labels)
            # print("text_label.shape:: ",text_labels)

            anc_IT,pos_IT,neg_IT = ImageSelector(outputs,text_outs,labels)
            anc_TI,pos_TI,neg_TI = TextSelector(text_outs,outputs,labels)

            loss_rank = criterion(anc_IT,pos_IT,neg_IT)+criterion(anc_TI,pos_TI,neg_TI)
            optimizer.zero_grad()
            loss_rank.backward()
            optimizer.step()

            loss_avg.append(loss_rank)
            if batch_num % 10 == 0:
                loss_avg= sum(loss_avg) /len(loss_avg)
                logger.info('Stage II training : {} [{}]]\t Rank_loss:{:.6f}'.format(epoch+1,batch_num*len(inputs),loss_avg))
                loss_avg = []
        
        if (epoch+1)%2==0 or epoch+1 == num_epochs:
            ##Testing / Vlidaing
            torch.cuda.empty_cache()
            # model.mode = 'test'
            CMC,mAP = test(model,arg.datasets,128)
            logger.info('Testing: Top1:%.2f Top5:%.2f Top10:%.2f mAP:%.2f' % (CMC[0],CMC[4],CMC[9],mAP))

        logger.info('-'*10)
    time_cost = time.time()-start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_cost//60,time_cost%60
    ))
    utils.save_network(model,save_dir_path,'final_r')

def getDataloader(dataset,batch_size,part,shuffle=True):
    """
    return dataset loader
    """
    transforms_list = [
        transforms.Resize(size=(224,224),interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]

    data_transfrom = transforms.Compose(transforms_list)
    image_dataset = datasets.ImageFolder(os.path.join(dataset,part),data_transfrom)
    # dataloader = torch.utils.data.DataLoader(image_dataset,batch_size=batch_size,shuffle=shuffle,num_workers=16,drop_last=True)
    print("Done>>>>")

    return image_dataset

from loader import ClassUniformlySampler
from load_txt_rank import load_data
import random

class IterLoader:
    def __init__(self,loader):
        self.loader=loader
        self.iter = iter(self.loader)
    
    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Loader image dataset ,PK samples
    seeds = random.randint(0,100)
    datasets_img = getDataloader(arg.datasets,arg.batch_size,'train',shuffle=False)
    loader = torch.utils.data.DataLoader(datasets_img,batch_size=32,num_workers=16,drop_last=True,
                                        sampler=ClassUniformlySampler(datasets_img,class_position=1,k=4,seeds = seeds))
    # dataloader_img = IterLoader(loader)
    # print('labels_img: ',dataloader_img.next_one()[1])

    ##Loader txt dataset , PK samples 
    dataset_text = load_data(arg.datasets,'train_npy',arg.batch_size)
    loader_txt = torch.utils.data.DataLoader(dataset_text,batch_size=32,num_workers=16,drop_last=True,
                                            sampler=ClassUniformlySampler(dataset_text,class_position=1,k=4,seeds=seeds))
    # dataloader_txt = IterLoader(loader_txt)
    # print('dataloader_txt: ',dataloader_txt.next_one()[1])

    ##############################################
    model = Merge_image_text(num_class=11003,mode = 'test')   #Stage II ,change to 'test',Stage I:'train'
    model = model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = TripletLoss(margin = 1).cuda()  #no margin means soft-margin
 
    #delete module parallel
    optimizer = optim.SGD([
        {'params':model.image_feature.backbone.parameters(),'lr':arg.learning_rate},
        # {'params':model.image_feature.fc1.parameters(),'lr':arg.learning_rate},
        # {'params':model.image_feature.fc.parameters(),'lr':arg.learning_rate},
        {'params':model.text_feature.parameters(),'lr':arg.learning_rate/10}
    ],lr=0.001,momentum=0.9,weight_decay = 5e-4,nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)

    #---------------------Start training----------------------
    #Stage I
    # train(model,criterion,optimizer,scheduler,train_dataloder,train_dataloder_text,arg.epochs,device,'I')
    ##Stage II
    model.load_state_dict(torch.load('./model2_56/data_56/net_final.pth'),strict=False)
    train_rank(model,criterion,optimizer,scheduler,loader,loader_txt,arg.epochs,device,'II')
    torch.cuda.empty_cache()