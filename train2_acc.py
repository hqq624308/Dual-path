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
from load_text56 import load_data
from rank_loss import ImageSelector,TextSelector
from loss import TripletLoss

from model import Merge_image_text
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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '2'

# -------------------------------------Train Function--------------------------------------
def train(model,criterion,optimizer,scheduler,dataloder,text_loader,num_epochs,device,stage):
    start_time = time.time()

    # Logger instance
    logger = utils.Logger(save_dir_path)
    logger.info('-'*10)
    logger.info(vars(arg))
    logger.info('Stage: '+stage)

    print("################################### Train stage I ######################################")
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1,num_epochs))

        model.train()
        scheduler.step()

        ##Training 
        running_loss = 0.0
        running_text_loss = 0.0
        batch_num = 0
        img_cor = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        txt_cor = torch.zeros(1).squeeze().cuda()
        txt_total = torch.zeros(1).squeeze().cuda()
    
        for (inputs,labels),(text_inputs,text_labels) in zip(dataloder,text_loader):
            batch_num += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            text_inputs = text_inputs.to(device)
            text_labels = text_labels.to(device,dtype=torch.int64)

            outputs,text_outs = model(inputs,text_inputs)

            ###Intance loss
            loss = criterion(outputs,labels)
            text_loss = criterion(text_outs,text_labels)
            optimizer.zero_grad()
            loss.backward()
            text_loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)
            running_text_loss += text_loss.item()*text_inputs.size(0)

            #Accurate
            img_pre = torch.argmax(outputs,1)
            img_cor += (img_pre == labels).sum().float()
            total += len(labels)
            
            txt_pre = torch.argmax(text_outs,1)
            txt_cor += (txt_pre == text_labels).sum().float()
            txt_total += len(text_labels)

            if batch_num % 10 == 0:
                logger.info('Train image epoch : {} [{}/{}]\t Image Loss:{:.6f}\t || Text Loss:{:.6f}'.format(epoch+1,batch_num*len(inputs),len(dataloder.dataset.imgs),
                running_loss/(batch_num*arg.batch_size),running_text_loss/(batch_num*arg.batch_size)))
                
        # logger.info("Img_acc: {:.2f} \t   Txt_acc: {:.2f}".format((img_cor/total).cpu().detach().data.numpy(),(txt_cor/txt_total).cpu().detach().data.numpy()))
        logger.info('Epoch {}:Done!!!'.format(epoch+1))

        loss_val_runing = 0.0
        loss_val_runing_text = 0.0

        img_cor_val = torch.zeros(1).squeeze().cuda()
        txt_cor_val = torch.zeros(1).squeeze().cuda()
        if (epoch+1) % 2 == 0 or epoch+1 == num_epochs:
            ##Testing / Vlidating
            torch.cuda.empty_cache()
            model.mode = 'test'
            CMC,mAP = test(model,arg.datasets,128)
            logger.info('Testing: Top1:%.2f Top5:%.2f Top10:%.2f mAP:%.2f' % (CMC[0],CMC[4],CMC[9],mAP))
            model.mode = 'train'

            # gallery_dataloder = utils.getDataloader(
            #     arg.datasets,arg.batch_size,'val',shuffle=False,augment=False
            # )
            # text_dataloder = load_data(arg.datasets,'val_npy', arg.batch_size)

            # for (inputs_val,label),(text_inputs_val,text_label) in zip(gallery_dataloder,text_dataloder):
            #     inputs_val = inputs_val.to(device)
            #     label = label.to(device)
            #     text_inputs_val = text_inputs_val.to(device)
            #     text_label = text_label.to(device,dtype=torch.int64)
                
            #     img_val_out,text_outputs_val = model(inputs_val,text_inputs_val)
                
            #     loss_img = criterion(img_val_out,label)
            #     loss_val_text = criterion(text_outputs_val,text_label)
            #     loss_val_runing += loss_img.item()*inputs_val.size(0)
            #     loss_val_runing_text += loss_val_text.item()*text_inputs_val.size(0)

            #     #Accurate
            #     img_pre_val = torch.argmax(img_val_out,1)
            #     img_cor_val += (img_pre_val == label).sum().float()
            #     ###
            #     txt_pre_val = torch.argmax(text_outputs_val,1)
            #     txt_cor_val += (txt_pre_val == text_label).sum().float()
            

            # # logger.info("Img_VAL_acc: {:.2f} \t   Txt_VAL_acc: {:.2f}".format(
            # #     (img_cor_val/len(gallery_dataloder.dataset.imgs)).cpu().detach().data.numpy(),(txt_cor_val/len(text_dataloder.dataset)).cpu().detach().data.numpy()))
            # result = loss_val_runing/len(gallery_dataloder.dataset.imgs)
            # logger.info('[**]Validing image Loss: {:.4f}'.format(result))
            
            # result_text = loss_val_runing_text/len(text_dataloder.dataset)  ###Note
            # logger.info('[**]Validing text Loss: {:.4f}'.format(result_text))

        logger.info('-'*10)
    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60,time_elapsed%60
    ))
    #Save final model weithts
    utils.save_network(model,save_dir_path,'final')

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##It is a classifier task for image 
    train_dataloder = utils.getDataloader(arg.datasets,arg.batch_size,'train',shuffle=True,augment=True)
    train_dataloder_text = load_data(arg.datasets,'train_npy',arg.batch_size)

    model = Merge_image_text(num_class=len(train_dataloder.dataset.classes),mode = 'train')   #Stage II ,change to 'test',Stage I:'train'

    # if torch.cuda.device_count() > 1:
    #     print("Let's use",torch.cuda.device_count(), "GPUs")
    #     model = nn.DataParallel(model,device_ids =[0,1,2])

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    ignore_param = list(map(id,model.image_feature.backbone.parameters()))
    base_param = filter(lambda p: id(p) not in ignore_param,model.parameters())
    optimizer = optim.SGD([
        {'params':base_param,'lr':0.001},
        {'params':model.image_feature.backbone.parameters(),'lr':0.00001},
    ],momentum=0.9,weight_decay = 5e-4,nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)

    #---------------------Start training----------------------
    #Stage I
    train(model,criterion,optimizer,scheduler,train_dataloder,train_dataloder_text,arg.epochs,device,'I')
    ##Stage II
    # model.load_state_dict(torch.load('./model_test_save/data_test/net_final.pth'))
    # train_rank(model,criterion,optimizer,scheduler,train_dataloder,train_dataloder_text,arg.epochs,device,'II')
    torch.cuda.empty_cache()