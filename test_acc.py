# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import argparse
import torch 
import torch.nn as nn 
from torch.autograd import Variable
from torchvision import datasets ,transforms
from sklearn.metrics import average_precision_score
# from model import Merge_image_text
from model import Merge_image_text
import utils
from load_text56 import load_data

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'

def extract_feature(model,inputs,require_norm,vectorize,require_grad=False):
    ##Move to model's device
    inputs = inputs.to(next(model.parameters()).device)

    with torch.set_grad_enabled(require_grad):
        features = model(inputs)

    size = features.shape

    if require_norm:
        #[N,C*H]
        features = features.view(size[0],-1)

        #norm feature
        fnorm = features.norm(p=2,dim=1)
        features = features.div(fnorm.unsqueeze(dim=1))

    if vectorize:
        features = features.view(size[0],-1)
    else:
        #Back to [N,C,H=S]
        features = features.view(size)
    
    return features


def evaluate(query,gallery,query_labels,gallery_labels):
    """
    Evaluate the CMC and mAP

    Returns:
    torch.IntTensor,float  --- CMC list,mAP
    """
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    AP = 0
    print('CMC.shape: ',CMC.shape)
    print("Gallrery_label::",gallery_labels)
    
    for i in range(len(query_labels)):
        query_feature = query[i]
        query_label = query_labels[i]
        # print('query_label: ',query_label)

        #Prediction score 
        score = np.dot(gallery,query_feature)
        # print('score:: ',score)
        # print('score.shape:: ',score.shape)
        match_query_index = np.argwhere(gallery_labels == query_label)  #Find the match index 
        # print('match_query_index:: ',match_query_index)
        index = np.arange(len(gallery_labels))

        ##compute AP  ---Q: Why not sort the sore when compute the AP
        y_true = np.in1d(index,match_query_index)
        y_score = score[index]
        AP += average_precision_score(y_true,y_score)

        ##compute CMC 
        ##Sort the index by their scores,from large to small
        sorted_index = np.argsort(y_score)[::-1]
        sort_y_true = y_true[sorted_index]
        match_index = np.argwhere(sort_y_true==True)

        if match_index.size > 0:
            first_match_index = match_index.flatten()[0]
            CMC[first_match_index:] += 1

        CMC = CMC.float()
        CMC = CMC / len(query_labels) * 100
        mAP = AP / len(query_labels) * 100

        return CMC,mAP

def evaluate_my(query,gallery,query_labels,gallery_labels):
    """
    Evaluate the CMC and mAP

    Returns:
    torch.IntTensor,float  --- CMC list,mAP
    """
    
    print("Start to evaluate the Model>>>>")
    AP = 0
    CMC = []
    
    for i in range(len(query_labels)):
        query_feature = query[i]
        query_label = query_labels[i]
        # print('query_label: ',query_label)

        #Prediction score 
        score = np.dot(gallery,query_feature)
        # print('score:: ',score)
        # print('score.shape:: ',score.shape)
        match_query_index = np.argwhere(gallery_labels == query_label)  #Find the match index 
        # print('match_query_index:: \n',match_query_index)
        # index = np.arange(len(gallery_labels))
        sorted_index = np.argsort(score)[::-1]

        #CMC
        y_true = np.in1d(sorted_index,match_query_index)
        # y_mask = [0]*len(y_true)
        y_mask = []
        ##Ap
        t= 0
        tems = []
        key = 0
        for i in range(len(y_true)):
            if y_true[i]:
                t+=1
                tem = t/(i+1)
                tems.append(tem)
                key = 1
                y_mask.append(key)
            else:
                y_mask.append(key)
        AP += np.mean(tems)
        CMC.append(y_mask)
    
    CMC_ = np.mean(CMC,axis=0)*100
    mAP = AP / len(query_labels)*100
    return CMC_,mAP

##---------------------------Starting testing -------------------------------------
def test(model,datasets,batch_size,requires_grad=False):
    model.eval()
    print("model.mode: ",model.mode)
    gallery_dataloder = utils.getDataloader(
        datasets,batch_size,'test',shuffle=False,augment=False)

    query_text_dataloder = load_data(datasets,'test_npy', batch_size)
    
    galleyr_features = []
    query_features = []
    img_label = []
    text_label = []
    print("Test data number:: ",len(gallery_dataloder.dataset.imgs))
    print("Dataloader Done!!")

    for (data_inputs,label_inputs),(text_data_inputs,text_label_inputs)  in zip(gallery_dataloder,query_text_dataloder):
        data_inputs = data_inputs.to(next(model.parameters()).device)
        label_inputs = label_inputs.to(next(model.parameters()).device)
        text_data_inputs = text_data_inputs.to(next(model.parameters()).device)
        text_label_inputs = text_label_inputs.to(next(model.parameters()).device,dtype=torch.int64)

        # print('data_inputs.shape:: ',data_inputs.shape)
        # print('text_data_inputs.shape:: ',text_data_inputs.shape)
        # print('label_input: ',label_inputs)
        # print("text_label: ",text_label_inputs)
        
        with torch.set_grad_enabled(requires_grad):
            image_feature,text_feature = model(data_inputs,text_data_inputs)

        ##Normalization 
        image_feature = image_feature.view(image_feature.shape[0],-1)
        fnorm = image_feature.norm(p=2,dim=1)
        image_feature=image_feature.div(fnorm.unsqueeze(dim=1))

        text_feature = text_feature.view(text_feature.shape[0],-1)
        tnorm = text_feature.norm(p=2,dim=1)    ###在指定维度1上计算2范数
        text_feature = text_feature.div(tnorm.unsqueeze(dim=1))

        query_features.append(text_feature)
        galleyr_features.append(image_feature)
        img_label.append(label_inputs)
        text_label.append(text_label_inputs)
    
    galleyr_features = torch.cat(galleyr_features,dim=0).detach().cpu().numpy()
    query_features = torch.cat(query_features,dim=0).detach().cpu().numpy()
    img_label = torch.cat(img_label,dim=0).detach().cpu().numpy()
    text_label = torch.cat(text_label,dim=0).detach().cpu().numpy()
    
    print("Data process Done>>")
    print("query_feature.shape: ",query_features.shape)
    print('gallery_feature.shape: ',galleyr_features.shape)
    print("labels.length ： ",len(img_label))
    print("text_label: ",len(text_label))

    cmc,mAP = evaluate_my(query_features,galleyr_features,text_label,img_label)
    # cmc,mAP = evaluate(query_features,galleyr_features,text_label,img_label)

    return cmc ,mAP

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing arugments')
    parser.add_argument('--save_path',type=str,default='./model_3')
    parser.add_argument('--which_epoch',default='final',type = str,help='0,1,2,3,...or final')
    parser.add_argument('--dataset',type=str,default='data')
    parser.add_argument('--batch_size',default=32,type=int,help='batch_size')
    parser.add_argument('--mode_type',default='test',type=str)
    arg = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(arg.save_path,arg.dataset)
    print('Save_path :: ',save_path)
    logger = utils.Logger(save_path)

    # model = utils.load_network(Merge_image_text(num_class=len(train_dataloader.dataset.classes),mode=arg.mode_type),
    #                             save_path,arg.which_epoch)
    model = utils.load_network(Merge_image_text(num_class=11003,mode=arg.mode_type),save_path,arg.which_epoch)
    model = model.to(device)

    CMC,mAP = test(model,arg.dataset,arg.batch_size)
    logger.info('Testing: top1:%.2f top5:%.2f top10:%.2f mAP:%.2f' % (CMC[0], CMC[4], CMC[9], mAP))

    torch.cuda.empty_cache()