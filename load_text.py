#coding:utf-8
import numpy as np 
import os 
import torch
import torch.nn as nn 
import random 
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F 
import torch.optim as optim

def load_data(dataset_path,part,batch_size):
    all_path = []
    data_path = os.path.join(dataset_path,part)
    singal = os.listdir(data_path)
    key=0
    for fsingal in singal:
        filepath = os.path.join(data_path,fsingal)
        filename = os.listdir(filepath)
        for fname in filename:
            ffpath = filepath+'/'+fname
            path = [key,ffpath]
            all_path.append(path)
        key+=1
        
    count = len(all_path)
    data_x = np.empty((count,300,1,44),dtype='float32')
    data_y = []

    random.shuffle(all_path)

    i=0
    for item in all_path:
        img = np.load(item[1])
        img_a = img[np.newaxis,:]
        img_b = img_a.transpose((2,0,1))

        data_x[i,:,:,:] = img_b
        i+=1
        data_y.append(int(item[0]))

    data_y = np.asarray(data_y)

    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    dataset = dataf.TensorDataset(data_x,data_y)
    loader = dataf.DataLoader(dataset,batch_size,shuffle=True,num_workers=16,drop_last=True)
    print("done##")
    return loader



if __name__ == "__main__":
    get = load_data('./data_test','val_npy',8)
    j=1
    print('get_length:: ',len(get))
    print(len(get.dataset))
    print('get_type：',type(get))
    print(len(get.dataset))
    for data,label in get:
        print("data:  ",data.shape)
        print("label:  ",label.data)
