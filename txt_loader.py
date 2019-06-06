#coding:utf-8
import numpy as np 
import os 
import copy 
import torch
import torch.nn as nn 

class Txtsample:
    def __init__(self,path,part):
        self.path = path 
        self.part = part 
        self.sample = self.__getSample__()

    def __getSample__(self):
        all_path = []
        data_path = os.path.join(self.path,self.part)
        singal = os.listdir(data_path)
        key = 0 
        for fsingal  in singal:
            filepath = os.path.join(data_path,fsingal)
            filename = os.listdir(filepath)
            for fname in filename:
                ffpath = filepath + '/' + fname
                all_path.append([key,ffpath])
            key+=1
        
        return all_path

    def __loadData__(self):
        
        pass 