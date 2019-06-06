#coding:utf-8
#This function is used for split data to train/val/test
import numpy as  np 
import json 
import os
import shutil
from tqdm import tqdm 

base_path = './data/img256/'

with open('./data/reid_raw.json','r') as load_f:
    load_dict = json.load(load_f)

print(type(load_dict))
print(type(load_dict[0]))
print(load_dict[0])
print(len(load_dict))
print(load_dict[-1]['id'])

# train_path = './data/train'
# val_path = './data/val'
# test_path = './data/test'

# text_train_path = './data/text_train'
# text_val_path = './data/text_val'
# text_test_path = './data/text_test'

val_path = './data_56/val'
text_val_path = './data_56/text_val'

# if not os.path.exists(train_path):
#     os.mkdir(train_path)
#     os.mkdir(val_path)
#     os.mkdir(test_path)

#     os.mkdir(text_train_path)
#     os.mkdir(text_val_path)
#     os.mkdir(text_test_path)

# os.mkdir(val_path)
os.mkdir(text_val_path)

train_num = 0
test_num = 0
val_num = 0

for i in tqdm(range(len(load_dict))):
    filepath = os.path.join(base_path,load_dict[i]['file_path'])
    split = load_dict[i]['split']
    caption = load_dict[i]['captions']
    id = load_dict[i]['id']

    name = load_dict[i]['file_path'].split('/')[-1]

    # if split == 'train':
    #     if not os.path.exists(train_path+'/'+str(id)):
    #         os.mkdir(train_path+'/'+str(id))
    #     if not os.path.exists(train_path+'/'+str(id)+'/'+name):
    #         shutil.copy(filepath,train_path+'/'+str(id)+'/'+name)
    #     else:
    #         num = len(os.listdir(train_path+'/'+str(id)))
    #         shutil.copy(filepath,train_path+'/'+str(id)+'/'+str(num)+'_'+name)

    #     with open('./data/text_train/'+str(id)+'.txt','a') as f:
    #         f.write(caption[0].strip('\n').replace('\n',' '))
    #         f.write('\n')
    #         train_num += 1

    if split == 'val':
        # if not os.path.exists(val_path+'/'+str(id)):
        #     os.mkdir(val_path+'/'+str(id))
        # if not os.path.exists(val_path+'/'+str(id)+'/'+name):
        #     shutil.copy(filepath,val_path+'/'+str(id)+'/'+name)
        # else:
        #     num = len(os.listdir(val_path+'/'+str(id)))
        #     shutil.copy(filepath,val_path+'/'+str(id)+'/'+str(num)+'_'+name)

        with open(text_val_path+'/'+str(id)+'.txt','a') as f:
            f.write(caption[0].strip('\n').replace('\n',' ') + caption[1].strip('\n').replace('\n',' '))
            f.write('\n')
            val_num += 1

    # if split == 'test':
    #     if not os.path.exists(test_path+'/'+str(id)):
    #         os.mkdir(test_path+'/'+str(id))
    #     if not os.path.exists(test_path+'/'+str(id)+'/'+name):
    #         shutil.copy(filepath,test_path+'/'+str(id)+'/'+name)
    #     else:
    #         num = len(os.listdir(test_path+'/'+str(id)))
    #         shutil.copy(filepath,test_path+'/'+str(id)+'/'+str(num)+'_'+name)

    #     with open('./data/text_test/'+str(id)+'.txt','a') as f:
    #         f.write(caption[0].strip('\n').replace('\n',' '))
    #         f.write('\n')
    #         test_num += 1
        
# print('test_txt: ',str(test_num))
# print("train_num : ",str(train_num))
print('val_num: ',str(val_num))