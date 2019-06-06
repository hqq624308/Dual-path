#coding:utf-8
import numpy as np 
import gensim 
import os
import string,re
from tqdm import tqdm

# save_path = './data/test_npy'
# path = './data/text_test'

# save_path = './data/train_npy'
# path = './data/text_train'

save_path1 = './data_56/val_npy'
path1 = './data_56/text_val'

# os.makedirs(save_path,exist_ok=True)
os.makedirs(save_path1,exist_ok=True)

#Delete (,.)
regex = re.compile('[%s]' % re.escape(string.punctuation))
def test_re(s): 
	return regex.sub('', s)

model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin',binary = True)
padding = [0]*300

# items = os.listdir(path)
# for i in tqdm(items):
#     with open(path+'/'+i,'r') as f:
#         line = f.readlines()
#         # print("Item: 　",i)
#         j = 0
#         for li in line:
#             lis = test_re(li)    #delete ',.'
#             content = lis.strip('\n').split(' ')
#             feature_map = []
#             num = np.random.randint(10)   ## Random start position
#             feature_map.extend([padding]*num)   ####position shift 
#             for word in content:
#                 if word in model.vocab:
#                     feature = model[word]
#                     if len(feature_map)<44:
#                         feature_map.append(feature)
#                     else:
#                         break      
#                 else:
#                     continue
                  
#             while len(feature_map)<44:
#                 feature_map.append(padding)

#             if not os.path.exists(save_path+"/"+i[:-4]):
#                 os.mkdir(save_path+'/'+i[:-4])
#             np.save(save_path+'/'+i[:-4]+'/'+str(j)+'.npy',feature_map)
#             j+=1



#######################################################
print("#"*20)
items = os.listdir(path1)
for i in tqdm(items):
    with open(path1+'/'+i,'r') as f:
        line = f.readlines()
        j = 0
        for li in line:
            lis = test_re(li)    #delete ',.'
            content = lis.strip('\n').split(' ')
            feature_map = []
            num = np.random.randint(10)   ## Random start position
            feature_map.extend([padding]*num)   ####position shift 
            for word in content:
                if word in model.vocab:
                    feature = model[word]
                    if len(feature_map)<56:
                        feature_map.append(feature)
                    else:
                        break      
                else:
                    continue
                  
            while len(feature_map)<56:
                feature_map.append(padding)

            if not os.path.exists(save_path1+"/"+i[:-4]):
                os.mkdir(save_path1+'/'+i[:-4])
            np.save(save_path1+'/'+i[:-4]+'/'+str(j)+'.npy',feature_map)
            j+=1