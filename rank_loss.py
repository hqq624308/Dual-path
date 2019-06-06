#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
# from utils import pdist_torch as pdist

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


def ImageSelector(embedsImage, embedsText, labels):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    Here is ::  Image-Text
    '''
    dist_mtx = pdist_torch(embedsImage, embedsText).detach().cpu().numpy()
    labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
    num = labels.shape[0]
    dia_inds = np.diag_indices(num)
    lb_eqs = labels == labels.T
    lb_eqs[dia_inds] = False
    dist_same = dist_mtx.copy()
    dist_same[lb_eqs == False] = -np.inf
    pos_idxs = np.argmax(dist_same, axis = 1)
    dist_diff = dist_mtx.copy()
    lb_eqs[dia_inds] = True
    dist_diff[lb_eqs == True] = np.inf
    neg_idxs = np.argmin(dist_diff, axis = 1)
    pos = embedsText[pos_idxs].contiguous().view(num, -1)
    neg = embedsText[neg_idxs].contiguous().view(num, -1)
    return embedsImage, pos, neg

def TextSelector(embedsImage, embedsText, labels):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    Here is ::  Text-Image
    '''
    dist_mtx = pdist_torch(embedsImage, embedsText).detach().cpu().numpy()
    labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
    num = labels.shape[0]
    dia_inds = np.diag_indices(num)
    lb_eqs = labels == labels.T
    lb_eqs[dia_inds] = False
    dist_same = dist_mtx.copy()
    dist_same[lb_eqs == False] = -np.inf
    pos_idxs = np.argmax(dist_same, axis = 1)
    dist_diff = dist_mtx.copy()
    lb_eqs[dia_inds] = True
    dist_diff[lb_eqs == True] = np.inf
    neg_idxs = np.argmin(dist_diff, axis = 1)
    pos = embedsImage[pos_idxs].contiguous().view(num, -1)
    neg = embedsImage[neg_idxs].contiguous().view(num, -1)
    return embedsText, pos, neg

if __name__ == '__main__':
    embedimg = torch.randn(10, 128)
    embedtxt = torch.randn(10, 128)

    labels = torch.tensor([0,1,2,2,0,1,2,1,1,0])

    # selector = ImageSelector()
    anchor, pos, neg = ImageSelector(embedimg, embedtxt ,labels)
    print('anchor.shape:: ',anchor.shape)
    print('pos.shape :: ',pos.shape)
    print('neg.shape :: ',neg.shape)