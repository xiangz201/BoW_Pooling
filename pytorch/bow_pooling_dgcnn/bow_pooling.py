#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Xiang Zhang
@Contact: 1801210733@pku.edu.cn
@File: bow_pooling.py
"""
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans


def similarity_matrix(inputs, dictionary, stype='cosine'):
    """ calculate the similarity between local descriptors and clustering centers
        Args:
            inputs: N points position data, 3-D tensor, [B, C, N]
            dictionary: K clustering centers, 3-D tensor, [B, K, C]
            stype : similarity metric, str, default: 'cosine'
        Return:
            outputs: the similarity matrix, 3-D tensor, [B, K, N]
    """
    if stype == 'cosine':
        outputs = torch.matmul(dictionary, inputs)
    elif stype == 'l1':
        inputs = inputs.permute(0, 2, 1)
        outputs = dictionary[:, :, None] - inputs[:, None]
        outputs = torch.abs(outputs)
        outputs = outputs.sum(dim=-1, keepdim=False)
    elif stype == 'l2':
        inputs = inputs.permute(0, 2, 1)
        outputs = dictionary[:, :, None] - inputs[:, None]
        outputs = torch.sqrt(outputs)
        outputs = outputs.sum(dim=-1, keepdim=False)
    return outputs


def tlu(inputs, p):
    """ suppress the expression of unimportant local descriptors
        Args:
            inputs: the similarity matrix, 3-D tensor, [B, K, N]
            p: the percentage of points that are kept, float
        Return:
            outputs: the similarity matrix after suppression, 3-D tensor, [B, K, N]
    """
    B, num_dim, num = inputs.size()
    device = torch.device('cuda')
    weight = torch.zeros_like(inputs, device=device)
    l = int((num + 1) * p)
    flag = inputs.topk(k=l, dim=1)
    flag = flag[0][:, l - 1, :].unsqueeze(dim=1).repeat(1, inputs.size(1), 1)
    weight[inputs >= flag] = 1.0
    weight[inputs < flag] = 0.0
    outputs = weight * inputs
    return outputs


def aggregation(inputs, atype='sum'):
    """ aggregate the similarity matrix to form a histogram
        Args:
            inputs: similarity matrix, 3-D tensor, [B, K, N]
            atype: aggregation method, str, default: 'sum'
        Return:
            outputs: a histogram shows how the primitives construct 3D objects, 2-D tensor, [B, K]
    """
    if atype == 'sum':
        outputs = inputs.sum(dim=-1, keepdim=False)
        # return outputs
    elif atype == 'max':
        outputs = inputs.max(dim=-1, keepdim=False)[0]
    return outputs


def update_dictionary(args, device, model, train_loader):
    """ cluster dictionary from train_loader with KMeans algorithm
        Args:
            args: global args with item 'dictionary', or use global dictionary
            device: torch.device('cuda')
            model: model of network
            train_loader: torch.utils.data.DataLoader, in it: data [B, C, N], label [B, 1]
    """
    # Start cluster data
    count = 0
    batch_size = 0
    num_points = 0
    dictionary_x = []
    for data, label in train_loader:
        count = count + 1
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size(0)
        num_points = data.size(2)
        _, x = model(data)
        dictionary_x.append(x.cpu().detach().numpy())
    dictionary = []
    # use random selected positions features, less time and comparable effect
    for i in np.random.randint(0, count, 25):
        for j in np.random.randint(0, batch_size, 10):
            for k in np.random.randint(0, num_points, 200):
                dictionary.append(dictionary_x[i][j, :, k])
    """
    # use all positions features
    for i in range(count):
        for j in range(batch_size):
            for k in range(num_points):
                dictionary.append(dictionary_x[i][j, :, k])
    """
    del dictionary_x, data, label, batch_size, count
    dictionary = np.array(dictionary)
    mm = MiniBatchKMeans(n_clusters=args.dictionary.size(0), init_size=args.dictionary.size(0) * 3).fit(dictionary)
    del dictionary
    cluster_center = torch.from_numpy(mm.cluster_centers_).to(device)
    np.savetxt("dictionary.txt", cluster_center.cpu().detach().numpy(), fmt='%f', delimiter=',')
    args.dictionary = cluster_center
    del cluster_center
    # Finish cluster data


def EM_Update(args, device, model, train_loader):
    """update dictionary with EM algorithm
        Args:
            args: global args
            device: torch.device('cuda')
            model: model of network
            train_loader: data
    """
    # Start update dictionary with EM algorithm
    update_dictionary(args, device, model, train_loader)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in model_dict.items() if 'dictionary' in k}
    for key in pretrained_dict:
        pretrained_dict[key] = args.dictionary
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    del pretrained_dict, model_dict
    # os.system('cp dictionary.txt checkpoints' + '/' + args.exp_name + '/' + 'dictionary.txt')
    # Finish update dictionary with EM algorithm


class Bow_Pooling(nn.Module):
    """ bow pooling implementation"""

    def __init__(self, dictionary, p=0.5, stype='cosine', atype='sum'):
        super(Bow_Pooling, self).__init__()
        self.dictionary = dictionary
        self.p = p
        self.stype = stype
        self.atype = atype

    def forward(self, inputs):
        """
        Input:
            inputs: input points position data, [B, C, N]
        Return:
            outputs: a histogram shows how the primitives construct 3D objects, 2-D tensor, [B, K]
        """
        batch_size = inputs.size(0)
        dictionary = self.dictionary.unsqueeze(dim=0)
        dictionary = dictionary.repeat(batch_size, 1, 1)
        inputs = similarity_matrix(inputs, dictionary, stype=self.stype)
        inputs = tlu(inputs, self.p)
        outputs = aggregation(inputs, atype=self.atype)
        return outputs


if __name__ == "__main__":
    # bow_pooling_demo()
    device = torch.device('cuda')
    d = torch.randint(0, 5, [4, 3])  # dictionary, [K,C], 4 clustering centers with 3-dim vector
    print(d)
    x = torch.randint(0, 5, [2, 3, 5])  # points position data, [B,C,N], batch_size=2, 5 points with 3-dim vector
    print(x)
    bow_pooling = Bow_Pooling(dictionary=d)
    h = bow_pooling(x)
    print(h)
    print(h.shape)
