import numpy as np
import torch
import random

def get_indices(args, class_num, gt_reshape):

    train_index = []
    test_index = []
    val_index = []

    for i in range(class_num):

        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        # print("Class ",i,":", samplesCount)
        train_num = np.ceil(samplesCount * args.train_ratio).astype('int32')
        val_num = np.ceil(samplesCount * args.val_ratio).astype('int32')
        np.random.shuffle(idx)
        train_index.append(idx[:train_num])
        val_index.append(idx[train_num:train_num + val_num])
        test_index.append(idx[train_num + val_num:])

    train_index = np.concatenate(train_index, axis=0)
    val_index = np.concatenate(val_index, axis=0)
    test_index = np.concatenate(test_index, axis=0)

    train_index = torch.LongTensor(train_index)
    val_index = torch.LongTensor(val_index)
    test_index = torch.LongTensor(test_index)

    #print('train_index.shape', train_index.shape)  #torch.Size([295])

    return train_index, val_index, test_index


def sampling(args, gt):

    train_rand_idx = []
    gt_1d = gt


    idx = np.where(gt_1d < 3)[-1]
    samplesCount = len(idx)
    rand_list = [i for i in range(samplesCount)]
    rand_idx = random.sample(rand_list, np.ceil(samplesCount * args.train_ratio).astype('int32'))
    rand_real_idx_per_class = idx[rand_idx]
    train_rand_idx.append(rand_real_idx_per_class)

    train_rand_idx = np.array(train_rand_idx)
    train_index = []
    for c in range(train_rand_idx.shape[0]):
        a = train_rand_idx[c]
        for j in range(a.shape[0]):
            train_index.append(a[j])
    train_index = np.array(train_index)

    train_index = set(train_index)
    all_index = [i for i in range(len(gt_1d))]
    all_index = set(all_index)

    background_idx = np.where(gt_1d == 0)[-1]
    background_idx = set(background_idx)
    test_index = all_index - train_index - background_idx

    val_count = int(0.01 * (len(test_index) + len(train_index)))
    val_index = random.sample(test_index, val_count)
    val_index = set(val_index)
    test_index = test_index - val_index

    test_index = list(test_index)
    train_index = list(train_index)
    val_index = list(val_index)

    return train_index, val_index, test_index
