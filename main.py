"""Running MixHop."""
import torch
from param_parser import parameter_parser
import model
from utils import tab_printer
import numpy as np
import time
from loadData import data_reader, split_data
from createGraph import SLIC, create_graph
import train
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


args = parameter_parser()
torch.manual_seed(args.seed)
#args.epochs = -1
tab_printer(args)

# load data
def load_data():
    data1 = data_reader.USA().normal_cube1
    data2 = data_reader.USA().normal_cube2
    data_gt = data_reader.USA().truth
    return data1, data2, data_gt

data1, data2, data_gt = load_data()
class_num = np.max(data_gt)
height, width, bands = data1.shape
gt_reshape = np.reshape(data_gt, [-1])

print('data1.shape:', data1.shape)
print('data2.shape:', data2.shape)
print('data_gt.shape:', data_gt.shape)
print('gt_reshape.shape:', gt_reshape.shape)
print('class_num:', class_num)

###################################### 显示图片 ######################################
#plt.figure()
#plt.imshow(data1[:,:,[0,1,2]])
#plt.show()
###################################### 显示图片 ######################################

# Concat
T1 = torch.from_numpy(data1)
T2 = torch.from_numpy(data2)
data = torch.cat([T1, T2], dim=-1)  # 按倒数第一维cat

ITER = 1
for i in range(ITER):

    print('ITER: ', i+1)

    # superpixels
    ls = SLIC.SLIC(data, args.n_segments_init, args)
    tic0 = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。

    # print('superpixel_scale', superpixel_scale)    <Santa8938>
    S, Q1, Q2, Seg, superpixel_count = ls.get_Q_and_S_and_Segments()
    A1, A2 = ls.get_A(sigma=0.1)
    S = torch.from_numpy(S).to(args.device)
    print('S.shape:', S.shape)

    toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0
    print('LDA_SLIC_Time', LDA_SLIC_Time)

    # split datasets
    segments = torch.tensor(Seg).flatten().to(args.device)
    print('segments.shape', segments.shape)  # (58800,)
    gt = torch.LongTensor(gt_reshape)
    print('gt.shape', gt.shape)  # torch.Size([58800])

    train_index, val_index, test_index = split_data.get_indices(args, class_num, gt_reshape)
    # train_index, val_index, test_index = split_data.sampling(args, gt_reshape)

    TRAIN_SIZE = len(train_index)
    print('Train size: ', TRAIN_SIZE)  # \294
    TEST_SIZE = len(test_index)
    print('Test size: ', TEST_SIZE)  # \57918
    VAL_SIZE = len(val_index)
    print('Validation size: ', VAL_SIZE)  # \588

    # create graph
    train_samples_gt, test_samples_gt, val_samples_gt = create_graph.get_label(gt_reshape,
                                                                               train_index, val_index, test_index)

    print('train_samples_gt.shape', train_samples_gt.shape)  # (21025,)

    train_label_mask, test_label_mask, val_label_mask = create_graph.get_label_mask(train_samples_gt,
                                                                                    test_samples_gt, val_samples_gt,
                                                                                    data_gt, class_num)

    print('train_label_mask.shape', train_label_mask.shape)

    # label transfer to one-hot encode
    train_gt = np.reshape(train_samples_gt, [height, width])
    test_gt = np.reshape(test_samples_gt, [height, width])
    val_gt = np.reshape(val_samples_gt, [height, width])

    print('train_gt.shape.shape', train_gt.shape)

    train_gt_onehot = create_graph.label_to_one_hot(train_gt, class_num)
    test_gt_onehot = create_graph.label_to_one_hot(test_gt, class_num)
    val_gt_onehot = create_graph.label_to_one_hot(val_gt, class_num)

    print('train_gt_onehot.shape', train_gt_onehot.shape)

    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
    val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(args.device)

    train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
    test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
    val_gt_onehot = torch.from_numpy(val_gt_onehot.astype(np.float32)).to(args.device)

    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(args.device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)

    # model
    net = model.MixHopNetwork(T1, T2, args, bands, class_num, segments).to(args.device)

    # train
    print("training on ", args.device)
    trainer = train.Trainer(net, A1, Q1, A2, Q2, S, train_gt_onehot, val_gt_onehot, test_gt_onehot, train_samples_gt,
                            val_samples_gt, test_samples_gt, train_label_mask, val_label_mask, test_label_mask,
                            segments, data_gt, height, width, args)
    trainer.fit()