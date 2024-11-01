import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing
import utils

# 对labels做后处理，防止出现label不连续现象
def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

class SLIC(object):
    def __init__(self, HSI, n_segments, args):

        self.n_segments = n_segments  # 分割数
        height, width, bands = HSI.shape
        #print('HSI.shape: ', HSI.shape)
        data = np.reshape(np.array(HSI), [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.args = args

    def get_Q_and_S_and_Segments(self):
        img = self.data
        (h, w, d) = img.shape
        chanels = int(d / 2)
        segments = slic(img, n_segments=self.n_segments, start_label=0, compactness=0.1, max_num_iter=20)
        #Farm:start_label=0, compactness=10, max_num_iter=10
        #USA:start_label=0, compactness=0.1, max_num_iter=20
        #Santa:start_label=0, compactness=0.1, max_num_iter=20
        #Bay:start_label=0, compactness=0.1, max_num_iter=20

        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        print('segments.shape', segments.shape)  #segments.shape (984, 740)
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        #print(segments)
        #print('type(segments)', type(segments))
        print("superpixel_count", superpixel_count)

        ###################################### 显示超像素图片 ######################################
        #print(img.shape)
        out = mark_boundaries(img[:, :, [0, 1, 2]], segments)
        plt.figure()
        plt.imshow(out)
        plt.show()
        ###################################### 显示超像素图片 ######################################

        image = torch.from_numpy(img)
        T1, T2 = torch.split(image, chanels, dim=2)
        T1 = np.array(T1)
        T2 = np.array(T2)
        #print('T1.shape', T1.shape)
        #print('T2.shape', T2.shape)

        segments = np.reshape(segments, [-1])

        Q1 = np.zeros([superpixel_count, chanels], dtype=np.float32)
        x1 = np.reshape(T1, [-1, chanels])
        #print('x1.shape', x1.shape)
        Q2 = np.zeros([superpixel_count, chanels], dtype=np.float32)
        x2 = np.reshape(T2, [-1, chanels])

        S = np.zeros([h * w, superpixel_count], dtype=np.float32)

        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels1 = x1[idx]
            pixels2 = x2[idx]
            superpixel1 = np.sum(pixels1, 0) / count
            superpixel2 = np.sum(pixels2, 0) / count
            Q1[i] = superpixel1
            Q2[i] = superpixel2
            S[idx, i] = 1

        self.Q1 = Q1
        self.Q2 = Q2
        self.S = S

        print('Q1.shape: ', Q1.shape)
        print('Q2.shape: ', Q2.shape)

        Q1 = utils.feature_reader(Q1, self.args)
        Q2 = utils.feature_reader(Q2, self.args)


        return S, Q1, Q2, self.segments, self.superpixel_count

    def get_A(self, sigma: float):
        A1 = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        A2 = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)

        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                '''print(sub)[[2656 2657].
 [2656 2657]]'''
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:

                    idx1 = sub_max
                    idx2 = sub_min
                    if A1[idx1, idx2] != 0:
                        continue

                    '''print(idx1) 3707'''
                    pix1 = self.Q1[idx1]
                    '''print(pix1.shape)  (224,)'''
                    pix2 = self.Q1[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A1[idx1, idx2] = A1[idx2, idx1] = diss

        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A2[idx1, idx2] != 0:
                        continue

                    pix1 = self.Q2[idx1]
                    pix2 = self.Q2[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A2[idx1, idx2] = A2[idx2, idx1] = diss

        print('A1.shape: ', A1.shape)
        print('A2.shape: ', A2.shape)

        A1 = utils.create_propagator_matrix(A1, self.args)
        A2 = utils.create_propagator_matrix(A2, self.args)

        #print(A1)
        #print(A2)

        return A1, A2






