import torch
from tqdm import trange
from layers import SparseNGCNLayer, DenseNGCNLayer, ListModule
import torch.nn.functional as F
from torch import nn
import numpy as np
import scipy.io as sio

def AmplificationModule(H1, H2, args):
    CS = F.cosine_similarity(H1, H2, dim=0)
    #print(CS.shape)  #torch.Size([288])
    S = torch.sigmoid(torch.ones(CS.shape).to(args.device) - CS)

    H1 = H1 * S
    H2 = H2 * S

    return H1, H2


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        #print('self.point_conv.shape', out.shape)
        out = self.Act1(out)
        #print('self.Act1.shape', out.shape)
        out = self.depth_conv(out)
        #print('depth_conv.shape', out.shape)
        out = self.Act2(out)
        #print('self.Act2.shape', out.shape)
        return out

class MixHopNetwork(torch.nn.Module):

    def __init__(self, T1, T2, args, feature_number, class_number, segments):
        super(MixHopNetwork, self).__init__()
        self.args = args
        self.feature_number = feature_number
        self.class_number = class_number
        self.segments = segments
        self.calculate_layer_sizes()
        self.setup_layer_structure()
        self.T1 = T1
        self.T2 = T2

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = sum(self.args.layers_1)
        self.abstract_feature_number_2 = sum(self.args.layers_2)
        self.order_1 = len(self.args.layers_1)
        #print("self.order_1", self.order_1)  #self.order_1 3
        self.order_2 = len(self.args.layers_2)

    def setup_layer_structure(self):

        self.upper_layers = [SparseNGCNLayer(self.feature_number, self.args.layers_1[i - 1], i, self.args.dropout) for i
                             in range(1, self.order_1 + 1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [DenseNGCNLayer(self.abstract_feature_number_1, self.args.layers_2[i - 1], i, self.args.dropout) for i in range(1, self.order_2 + 1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        #self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2 * 2, self.class_number)
        self.cnn = SSConv(self.abstract_feature_number_2 * 2, 32)
        self.fc = torch.nn.Linear(32, self.class_number)
        '''self.fully_connected = nn.Sequential(
            nn.Linear(self.abstract_feature_number_2 * 2, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2)
        )'''

    def forward(self, A1, Q1, A2, Q2, S):

        abstract_features_11 = torch.cat(
            [self.upper_layers[i](A1, Q1) for i in range(self.order_1)], dim=1)
        abstract_features_21 = torch.cat(
            [self.upper_layers[i](A2, Q2) for i in range(self.order_1)], dim=1)

        #print(abstract_features_11.device)

        abstract_features_11, abstract_features_21 = AmplificationModule(abstract_features_11, abstract_features_21, self.args)

        abstract_features_12 = torch.cat(
            [self.bottom_layers[i](A1, abstract_features_11) for i in range(self.order_2)],
            dim=1)
        abstract_features_22 = torch.cat(
            [self.bottom_layers[i](A2, abstract_features_21) for i in range(self.order_2)],
            dim=1)

        abstract_features = torch.cat([abstract_features_12, abstract_features_22], dim=-1)

        #print(abstract_features.shape)
        abstract_features = torch.matmul(S, abstract_features)

        x = torch.reshape(abstract_features, [self.T1.shape[0], self.T1.shape[1], abstract_features.shape[-1]])
        abstract_features = torch.unsqueeze(x.permute([2, 0, 1]), 0)
        #print(abstract_features.shape)  torch.Size([1, 272, 600, 500])
        abstract_features = self.cnn(abstract_features)
        abstract_features = torch.squeeze(abstract_features, 0).permute([1, 2, 0]).reshape([S.shape[0], -1])
        predictions = F.softmax(self.fc(abstract_features), dim=1)

        #predictions = F.softmax(self.fully_connected(abstract_features), dim=1)

        #print('predictions.shape', predictions.shape)  #torch.Size([14699, 2])

        return predictions
