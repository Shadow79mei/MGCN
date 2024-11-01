import torch
import time
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def list_to_map(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 1:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 0:
            y[index] = np.array([255, 255, 255]) / 255.
    return y    #dataset:USA
'''def list_to_map(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([125, 125, 125]) / 255.
        if item == 1:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([255, 255, 255]) / 255.
    return y    #dataset:Santa'''

'''def list_to_map(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 255, 255]) / 255.
    return y'''

def list_to_colormap(x_list, gt):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if gt[index] != 0:
           if item == 1:
              y[index] = np.array([255, 255, 255]) / 255.
           if item == 0:
              y[index] = np.array([0, 0, 0]) / 255.
        else:
            y[index] = np.array([125, 125, 125]) / 255.
    return y


def generate_png(pred_test, gt_hsi, run_date, Dataset, path_result):

    gt = gt_hsi.flatten()
    x_label = pred_test
    x = np.ravel(x_label)

    #y_list = list_to_colormap(x, gt)
    y_list = list_to_map(x)
    y_gt = list_to_map(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    classification_map(y_re, gt_hsi, 300,
                       path_result + run_date + Dataset + '.png')

    classification_map(gt_re, gt_hsi, 300,
                       path_result + Dataset + '_gt.png')

    print('------Get classification maps successful-------')



class Trainer(object):

    def __init__(self, net, A1, Q1, A2, Q2, S, train_gt_onehot, val_gt_onehot, test_gt_onehot, train_samples_gt, val_samples_gt, test_samples_gt, train_label_mask, val_label_mask, test_label_mask, segments, data_gt, height, width, args):
        self.args = args
        self.model = net
        self.A1 = A1
        self.Q1 = Q1
        self.A2 = A2
        self.Q2 = Q2
        self.S = S
        self.segments = segments
        self.train_gt_onehot = train_gt_onehot
        self.val_gt_onehot = val_gt_onehot
        self.test_gt_onehot = test_gt_onehot
        self.train_samples_gt = train_samples_gt
        self.val_samples_gt = val_samples_gt
        self.test_samples_gt = test_samples_gt
        self.train_label_mask = train_label_mask
        self.val_label_mask = val_label_mask
        self.test_label_mask = test_label_mask
        self.data_gt = data_gt
        self.height = height
        self.width = width

    def fit(self):
        """
        Fitting a neural network with early stopping.
        """
        print("\n\n==================== train ====================\n")
        epochs = self.args.epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()
        zeros = torch.zeros([self.height * self.width]).to(self.args.device).float()
        no_improvement = 0
        accuracy = 0
        best_loss = 9999
        self.model.train()
        tic1 = time.time()

        train_loss_list = []
        valida_loss_list = []
        train_acc_list = []
        valida_acc_list = []
        Epoch = 0

        for i in range(epochs + 1):
            self.optimizer.zero_grad() # zero the gradient buffers
            output = self.model(self.A1, self.Q1, self.A2, self.Q2, self.S)
            # print('output.shape', output.shape)
            loss = self.compute_loss(output, self.train_samples_gt, self.train_gt_onehot, self.train_label_mask, self.segments)
            loss.backward(retain_graph=False)
            self.optimizer.step()  # Does the update

            trainOA = self.evalute(output, self.train_samples_gt, self.train_gt_onehot, zeros, self.segments)

            if i % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    output = self.model(self.A1, self.Q1, self.A2, self.Q2, self.S)
                    valloss = self.compute_loss(output, self.val_samples_gt, self.val_gt_onehot, self.val_label_mask, self.segments)
                    valOA = self.evalute(output, self.val_samples_gt, self.val_gt_onehot, zeros, self.segments)
                    print("{}\ttrain loss={:.4f}\t train OA={:.4f} val loss={:.4f}\t val OA={:.4f}".format(str(i + 1), loss, trainOA, valloss, valOA))

                    if valOA < accuracy:
                        no_improvement = no_improvement + 1
                        if no_improvement == self.args.early_stopping:
                            Epoch = i
                            print("Early stopping")
                            break
                    else:
                        no_improvement = 0
                        accuracy = valOA
                        torch.save(self.model.state_dict(), self.args.path_weight + r"model.pt")
                        print("Save model")

                    # 绘图部分
                    train_loss_list.append(loss.cpu())
                    train_acc_list.append(trainOA.cpu())
                    valida_loss_list.append(valloss.cpu())
                    valida_acc_list.append(valOA.cpu())

            # scheduler.step(valloss)
            torch.cuda.empty_cache()
            self.model.train()

            # if i % 10 == 0:
            print("{}\ttrain loss={:.4f}\t train OA={:.4f}".format(str(i + 1), loss, trainOA))
        
        toc1 = time.time()

        print("\n\n====================training done. starting evaluation...========================\n")
        # test
        torch.cuda.empty_cache()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(self.args.path_weight + r"model.pt"))
            self.model.eval()
            tic2 = time.time()
            output = self.model(self.A1, self.Q1, self.A2, self.Q2, self.S)
            toc2 = time.time()
            testloss = self.compute_loss(output, self.test_samples_gt, self.test_gt_onehot, self.test_label_mask, self.segments)
            testOA = self.evalute(output, self.test_samples_gt, self.test_gt_onehot, zeros, self.segments)
            print("\ttest loss={:.4f}\t test OA={:.4f}".format(testloss, testOA))

        torch.cuda.empty_cache()

        training_time = toc1 - tic1
        testing_time = toc2 - tic2
        training_time, testing_time

        if self.args.saveresults:
             self.classification_report(output, training_time, testing_time)
             
        # 绘图部分
        plt.figure(figsize=(8, 8.5))
        train_accuracy = plt.subplot(221)
        train_accuracy.set_title('train_accuracy')
        plt.plot(np.linspace(1, Epoch, len(train_acc_list)), train_acc_list, color='green')
        plt.xlabel('epoch')
        plt.ylabel('train_accuracy')

        test_accuracy = plt.subplot(222)
        test_accuracy.set_title('valida_accuracy')
        plt.plot(np.linspace(1, Epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
        plt.xlabel('epoch')
        plt.ylabel('test_accuracy')

        loss_sum = plt.subplot(223)
        loss_sum.set_title('train_loss')
        plt.plot(np.linspace(1, Epoch, len(train_loss_list)), train_loss_list, color='red')
        plt.xlabel('epoch')
        plt.ylabel('train loss')

        test_loss = plt.subplot(224)
        test_loss.set_title('valida_loss')
        plt.plot(np.linspace(1, Epoch, len(valida_loss_list)), valida_loss_list, color='gold')
        plt.xlabel('epoch')
        plt.ylabel('valida loss')

        #plt.show()
        plt.savefig(self.args.path_result + self.args.dataset_name + '_loss.png')


    def compute_loss(self, output, train_samples_gt, train_samples_gt_onehot, train_label_mask, segments):
        real_labels = train_samples_gt_onehot

        '''new_output = torch.zeros((segments.shape[0], output.shape[-1]), dtype=torch.float).to(segments.device)

        for idx, c in enumerate(output):
            new_output[segments == idx] = c'''

        available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()  # 有效标签的个数

        we = -torch.mul(real_labels, torch.log(output + 1e-12))
        we = torch.mul(we, train_label_mask)
        pool_cross_entropy = torch.sum(we) / available_label_count
        return pool_cross_entropy

    def evalute(self, network_output, train_samples_gt, train_samples_gt_onehot, zeros, segments):

        with torch.no_grad():
            available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
            available_label_count = available_label_idx.sum()  # 有效标签的个数

            output = torch.argmax(network_output, 1)
            '''new_output = torch.zeros((segments.shape[0]), dtype=torch.long).to(segments.device)

            # print('output:', output)

            for idx, c in enumerate(output):
                new_output[segments == idx] = c'''

            correct_prediction = torch.where(output == torch.argmax(train_samples_gt_onehot, 1),
                                             available_label_idx, zeros).sum()
            OA = correct_prediction.cpu() / available_label_count
            return OA

    def classification_report(self, output, training_time, testing_time):
        test_label_mask_cpu = self.test_label_mask.cpu().numpy()[:, 0].astype('bool')
        test_samples_gt_cpu = self.test_samples_gt.cpu().numpy().astype('int64')
        predict = torch.argmax(output, 1).cpu().numpy()

        '''new_predict = torch.zeros((self.segments.shape[0]), dtype=torch.long).to(self.args.device)
        for idx, c in enumerate(predict):
            new_predict[self.segments == idx] = c
        new_predict = new_predict.cpu().numpy()'''

        classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu],
                                               predict[test_label_mask_cpu] + 1, digits=4)
        kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1)

        print(classification, kappa)

        # store results
        print("save results")
        run_date = time.strftime('%Y%m%d-%H%M-', time.localtime(time.time()))
        f = open(self.args.path_result + run_date + self.args.dataset_name + '.txt', 'a+')
        str_results = '\n ======================' \
                        + '\nrun data = ' + run_date \
                        + "\nlearning rate = " + str(self.args.learning_rate) \
                        + "\nepochs = " + str(self.args.epochs) \
                        + "\ntrain ratio = " + str(self.args.train_ratio) \
                        + "\nval ratio = " + str(self.args.val_ratio) \
                        + "\nlayers_1 = " + str(self.args.layers_1) \
                        + "\nlayers_2 = " + str(self.args.layers_2) \
                        + '\ntrain time = ' + str(training_time) \
                        + '\ntest time = ' + str(testing_time) \
                        + '\n' + classification \
                        + "kappa = " + str(kappa) \
                        + '\n'
        f.write(str_results)
        f.close()

        # 保存图像
        generate_png(predict, self.data_gt-1, run_date, self.args.dataset_name, self.args.path_result)
        #generate_png(predict, self.data_gt, run_date, self.args.dataset_name, self.args.path_result)  #dataset:Santa


