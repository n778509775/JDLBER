#!/usr/bin/env python
import torch.utils.data
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from tkinter import _flatten
import pandas as pd
from function import pre_processing, plot_clas_loss

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import network as models
import math
import argparse
import pylib
from sklearn.metrics import roc_auc_score

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

# CUDA
device_id = 0 # ID of GPU to use
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.set_device(device_id)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

plt.ioff()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default='data/')
    parser.add_argument('--train_file', type=str, default='3.csv')
    parser.add_argument('--test_file', type=str, default='2.csv')
    parser.add_argument('--MALDI_MS', type=bool, default=True, help='flag of dataset, False for CyTOF, True for MALDI_MS')
    parser.add_argument('--code_save', type=str, default='code_list.pkl')
    parser.add_argument('--take_log', type=bool, default=False)
    parser.add_argument('--standardization', type=bool, default=False)
    parser.add_argument('--scaling', type=bool, default=False)
    parser.add_argument('--plots_dir', type=str, default='plots/')

    parser.add_argument('--batch_size', type=int, default=200, help='mini-batch size, 200 for both')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of total iterations, 100 for both')
    parser.add_argument('--lr_step', type=int, default=10000, help='step decay of learning rates, 10000 for both')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='learning rate for network, 1e-4 for CyTOF, 1e-3 for MALDI_MS')
    parser.add_argument('--l2_decay', type=float, default=5e-5, help='preventing overfitting, 5e-5 for both')
    parser.add_argument('--log_interval', type=int, default=10)

    config = parser.parse_args()

    data_folder = config.data_folder
    train_file = data_folder + config.train_file
    test_file = data_folder + config.test_file
    code_save_file = data_folder + config.code_save
    dataset_file_list = [train_file, test_file]
    plots_dir = config.plots_dir
    if config.MALDI_MS == True:    
        sumple_num = pd.read_csv(data_folder+'sumple-num-'+config.test_file)
        sumple_num = list(_flatten(np.array(sumple_num).tolist()))
        sample_ids = len(sumple_num)

    # read data
    pre_process_paras = {'take_log': config.take_log, 'standardization': config.standardization, 'scaling': config.scaling}
    dataset_list = pre_processing(dataset_file_list, pre_process_paras)

    # training
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    num_inputs = len(dataset_list[0]['feature'])

    # construct a DataLoader for each batch
    batch_loader_dict = {}
    for i in range(len(dataset_list)):
        gene_exp = dataset_list[i]['mz_exp'].transpose()
        labels = dataset_list[i]['labels']  

        # construct DataLoader list
        if cuda:
            torch_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(gene_exp).cuda(), torch.LongTensor(labels).cuda())
        else:
            torch_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(gene_exp), torch.LongTensor(labels))
        data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size,
                                                        shuffle=True, drop_last=True)
        batch_loader_dict[i+1] = data_loader

    # create model
    discriminator = models.Discriminator(num_inputs=num_inputs)
    #focalLoss = models.FocalLoss()
    if cuda:
        discriminator.cuda()

    # training
    loss_classifier_list = []
    for epoch in range(1, num_epochs + 1):
        log_interval = config.log_interval
        base_lr = config.base_lr
        lr_step = config.lr_step
        num_epochs = config.num_epochs
        l2_decay = config.l2_decay

        # step decay of learning rate
        learning_rate = base_lr * math.pow(0.9, epoch / lr_step)
        # regularization parameter between two losses

        if epoch % log_interval == 0:
            print('{:}, Epoch {}, learning rate {:.3E}'.format(time.asctime(time.localtime()), epoch, learning_rate))
                
        optimizer = torch.optim.Adam([
            {'params': discriminator.parameters()},
        ], lr=learning_rate, weight_decay=l2_decay)

        discriminator.train()

        iter_data_dict = {}
        for cls in batch_loader_dict:
            iter_data = iter(batch_loader_dict[cls])
            iter_data_dict[cls] = iter_data
        # use the largest dataset to define an epoch
        num_iter = 0
        for cls in batch_loader_dict:
            num_iter = max(num_iter, len(batch_loader_dict[cls]))

        total_clas_loss = 0
        num_batches = 0

        for it in range(0, num_iter):
            data_dict = {}
            label_dict = {}
            code_dict = {}
            reconstruct_dict = {}
            for cls in iter_data_dict:
                data, labels = iter_data_dict[cls].next()
                data_dict[cls] = data
                label_dict[cls] = labels
                if it % len(batch_loader_dict[cls]) == 0:
                    iter_data_dict[cls] = iter(batch_loader_dict[cls])
                data_dict[cls] = Variable(data_dict[cls])
                label_dict[cls] = Variable(label_dict[cls])
            
            Disc_a = discriminator(data_dict[1])

            optimizer.zero_grad()

            #Loss
            # classifier loss for dignosis
            loss_classification = torch.FloatTensor([0])
            if cuda:
                loss_classification = loss_classification.cuda()
            for cls in range(len(label_dict[1])):
                loss_classification += F.binary_cross_entropy(torch.squeeze(Disc_a)[cls], label_dict[1][cls].float())

            loss = loss_classification

            loss.backward()
            optimizer.step()

            # update total loss
            num_batches += 1
            total_clas_loss += loss_classification.data.item()

        avg_clas_loss = total_clas_loss / num_batches

        if epoch % log_interval == 0:
            print('Avg_classify_loss {:.3E}'.format(avg_clas_loss))            

        loss_classifier_list.append(avg_clas_loss)

    plot_clas_loss(loss_classifier_list, plots_dir+'clas_loss.png')

    # testing: extract codes
    discriminator.eval()

    test_data1 = torch.from_numpy(dataset_list[1]['mz_exp'].transpose())
    test_label1 = torch.from_numpy((np.array(dataset_list[1]['labels']))).cuda()

    Disc_b = discriminator(test_data1.float().cuda())

    #"Sample" Level
    print("-----Sample Level-----")
    pred_b = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in Disc_b])).cuda()

    #Accuracy
    num_correct_b = 0
    num_correct_b += torch.eq(pred_b, test_label1).sum().float().item()

    Acc_b = num_correct_b/len(test_label1)
    print("Accuracy is ", Acc_b)

    #F_score
    def matric(cluster, labels):
        TP, TN, FP, FN = 0, 0, 0, 0
        n = len(labels)
        for i in range(n):
            if cluster[i]:
                if labels[i]:
                    TP += 1
                else:
                    FP += 1
            elif labels[i]:
                FN += 1
            else:
                TN += 1
        return TP, TN, FP, FN

    TP, TN, FP, FN = matric(pred_b, test_label1)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)
    print("F_score is ",f_score)

    #AUC
    print("AUC is ",roc_auc_score(test_label1.cpu(), pred_b.cpu()))

    #MCC
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    print("MCC is ",MCC)

    #"Subject" Level
    if config.MALDI_MS == True:    
        print("-----Subject Level-----")
        count=0
        standard = []
        predict = []
        for i in range(sample_ids):
            subject_label1=test_label1[count:count + sumple_num[i]]
            avg_subject_label1=np.median(subject_label1.cpu().detach().numpy())
            standard.append(avg_subject_label1)
            subject_pred=Disc_b[count:count + sumple_num[i]]
            avg_subject_pred=np.median(subject_pred.cpu().detach().numpy())
            predict.append(avg_subject_pred)
            count = count + sumple_num[i]

        pred_b = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in predict])).cuda()
        standard = torch.from_numpy(np.array(standard)).cuda()

        #Accuracy
        num_correct_b = 0
        num_correct_b += torch.eq(pred_b, standard).sum().float().item()

        Acc_b = num_correct_b/sample_ids
        print("Accuracy is ", Acc_b)

        #F_score
        TP, TN, FP, FN = matric(pred_b, standard)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_score = 2 * precision * recall / (precision + recall)
        print("F_score is ",f_score)

        #AUC
        print("AUC is ",roc_auc_score(standard.cpu(), pred_b.cpu()))

        #MCC
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print("MCC is ",MCC)

