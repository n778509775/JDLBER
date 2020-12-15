#!/usr/bin/env python
import torch.utils.data
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from tkinter import _flatten
from sklearn import decomposition
import pandas as pd
from pre_processing import pre_processing
from auxiliary import *

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import network as models
from mmd import mix_rbf_mmd2
import math
import time
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

# sigma for MMD
base = 1.0
sigma_list = [1, 2, 4, 8, 16]
sigma_list = [sigma / base for sigma in sigma_list]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', type=str, default='data/')
    parser.add_argument('-l', '--dataset_file_list', nargs='+', help='<Required> Set flag', required=True, type=str)
    parser.add_argument('--sumple_count', type=str, default='2.csv')
    parser.add_argument('--code_save', type=str, default='code_list.pkl')
    parser.add_argument('--take_log', type=bool, default=False)
    parser.add_argument('--standardization', type=bool, default=False)
    parser.add_argument('--scaling', type=bool, default=False)
    parser.add_argument('--plots_dir', type=str, default='plots/')

    parser.add_argument('--batch_size', type=int, default=200, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of total iterations for training')
    parser.add_argument('--lr_step', type=int, default=1000, help='step decay of learning rates')
    parser.add_argument('--base_lr', type=float, default=1e-5, help='learning rate for network')
    parser.add_argument('--l2_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=int, default=1, help='regularization between classification and others')
    parser.add_argument('--log_interval', type=int, default=1)

    config = parser.parse_args()

    data_folder = config.data_folder
    code_save_file = data_folder + config.code_save
    dataset_file_list = [data_folder+f for f in config.dataset_file_list]
    data_num = len(dataset_file_list)    
    plots_dir = config.plots_dir
    sumple_count = config.sumple_count    

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
    encoder = models.Encoder(num_inputs=num_inputs)
    decoder_a = models.Decoder_a(num_inputs=num_inputs)
    decoder_b = models.Decoder_b(num_inputs=num_inputs)
    decoder_c = models.Decoder_c(num_inputs=num_inputs)
    discriminator = models.Discriminator(num_inputs=num_inputs)

    if cuda:
        encoder.cuda()
        decoder_a.cuda()
        decoder_b.cuda()
        decoder_c.cuda()
        discriminator.cuda()

    # training
    loss_total_list = []  # list of total loss
    loss_reconstruct_list = []
    loss_transfer_list = []
    loss_classifier_list = []
    for epoch in range(1, num_epochs + 1):
        log_interval = config.log_interval
        base_lr = config.base_lr
        lr_step = config.lr_step
        num_epochs = config.num_epochs
        l2_decay = config.l2_decay
        gamma = config.gamma

        # step decay of learning rate
        learning_rate = base_lr * math.pow(0.9, epoch / lr_step)
        # regularization parameter between two losses
        gamma_rate = 2 / (1 + math.exp(-10 * (epoch) / num_epochs)) - 1
        gamma = gamma_rate * gamma

        if epoch % log_interval == 0:
            print('{:}, Epoch {}, learning rate {:.3E}, gamma {:.3E}'.format(time.asctime(time.localtime()), epoch, learning_rate, gamma))
                
        optimizer = torch.optim.Adam([
            {'params': encoder.parameters()},
            {'params': decoder_a.parameters()},
            {'params': decoder_b.parameters()},
            {'params': decoder_c.parameters()},
            {'params': discriminator.parameters()},
        ], lr=learning_rate, weight_decay=l2_decay)

        encoder.train()
        decoder_a.train()
        decoder_b.train()
        decoder_c.train()
        discriminator.train()

        iter_data_dict = {}
        for cls in batch_loader_dict:
            iter_data = iter(batch_loader_dict[cls])
            iter_data_dict[cls] = iter_data

        # use the largest dataset to define an epoch
        num_iter = 0
        for cls in batch_loader_dict:
            num_iter = max(num_iter, len(batch_loader_dict[cls]))

        total_loss = 0
        total_reco_loss = 0
        total_tran_loss = 0
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

            #for cls in data_dict:                
            c_a = encoder(data_dict[1])
            c_b = encoder(data_dict[2])
            if data_num == 3:
                c_c = encoder(data_dict[3])  
            rec_a = decoder_a(c_a)
            rec_b = decoder_b(c_b)
            if data_num == 3:
                rec_c = decoder_c(c_c)
            Disc_a = discriminator(c_a)

            code_dict[1] = c_a
            code_dict[2] = c_b
            if data_num == 3:
                code_dict[3] = c_c
            reconstruct_dict[1] = rec_a
            reconstruct_dict[2] = rec_b
            if data_num == 3:
                reconstruct_dict[3] = rec_c

            optimizer.zero_grad()

            loss_classification = torch.FloatTensor([0])
            if cuda:
                loss_classification = loss_classification.cuda()
            for cls in range(len(label_dict[1])):
                loss_classification += F.binary_cross_entropy(torch.squeeze(Disc_a)[cls], label_dict[1][cls].float())

            loss_transfer = torch.FloatTensor([0])
            if cuda:
                loss_transfer = loss_transfer.cuda()
            if data_num == 2:
                mmd2_D = mix_rbf_mmd2(code_dict[1], code_dict[2], sigma_list)
            elif data_num == 3:
                mmd2_D = mix_rbf_mmd2(code_dict[1], code_dict[2], sigma_list) + mix_rbf_mmd2(code_dict[1], code_dict[3], sigma_list) + mix_rbf_mmd2(code_dict[2], code_dict[3], sigma_list)
            loss_transfer += mmd2_D

            loss_reconstruct = torch.FloatTensor([0])
            if cuda:
                loss_reconstruct = loss_reconstruct.cuda()
            for cls in data_dict:
                loss_reconstruct += F.mse_loss(reconstruct_dict[cls], data_dict[cls])

            loss = 0.001 * loss_reconstruct + 1 * loss_transfer + 0.001 * loss_classification

            loss.backward()
            optimizer.step()

            # update total loss
            num_batches += 1
            total_loss += loss.data.item()
            total_reco_loss += loss_reconstruct.data.item()
            total_tran_loss += loss_transfer.data.item()
            total_clas_loss += loss_classification.data.item()

        avg_total_loss = total_loss / num_batches
        avg_reco_loss = total_reco_loss / num_batches
        avg_tran_loss = total_tran_loss / num_batches
        avg_clas_loss = total_clas_loss / num_batches

        if epoch % log_interval == 0:
            print('Avg_loss {:.3E}\t Avg_reconstruct_loss {:.3E}\t Avg_transfer_loss {:.3E}\t Avg_classify_loss {:.3E}'.format(avg_total_loss, avg_reco_loss, avg_tran_loss, avg_clas_loss))            

        loss_total_list.append(avg_total_loss)
        loss_reconstruct_list.append(avg_reco_loss)
        loss_transfer_list.append(avg_tran_loss)
        loss_classifier_list.append(avg_clas_loss)

    plot_recon_loss(loss_reconstruct_list, plots_dir+'recon_loss.png')
    #plot_trans_loss(loss_transfer_list, plots_dir+'trans_loss.png')
    plot_clas_loss(loss_classifier_list, plots_dir+'clas_loss.png')

    # testing
    encoder.eval()
    decoder_a.eval()
    decoder_b.eval()
    decoder_c.eval()
    discriminator.eval()

    sumple_num = pd.read_csv(data_folder+"sumple-num-"+sumple_count)
    sumple_num = list(_flatten(np.array(sumple_num).tolist()))
    sample_ids = len(sumple_num)

    if data_num == 2:
        test_data1 = torch.from_numpy(dataset_list[1]['mz_exp'].transpose())
        test_label1 = torch.from_numpy((np.array(dataset_list[1]['labels']))).cuda()
        c_b1 = encoder(test_data1.float().cuda())
        Disc_b = discriminator(c_b1)
        #"Sample" Level
        print("-----Sample Level-----")
        pred1 = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in Disc_b])).cuda()

        #Accuracy
        num_correct1 = 0
        num_correct1 += torch.eq(pred1, test_label1).sum().float().item()
        Acc1 = num_correct1/len(test_label1)
        print(Acc1)
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

        TP, TN, FP, FN = matric(pred1, test_label1)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_score = 2 * precision * recall / (precision + recall)
        print("F_score is ",f_score)

        #AUC
        print("AUC is ",roc_auc_score(test_label1.cpu(), pred1.cpu()))

        #MCC
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print("MCC is ",MCC)

        #"Subject" Level
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

        pred1 = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in predict])).cuda()
        standard = torch.from_numpy(np.array(standard)).cuda()

        #Accuracy
        num_correct1 = 0
        num_correct1 += torch.eq(pred1, standard).sum().float().item()
        Acc1 = num_correct1/sample_ids
        print(Acc1)
        #F_score
        TP, TN, FP, FN = matric(pred1, standard)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f_score = 2 * precision * recall / (precision + recall)
        print("F_score is ",f_score)

        #AUC
        print("AUC is ",roc_auc_score(standard.cpu(), pred1.cpu()))

        #MCC
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print("MCC is ",MCC)

    elif data_num == 3:
        test_data = torch.from_numpy(np.hstack((dataset_list[1]['mz_exp'],dataset_list[2]['mz_exp'])).transpose())
        test_label = torch.from_numpy((np.array(dataset_list[1]['labels'] + dataset_list[2]['labels']))).cuda()
        c_bc1 = encoder(test_data.float().cuda())
        Disc_bc = discriminator(c_bc1)

        pred = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in Disc_bc])).cuda()
        num_correct = 0
        num_correct += torch.eq(pred, test_label).sum().float().item()
        Acc = num_correct/len(test_label)
        print(Acc)

