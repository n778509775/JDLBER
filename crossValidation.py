import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse
import pandas as pd
from tkinter import _flatten

import torch
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.preprocessing import scale, minmax_scale, Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from network import Discriminator
from function import plot_clas_loss

def read_csv_faster(filename):
	data_df = pd.read_csv(filename,index_col=1)
	dataset = {}
	dataset['labels'] = data_df.iloc[:,0].tolist()
	dataset['mz_exp'] = np.transpose(np.array(data_df.iloc[:,1:]))
	dataset['feature'] = data_df.columns.values.tolist()[1:]
	return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/')
parser.add_argument('--train_file', type=str, default='3.csv')
parser.add_argument('--MALDI_MS', type=bool, default=True, help='flag of dataset, False for CyTOF, True for MALDI_MS')
parser.add_argument('--num_epochs', type=int, default=25, help='number of total iterations, 15 for CyTOF, 25 for MALDI_MS')
parser.add_argument('--batch_size', type=int, default=200, help='mini-batch size, 200 for both')
parser.add_argument('--base_lr', type=float, default=1e-4, help='learning rate for network, 1e-4 for both')
parser.add_argument('--lr_step', type=int, default=1000, help='step decay of learning rates, 1000 for both')
parser.add_argument('--l2_decay', type=float, default=5e-5, help='preventing overfitting, 5e-5 for both')
config = parser.parse_args()

train_file = config.data_folder + config.train_file
if config.MALDI_MS == True:    
    sumple_num = pd.read_csv(config.data_folder + 'sumple-num-'+config.train_file)
    sumple_num = list(_flatten(np.array(sumple_num).tolist()))
    sample_ids = len(sumple_num)

num_epochs = config.num_epochs
batch_size = config.batch_size # batch size for each cluster
base_lr = config.base_lr
lr_step = config.lr_step  # step decay of learning rates
l2_decay = config.l2_decay
dataset = read_csv_faster(train_file)
FinalData = dataset['mz_exp'].transpose()
AllLabel = dataset['labels']

num_inputs = FinalData.shape[1]
discriminator = Discriminator(num_inputs=num_inputs)

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

if config.MALDI_MS == True:   
	subjectA=[]
	subjectB=[]
	subjectC=[]
	subjectD=[]
sampleA=[]
sampleB=[]
sampleC=[]
sampleD=[]

id_num = np.arange(len(AllLabel))
count = 0
group_data = []
group_label = []
if config.MALDI_MS == True:    
    for i in range(sample_ids):
        group_id=id_num[count:count + sumple_num[i]]
        group_data.append(group_id.tolist())
        label_id=AllLabel[count:count + sumple_num[i]]
        group_label.append(np.median(label_id))
        count = count + sumple_num[i]

zzz=np.arange(len(group_data)*2).reshape((len(group_data), 2))
X=FinalData
if config.MALDI_MS == True:  
	zzz=np.arange(len(group_data)*2).reshape((len(group_data), 2))  
	y=np.array(group_label)
	group_data = np.array(group_data)
	AllLabel = np.array(AllLabel)
else:
	zzz=np.arange(len(AllLabel)*2).reshape((len(AllLabel), 2))
	y=np.array(AllLabel)

skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=random.randint(0,99))
for train_index,test_index in skf.split(zzz,y):
	if config.MALDI_MS == True:    
		flatten_train_index, flatten_test_index = np.array(_flatten(group_data[train_index].tolist())), np.array(_flatten(group_data[test_index].tolist()))
		X_train, X_test = X[flatten_train_index], X[flatten_test_index]
		y_train, y_test = AllLabel[flatten_train_index], AllLabel[flatten_test_index]
	else:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
	torch_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
	data_loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	loss_classifier_list = []
	for epoch in range(1, num_epochs + 1):
		learning_rate = base_lr * math.pow(0.9, epoch / lr_step)
		optimizer = torch.optim.Adam([{'params': discriminator.parameters()},], lr=learning_rate, weight_decay=l2_decay)

		discriminator.train()
		iter_data = iter(data_loader)
		num_iter = len(data_loader)
		total_clas_loss = 0
		num_batches = 0
		for it in range(0, num_iter):
			data, label = iter_data.next()
			if it % len(data_loader) == 0:
				iter_data = iter(data_loader)
			data = Variable(torch.FloatTensor(data))
			label = Variable(torch.LongTensor(label))
			Disc_a = discriminator(data)

			optimizer.zero_grad()
			loss_classification = torch.FloatTensor([0])
			for cls in range(len(label)):
				loss_classification += F.binary_cross_entropy(torch.squeeze(Disc_a)[cls], label[cls].float())
			loss = loss_classification
			loss.backward()
			optimizer.step()

			num_batches += 1
			total_clas_loss += loss_classification.data.item()
		avg_clas_loss = total_clas_loss / num_batches
		loss_classifier_list.append(avg_clas_loss)
	plot_clas_loss(loss_classifier_list, 'clas_loss.png')
	discriminator.eval()

	Disc_b = discriminator(torch.from_numpy(X_test).float())

	#"Subject" Level
	if config.MALDI_MS == True:    
		count=0
		predict = []
		for i in test_index:
			subject_pred=Disc_b[count:count + sumple_num[i]]
			avg_subject_pred=np.median(subject_pred.detach().numpy())
			predict.append(avg_subject_pred)
			count = count + sumple_num[i]
	      
		pred_b = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in predict]))
		test_label = torch.from_numpy(y[test_index])
		num_correct_b = 0
		num_correct_b += torch.eq(pred_b, test_label).sum().float().item()
		Acc_b = num_correct_b/len(test_label)
		TP, TN, FP, FN = matric(pred_b, test_label)
		precision = TP / (TP + FP)
		recall = TP / (TP + FN)
		f_score = 2 * precision * recall / (precision + recall)
		AUC = roc_auc_score(test_label.cpu(), pred_b.cpu())
		MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
		subjectA.append(Acc_b)
		subjectB.append(f_score)
		subjectC.append(AUC)
		subjectD.append(MCC)

	#"Sample" Level
	pred_b2 = torch.from_numpy(np.array([1 if i > 0.5 else 0 for i in Disc_b]))
	test_label2 = torch.from_numpy(y_test)
	num_correct_b2 = 0
	num_correct_b2 += torch.eq(pred_b2, test_label2).sum().float().item()
	Acc_b2 = num_correct_b2/len(test_label2)
	TP2, TN2, FP2, FN2 = matric(pred_b2, test_label2)
	precision2 = TP2 / (TP2 + FP2)
	recall2 = TP2 / (TP2 + FN2)
	f_score2 = 2 * precision2 * recall2 / (precision2 + recall2)
	AUC2 = roc_auc_score(test_label2.cpu(), pred_b2.cpu())
	MCC2 = (TP2 * TN2 - FP2 * FN2) / math.sqrt((TP2 + FP2) * (TP2 + FN2) * (TN2 + FP2) * (TN2 + FN2))
	sampleA.append(Acc_b2)
	sampleB.append(f_score2)
	sampleC.append(AUC2)
	sampleD.append(MCC2)

if config.MALDI_MS == True:    
	print("Subject Level: ", np.mean(subjectA), np.mean(subjectB), np.mean(subjectC), np.mean(subjectD))
print("Sample Level: ", np.mean(sampleA), np.mean(sampleB), np.mean(sampleC), np.mean(sampleD))

