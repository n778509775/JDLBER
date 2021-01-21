#!/usr/bin/env python
# encoding: utf-8
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale, Imputer

min_var_est = 1e-8

def read_csv(filename, take_log):
	data_df = pd.read_csv(filename)
	dataset = {}
	dataset['labels'] = data_df.iloc[:,0].tolist()
	dataset['board'] = data_df.iloc[:,1].tolist()
	mz_exp = np.transpose(np.array(data_df.iloc[:,2:]))

	#remove NaN
	if np.any(np.isnan(mz_exp)):
		my_imputer = Imputer()
		mz_exp = my_imputer.fit_transform(mz_exp)

	if take_log:
		mz_exp = np.log2(mz_exp + 1)
	dataset['mz_exp'] = mz_exp
	dataset['feature'] = data_df.columns.values.tolist()[2:]
	return dataset

def pre_processing(dataset_file_list, pre_process_paras):
    # parameters
    take_log = pre_process_paras['take_log']
    standardization = pre_process_paras['standardization']
    scaling = pre_process_paras['scaling']

    dataset_list = []
    for data_file in dataset_file_list:
        dataset = read_csv(data_file, take_log)
        if standardization:
            scale(dataset['mz_exp'], axis=1, with_mean=True, with_std=True, copy=False)
        if scaling:  # scale to [0,1]
            minmax_scale(dataset['mz_exp'], feature_range=(0, 1), axis=1, copy=False)
        dataset_list.append(dataset)

    return dataset_list

def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    

    # Get the various sums of kernels that we'll use
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       
        diag_Y = torch.diag(K_YY)                       
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             
    K_XY_sums_0 = K_XY.sum(dim=0)                     

    Kt_XX_sum = Kt_XX_sums.sum()                       
    Kt_YY_sum = Kt_YY_sums.sum()                       
    K_XY_sum = K_XY_sums_0.sum()                       

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

def plot_recon_loss(loss_reconstruct_list, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(loss_reconstruct_list)), loss_reconstruct_list, "g-",linewidth=1)
    ax.legend(['loss_reconstruct'], loc="upper right")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_trans_loss(loss_transfer_list, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(loss_transfer_list)), loss_transfer_list, "r:",linewidth=1)
    ax.legend(['loss_calibration'], loc="upper right")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_clas_loss(loss_classifier_list, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(loss_classifier_list)), loss_classifier_list, "b--",linewidth=1)
    ax.legend(['loss_classification'], loc="upper right")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

