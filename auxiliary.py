# !/usr/bin/env python
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt

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

