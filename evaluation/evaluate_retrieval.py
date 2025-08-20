import argparse 
import numpy as np
import os, sys, subprocess
import pandas as pd
import pickle
import ast
import torch
from torch import nn
from torch.nn import functional as F
from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="Path to the directory to load the extracted JEGAL features")
args = parser.parse_args()

def load_feats(path):
    
    files = glob("{}/*.pkl".format(path))
    print("No of files = ", len(files))

    gesture_emb, content_emb, data_rows, phrases = [],[],[],[]
    
    for fname in tqdm(files):
        with open(fname, 'rb') as f:
            feats = pickle.load(f)

        # Take the temporal average of the gesture and content embeddings to get the video-level embeddings
        gesture_emb.append(feats["gesture_emb"].mean(axis=0).squeeze())
        content_emb.append(feats["content_emb"].mean(axis=0).squeeze())
        data_rows.append(feats["info"])
        phrases.append(feats["info"]["phrase"])

    return gesture_emb, content_emb, data_rows, phrases


def get_similarity_matrix(emb1, emb2):
    
    emb1 = torch.FloatTensor(emb1)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    
    emb2 = torch.FloatTensor(emb2)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    
    sim_mat = torch.matmul(emb1, emb2.t())
    
    return sim_mat


def compute_metrics(x):
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R5'] = float(np.sum(ind < 5)) / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) / len(ind)
    metrics['R25'] = float(np.sum(ind < 25)) / len(ind)
    metrics['R50'] = float(np.sum(ind < 50)) / len(ind)
    metrics['MR'] = np.median(ind) + 1

    return metrics


def print_computed_metrics(metrics):
    r5 = metrics['R5']
    r10 = metrics['R10']
    r25 = metrics['R25']
    r50 = metrics['R50']
    mr = metrics['MR']
    print('R@5: {:.2f} - R@10: {:.2f} - R@25: {:.2f} - R@50: {:.2f} | Median R: {:.1f}'.format(r5*100, r10*100, r25*100, r50*100, mr))

def get_metrics(emb1, emb2):

    sim_mat = get_similarity_matrix(emb1, emb2)
    # print("Sim matrix: ", sim_mat.shape)

    metrics = compute_metrics(sim_mat)
    print_computed_metrics(metrics)

    return


def get_all_metrics(path, mode="c2g"):
    
    gesture_emb, content_emb = load_feats(path)    
    # print("Gesture Features: {}, Content Features: {}".format(gesture_emb.shape, content_emb.shape))
    

    if mode == "c2g":
        get_metrics(content_emb, gesture_emb)
    elif mode == "g2c":
        get_metrics(gesture_emb, content_emb)
        
    

if __name__=="__main__":

    data_path = args.path

    print("Content to Gesture Retrieval scores:")
    get_all_metrics(data_path, mode="c2g")
    print("-------------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------------")
    
    print("Gesture to Content Retrieval scores:")
    get_all_metrics(data_path, mode="g2c")

