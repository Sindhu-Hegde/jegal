import argparse
import torch
import torch.nn.functional as F
import numpy as np
import ast
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


args = argparse.ArgumentParser()
args.add_argument("--path", type=str, required=True, help="Path to the JEGAL feature file")
args.add_argument("--fname", default="heatmap", help="Name of the heatmap to be saved")
args = args.parse_args()

def load_features(path):

    '''
    This function loads the features from the JEGAL feature file
    Args:
        - path (str): Path to the JEGAL feature file
    Returns:
        - gesture_emb (array): JEGAL Gesture embedding
        - content_emb (array): JEGAL Content embedding
        - word_boundaries (list): Word boundaries in the text
    '''

    with open(path, 'rb') as f:
        feats = pickle.load(f)

    return feats["gesture_emb"], feats["content_emb"], feats["info"]["word_boundaries"]

def get_attn_matrix(gesture_emb, content_emb, word_boundaries, temp=0.07):

    '''
    This function computes the attention matrix between the gesture and text embeddings
    Args:
        - gesture_emb (array): JEGAL Gesture embedding
        - content_emb (array): JEGAL Content embedding
        - word_boundaries (list/str): Word boundaries in the text
    Returns:
        - attn_mtx (array): Attention matrix
        - all_words (list): List of words in the text
    '''

    if type(word_boundaries) == str:
        word_boundaries = ast.literal_eval(word_boundaries)
    all_words = [word_boundaries[i][0] for i in range(len(word_boundaries))]

    content = torch.FloatTensor(content_emb.astype(np.float32))
    gesture = torch.FloatTensor(gesture_emb.astype(np.float32))

    attn_mat = torch.mm(gesture, content.t())
    attn_mat = attn_mat/temp
    attn_mat = F.softmax(attn_mat, dim=1)
    attn_mat = np.array(attn_mat).T
        
    return attn_mat, all_words


def plot(attn_mtx, words, fname="heatmap", thresh=0.8, alpha=0.6, cmap="jet"):

    '''
    This function plots the attention matrix
    Args:
        - attn_mtx (array): Attention matrix
        - words (list): List of words in the text
        - fname (str): Name of the heatmap to be saved
        - thresh (float): Threshold for the attention matrix
        - alpha (float): Alpha value for the attention matrix
        - cmap (str): Colormap for the attention matrix
    Returns:
        - attn_mtx (array): Attention matrix
    '''

    fig, ax = plt.subplots(1, 1, figsize=(16, 20))

    # Create RGBA heatmaps
    cmap_fn = plt.colormaps.get_cmap(cmap)
    attn_mtx_rgba = cmap_fn(attn_mtx.copy())
    
    attn_mtx_thresh = attn_mtx.copy()
    attn_mtx_thresh[attn_mtx_thresh < thresh] = 0.01
    attn_mtx_thresh_rgba = cmap_fn(attn_mtx_thresh)

    beta = 1 - alpha
    attn_mtx_thresh_rgba[..., 3] = (attn_mtx_thresh > 0).astype(float) * alpha
    attn_mtx_merged = cv2.addWeighted(attn_mtx_thresh_rgba, alpha, attn_mtx_rgba, beta, 0)

    im2 = ax.imshow(attn_mtx_merged, cmap=cmap)
    ax.set_yticks(list(range(len(words))))
    ax.set_yticklabels(words, fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.set_aspect('equal')

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.2)
    cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.show()
    fig.savefig(fname + '.png', dpi=fig.dpi)

    return attn_mtx


if __name__ == "__main__":

    gesture_emb, content_emb, word_boundaries = load_features(args.path)
    print("Gesture emb: ", gesture_emb.shape, "Content emb: ", content_emb.shape)

    attn_mtx, words = get_attn_matrix(gesture_emb, content_emb, word_boundaries)
    print("Attn mtx: ", attn_mtx.shape)
    print("Words: ", words)
    
    plot(attn_mtx, words, fname=args.fname)
    

