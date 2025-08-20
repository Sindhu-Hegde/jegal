import argparse
import numpy as np
import pickle
import ast
import torch
from torch.nn import functional as F

from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="Path to the directory to load the extracted JEGAL features")
parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for spotting")
parser.add_argument('--frame_threshold', type=int, default=9, help="Frame threshold for spotting")
args = parser.parse_args()


def load_feats(path):
    
    files = glob("{}/*.pkl".format(path))
    print("No of files = ", len(files))

    gesture_emb, content_emb, data_rows, phrases, word_boundaries = [],[],[],[],[]
    
    for fname in tqdm(files):
        with open(fname, 'rb') as f:
            feats = pickle.load(f)

        gesture_emb.append(feats["gesture_emb"])
        content_emb.append(feats["content_emb"])

        data_rows.append(feats["info"])
        phrases.append(feats["info"]["phrase"])
        word_boundaries.append(feats["info"]["word_boundaries"])

    return gesture_emb, content_emb, data_rows, phrases, word_boundaries
    

def get_attn_matrix(idx, gesture_emb, content_emb, word_boundaries, temp=0.07):

    gesture = gesture_emb[idx]
    content = content_emb[idx]
    wb = ast.literal_eval(word_boundaries[idx])
    all_words = [wb[i][0] for i in range(len(wb))]

    gesture = torch.FloatTensor(gesture.astype(np.float32))
    content = torch.FloatTensor(content.astype(np.float32))

    gesture = F.normalize(gesture, p=2, dim=-1) 
    content = F.normalize(content, p=2, dim=-1) 

    attn_mat = torch.mm(gesture, content.t())
    attn_mat = attn_mat/temp
    attn_mat = F.softmax(attn_mat, dim=1)
    attn_mat = np.array(attn_mat).T
        
    return attn_mat, all_words

def get_spotting_acc(data_rows, gesture_emb, content_emb, word_boundaries, thresh=0.5, frame_thresh=9):
    
    correct, total = 0, 0
    prog_bar = tqdm(range(len(gesture_emb)))
    for idx in prog_bar:
        
        attn_mtx, words = get_attn_matrix(idx, gesture_emb, content_emb, word_boundaries)
        
        target_word_boundary = ast.literal_eval(data_rows[idx].target_word_boundary)
        all_word_boundaries = ast.literal_eval(word_boundaries[idx])
        query_word, start_word, end_word = target_word_boundary[0], target_word_boundary[1], target_word_boundary[2]
        word_idx = all_word_boundaries.index(target_word_boundary)

        pred_idx = np.argmax(attn_mtx[word_idx])
        pred_score = attn_mtx[word_idx][pred_idx]
        
        start_word = start_word-frame_thresh
        if start_word<0:
            start_word = 0
        end_word = end_word+frame_thresh
        
        if pred_idx>=start_word and pred_idx<=end_word: 
            if pred_score>=thresh:
                correct+=1

        total+=1
        prog_bar.set_description("Accuracy: {} || Correct: {}, Total: {}".format((correct/total)*100, correct, total))
    
    accuracy = (correct/total)*100
    print("Word Spotting Accuracy: {}".format(accuracy))

    return accuracy
    
if __name__ == "__main__":

    # Load the features
    gesture_emb, content_emb, data_rows, phrases, word_boundaries = load_feats(path=args.path)

    # Evaluate the spotting accuracy
    accuracy = get_spotting_acc(data_rows, gesture_emb, content_emb, word_boundaries, thresh=args.threshold, frame_thresh=args.frame_threshold)
    
