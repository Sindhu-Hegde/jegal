import numpy as np
import os, argparse
import pandas as pd
import pickle
import ast
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="Path to the directory to load the extracted JEGAL features")
parser.add_argument('--file', type=str, required=True, help="Path to the AVS-ASD csv file")
args = parser.parse_args()


def read_data(fname):
	
	df = pd.read_csv(fname)
	print("Total files: {}".format(len(df)))

	return df


def load_feats(fname, load_content=True):
	
	with open(fname, 'rb') as f:
		feats = pickle.load(f)

	gesture_emb_temporal = feats["gesture_emb"]
	gesture_emb = torch.FloatTensor(gesture_emb_temporal.mean(axis=0)).unsqueeze(0)
	
	if load_content:
		content_emb_temporal = feats["content_emb"]
		content_emb = torch.FloatTensor(content_emb_temporal.mean(axis=0)).unsqueeze(0)
		return gesture_emb, content_emb
	
	return gesture_emb



def get_similarity_cos(query_emb, data_emb, temp=0.07):
	
	cos = nn.CosineSimilarity(dim=1)

	sim = cos(query_emb, data_emb)
	sim_scaled = sim/temp
	scores = F.softmax(sim_scaled, dim=0).cpu().detach().numpy()                
																	  
	return scores 


def evaluate_asd(df):
	
	num_correct1, total_vids1 = 0, 0
	num_correct3, total_vids3 = 0, 0
	num_correct5, total_vids5 = 0, 0

	prog_bar = tqdm(range(len(df)))
	for i in prog_bar:
		
		data = df.iloc[i]
		query_fname = data.filename
		query_gest_emb_path = os.path.join(args.path, query_fname.split("/")[0] + "__" + query_fname.split("/")[1] + ".pkl")
		if not os.path.exists(query_gest_emb_path):
			continue

		query_gest_emb, query_content_emb = load_feats(query_gest_emb_path, load_content=True)
		if query_gest_emb is None or query_content_emb is None:
			continue
		# print("Query gesture emb: {} | Query content emb: {}".format(query_gest_emb.shape, query_content_emb.shape))

		pos_idx = 0 
		all_gesture_embs =[query_gest_emb]

		all_neg_files = ast.literal_eval(data.neg_files)
		for neg_fname in all_neg_files:

			neg_gest_emb_path = os.path.join(args.path, neg_fname.split("/")[0] + "__" + neg_fname.split("/")[1] + ".pkl")
			if not os.path.exists(neg_gest_emb_path):
				continue
			neg_gest_emb = load_feats(neg_gest_emb_path, load_content=False)

			if neg_gest_emb is None:
				continue

			all_gesture_embs.append(neg_gest_emb)

			
		all_gesture_embs = torch.cat(all_gesture_embs)
		# print("CONTENT QUERY: {} | ALL GESTURE EMBs: {}".format(query_content_emb.shape, all_gesture_embs.shape))

		pred_idx_all = []
		for neg_idx in [2,4,6]:

			neg_gest_embs = all_gesture_embs[:neg_idx]
			scores = get_similarity_cos(query_content_emb, neg_gest_embs)
			max_sc_idx = np.argmax(scores)
			pred_idx_all.append(max_sc_idx)


		if pred_idx_all[0]==pos_idx:
			num_correct1 += 1
		total_vids1 += 1

		if pred_idx_all[1]==pos_idx:
			num_correct3 += 1
		total_vids3 += 1

		if pred_idx_all[2]==pos_idx:
			num_correct5 += 1
		total_vids5 += 1
		

		prog_bar.set_description('Acc-2-spk: {:.3f} | Acc-4-spk: {:.3f} | Acc-6-spk: {:.3f}'
							.format(num_correct1/total_vids1,
									num_correct3/total_vids3,
									num_correct5/total_vids5))

	print("Total videos evaluated: {}".format(total_vids5))
	print("2 spk: Correct: {} | Total: {} | Acc: {:.3f}".format(num_correct1, total_vids1, num_correct1/total_vids1))
	print("4 spk: Correct: {} | Total: {} | Acc: {:.3f}".format(num_correct3, total_vids3, num_correct3/total_vids3))
	print("6 spk: Correct: {} | Total: {} | Acc: {:.3f}".format(num_correct5, total_vids5, num_correct5/total_vids5))


	return

if __name__ == "__main__":


	df = read_data(args.file)
	evaluate_asd(df)