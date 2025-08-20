import argparse
import os, sys
import pandas as pd
import pickle
from tqdm import tqdm
import ast
from torch.utils import data as data_utils

sys.path.append("../")
from models.jegal import *
from dataset import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='Code to extract and save JEGAL features')
parser.add_argument('--file_path', required=True, help='Path to the csv file to extract features from')
parser.add_argument('--checkpoint_path', required=True, help='Path to the trained JEGAL checkpoint')
parser.add_argument('--res_dir', required=True, help='Path to the directory to save the extracted features')
parser.add_argument('--video_dir', required=True, help='Path to the directory to load the videos')
parser.add_argument('--feature_dir', required=True, help='Path to the directory to load the extracted GestSync visual features')
parser.add_argument('--modalities', required=False, default="vta", choices=['vta', 'vt', 'va', 'ta', 'v', 't', 'a'], help=("Modalities to use: vta, vt, va, ta, v, t, a"))
args = parser.parse_args()


def read_data(fname):

	df = pd.read_csv(fname)
	print("Total files: {}".format(len(df)))

	return df

def load_model(checkpoint_path, model):

	use_cuda = torch.cuda.is_available()
	
	if use_cuda:
		checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	
	s = checkpoint["state_dict"]
	new_s = {}

	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	print("Loaded checkpoint from: {}".format(checkpoint_path))

	model.eval()

	return model


def extract_embs(data_loader, model, res_dir, modalities="vta"):

	if not os.path.exists(res_dir):
		os.makedirs(res_dir)
		
	save_count = 0
	prog_bar = tqdm(data_loader)

	for batch_sample in prog_bar:
		
		if batch_sample == 0:
			print("Batch error: ", batch_sample)
			continue

		visual_feats = batch_sample["visual_feats"].cuda()			
		visual_mask = batch_sample["visual_mask"].cuda()
		text = batch_sample["text"]
		audio = batch_sample["audio"].cuda()			
		audio_mask = batch_sample["audio_mask"].cuda()
		wb = batch_sample["word_boundaries"]
		file = batch_sample["file"]
		info = batch_sample["info"]

		word_boundaries = []
		if modalities != "v":
			for i in range(len(visual_feats)):
				word_boundaries.append(ast.literal_eval(wb[i]))

		if visual_feats is None:
			continue

		gesture_emb, content_emb = None, None
		with torch.cuda.amp.autocast():
			with torch.no_grad():

				if "t" not in modalities:
					text = None
				if "a" not in modalities:
					audio = None
					audio_mask = None
				if "v" not in modalities:
					visual_feats = None
					visual_mask = None


				# Obtain the gesture and content embeddings
				gesture_emb, content_emb = model.forward_inference(visual_feats=visual_feats, visual_mask=visual_mask, text=text, audio=audio, audio_mask=audio_mask, word_boundaries=word_boundaries)

				torch.cuda.empty_cache()


		for i in range(len(gesture_emb)):

			# Normalize the embeddings
			if gesture_emb is not None:
				gesture_emb_norm = F.normalize(gesture_emb[i], p=2, dim=-1) 
				gesture_emb_norm = gesture_emb_norm.detach().cpu().numpy()
				
			if content_emb is not None:
				content_emb_norm = F.normalize(content_emb[i], p=2, dim=-1)
				content_emb_norm = content_emb_norm.detach().cpu().numpy()

			# Save the embeddings
			feat_dict = {"gesture_emb":gesture_emb_norm, "content_emb":content_emb_norm, "info":info[i]}
			output_fname = os.path.join(res_dir, file[i].split("/")[0] + "__" + file[i].split("/")[1] + ".pkl")
			with open(output_fname, 'wb') as f:
				pickle.dump(feat_dict, f)
			save_count += 1 

		prog_bar.set_description("Saved {} files".format(save_count))


	print("Saved JEGAL features at: ", res_dir)


if __name__ == "__main__":

	model = JEGAL().cuda()
	model = load_model(args.checkpoint_path, model)


	df_test = read_data(args.file_path)	
	print("Total test files being used: ", len(df_test))
	
	test_dataset = DataGenerator_Test(df=df_test, video_dir=args.video_dir, feature_dir=args.feature_dir, modalities=args.modalities)
	test_data_loader = data_utils.DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=lambda x: collate_data(x))
	print("Total test batch: ", len(test_data_loader))

	if not os.path.exists(args.res_dir):
		os.makedirs(args.res_dir)
	
	extract_embs(test_data_loader, model, os.path.join(args.res_dir, args.modalities), modalities=args.modalities)
	