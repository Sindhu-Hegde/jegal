import argparse
import numpy as np
import os, sys
import pandas as pd
import pickle
import cv2
import torch
import math

from glob import glob
from tqdm import tqdm
import math

from decord import VideoReader
from decord import cpu, gpu


sys.path.append("../")
from models.gestsync import *

from einops.layers.torch import Rearrange
from einops import rearrange

import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict

parser = argparse.ArgumentParser(description='Code to extract GestSync features')
parser.add_argument('--checkpoint_path_gestsync', required=True, help='Path of the trained GestSync model')
parser.add_argument('--file', required=True, help='CSV file containing the list of videos')
parser.add_argument('--video_dir', required=True, help='Folder path containing pre-processed videos')
parser.add_argument('--result_dir', required=True, help='Folder path to save the extracted GestSync features')

parser.add_argument('--batch_size', required=False, type=int, default=48, help='Batch size')
parser.add_argument('--rank', required=False,  type=int, default=0)
parser.add_argument('--nshard', required=False, type=int, default=1)

args = parser.parse_args()

torch.manual_seed(0)

# Initialize parameters
num_frames=25
height=270
width=480
fps=25
err=0

# Initialize the mediapipe holistic keypoint detection model
mp_holistic = mp.solutions.holistic

use_cuda = torch.cuda.is_available()
print("Use CUDA: ", use_cuda)

def get_filelist(args):

	'''
	This function returns the list of video files to process
	Args:
		- args (argparse): The arguments object containing the file and data path
	Returns:
		- filelist (list): The list of video files to process
	'''
	
	df = pd.read_csv(args.file)

	filelist = []
	for i in range(len(df)):
		filelist.append(os.path.join(args.video_dir, df.iloc[i]["filename"]+".avi"))

	
	return filelist

def load_checkpoint(path, model):
	'''
	This function loads the trained model from the checkpoint

	Args:
		- path (string) : Path of the checkpoint file
		- model (object) : Model object
	Returns:
		- model (object) : Model object with the weights loaded from the checkpoint
	'''	

	# Load the checkpoint
	if use_cuda:
		checkpoint = torch.load(path)
	else:
		checkpoint = torch.load(path, map_location="cpu")
	
	s = checkpoint["state_dict"]
	new_s = {}
	
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	if use_cuda:
		model.cuda()
		print("Model moved to cuda")


	print("Loaded checkpoint from: {}".format(path))

	return model.eval()

def load_model(checkpoint_path, model):

	'''
	This function loads the trained model from the checkpoint
	Args:
		- checkpoint_path (str): Path of the checkpoint file
		- model (object): Model object
	Returns:
		- model (object): Model object with the weights loaded from the checkpoint
	'''

	if use_cuda:
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	
	s = checkpoint["state_dict"]
	new_s = {}

	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	print("Loaded checkpoint from: {}".format(checkpoint_path))

	return model.eval()

def load_video_frames(video_file):

	'''
	This function extracts the frames from the video

	Args:
		- video_file (string) : Path of the video file
	Returns:
		- frames (list) : List of frames extracted from the video
	'''

	# Read the video
	try:
		vr = VideoReader(video_file, ctx=cpu(0))
	except:
		print("Oops! Could not load the input video file")
		return None

	# Extract the frames
	frames = []
	for k in range(len(vr)):
		img = vr[k].asnumpy()
		frames.append(img)

	return frames


def get_keypoints(frames):

	'''
	This function extracts the keypoints from the frames using MediaPipe Holistic pipeline

	Args:
		- frames (list) : List of frames extracted from the video
	Returns:
		- kp_dict (dict) : Dictionary containing the keypoints and the resolution of the frames
	'''

	resolution = frames[0].shape
	all_frame_kps = []

	with mp_holistic.Holistic(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	) as holistic:
		for frame in frames:

			results = holistic.process(frame)

			pose, left_hand, right_hand, face = None, None, None, None
			if results.pose_landmarks is not None:
				pose = protobuf_to_dict(results.pose_landmarks)['landmark']
			if results.left_hand_landmarks is not None:
				left_hand = protobuf_to_dict(results.left_hand_landmarks)['landmark']
			if results.right_hand_landmarks is not None:
				right_hand = protobuf_to_dict(results.right_hand_landmarks)['landmark']
			if results.face_landmarks is not None:
				face = protobuf_to_dict(results.face_landmarks)['landmark']

			frame_dict = {"pose":pose, "left_hand":left_hand, "right_hand":right_hand, "face":face}

			all_frame_kps.append(frame_dict)

	kp_dict = {"kps":all_frame_kps, "resolution":resolution}

	return kp_dict

def load_rgb_masked_frames(input_frames, kp_dict, width=480, height=270):

	'''
	This function masks the faces using the keypoints extracted from the frames

	Args:
		- input_frames (list) : List of frames extracted from the video
		- kp_dict (dict) : Dictionary containing the keypoints and the resolution of the frames
		- width (int) : Width of the frames
		- height (int) : Height of the frames
	Returns:
		- input_frames_masked (array) : Face-masked frames which act as input to the model
	'''

	# Face indices to extract the face-coordinates needed for masking
	face_oval_idx = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 
					176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]

	
	input_keypoints, resolution = kp_dict['kps'], kp_dict['resolution']

	input_frames_masked = []
	for i, frame_kp_dict in enumerate(input_keypoints):

		img = input_frames[i]
		face = frame_kp_dict["face"]

		if face is None:
			img = cv2.resize(img, (width, height))
			masked_img = cv2.rectangle(img, (0,0), (width,110), (0,0,0), -1)
		else:
			face_kps = []
			for idx in range(len(face)):
				if idx in face_oval_idx:
					x, y = int(face[idx]["x"]*resolution[1]), int(face[idx]["y"]*resolution[0])
					face_kps.append((x,y))

			face_kps = np.array(face_kps)
			x1, y1 = min(face_kps[:,0]), min(face_kps[:,1])
			x2, y2 = max(face_kps[:,0]), max(face_kps[:,1])
			masked_img = cv2.rectangle(img, (0,0), (resolution[1],y2+15), (0,0,0), -1)

		if masked_img.shape[0] != width or masked_img.shape[1] != height:
			masked_img = cv2.resize(masked_img, (width, height))

		input_frames_masked.append(masked_img)

	input_frames_masked = np.array(input_frames_masked) / 255.
	input_frames_masked = np.pad(input_frames_masked, ((12, 12), (0,0), (0,0), (0,0)), 'edge')
	# print("Masked input images: ", input_frames_masked.shape)      	# num_framesx270x480x3 

	return input_frames_masked




def extract_feats(video_files, gestsync_model, result_dir):

	'''
	This function extracts the GestSync features from the video frames and saves them
	Args:
		- video_files (list): List of video files to process
		- gestsync_model (object): GestSync model object
		- result_dir (str): Directory to save the extracted features
	'''

	global err

	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	save_count=0
	prog_bar = tqdm(video_files)
	for video_file in prog_bar:

		try:
			folder = os.path.join(result_dir, video_file.split("/")[-2])
			output_fname = os.path.join(folder, video_file.split("/")[-1].split(".")[0]+".npy")
			# print("Output name: ", output_fname)

			if os.path.exists(output_fname):
				save_count+=1
				prog_bar.set_description("No of files saved = {} | Err = {}".format(save_count, err))
				continue

			if not os.path.exists(folder):
				os.makedirs(folder)

			
			video_frames = load_video_frames(video_file)
			if video_frames is None:
				err+=1
				print("Error loading video frames: ", video_file)
				prog_bar.set_description("No of files saved = {} | Err = {}".format(save_count, err))
				continue
			# print("Video frames: ", len(video_frames))

			kp_dict = get_keypoints(video_frames)

			if kp_dict is None:
				err+=1
				print("Error loading keypoints: ", video_file)
				prog_bar.set_description("No of files saved = {} | Err = {}".format(save_count, err))
				continue

			input_frames_masked = load_rgb_masked_frames(video_frames, kp_dict)
			if input_frames_masked is None:
				err+=1
				print("Error loading masked frames: ", video_file)
				prog_bar.set_description("No of files saved = {} | Err = {}".format(save_count, err))
				continue


			pose = torch.FloatTensor(np.array(input_frames_masked)).unsqueeze(0)
			pose_sync = []
			for i in range(0, pose.shape[1], 1):
				if (i+num_frames <= pose.shape[1]):
					pose_sync.append(pose[:, i:i+num_frames,])
			
			visual_emb = []
			for batch_idx in range(0, len(pose_sync), args.batch_size):

				pose_inp = pose_sync[batch_idx:batch_idx+args.batch_size]
				pose_inp = torch.stack(pose_inp)
				t,b,f,h,w,c = pose_inp.shape
				pose_inp = rearrange(pose_inp, 't b f h w c -> (t b) f h w c')
				pose_inp = pose_inp.permute(0,4,1,2,3)

				with torch.cuda.amp.autocast():
					with torch.no_grad():
												
						vid_emb = gestsync_model.forward_vid(pose_inp.cuda(), return_feats=False)
						vid_emb = torch.mean(vid_emb, axis=-1)

					vid_emb = rearrange(vid_emb, '(t b) vd -> b t vd',t=t,b=b)
					# print("Visual embedding rearrange: ", vid_emb.shape)               # BxTx512
								
					visual_emb.append(vid_emb[0]) # Since batch size is 1

			visual_emb = torch.cat(visual_emb, dim=0)
			visual_emb = visual_emb.detach().cpu().numpy()

			# print("Output fname: {} | Visual emb: {}".format(output_fname, visual_emb.shape))
			np.save(output_fname, visual_emb)
			save_count+=1

		except Exception as e:
			err += 1
			print("Error: ", e, " | Video file: ", video_file)
			prog_bar.set_description("No of files saved = {} | Err = {}".format(save_count, err))
			continue

		prog_bar.set_description("No of files saved = {} | Err = {}".format(save_count, err))



if __name__ == "__main__":

	gestsync_model = GestSync()
	gestsync_model = gestsync_model.cuda()
	gestsync_model = load_model(args.checkpoint_path_gestsync, gestsync_model)

	video_files = get_filelist(args)
	print("Total videos: {}".format(len(video_files)))

	# Split the filelist into shards for parallel processing
	num_per_shard = math.ceil(len(video_files)/args.nshard)
	start_id, end_id = num_per_shard*args.rank, num_per_shard*(args.rank+1)
	print("Start : End = ", start_id, ":", end_id)
	video_files = video_files[start_id: end_id]

	# Extract and save the GestSync features
	extract_feats(video_files, gestsync_model, args.result_dir)
