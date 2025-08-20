import os
import argparse, math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
from tqdm import tqdm

from decord import VideoReader
from decord import cpu

import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict
import pandas as pd

import json
import h5py
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', required=True, help='Path containing the pre-processed videos')  
parser.add_argument('--file', required=True, help='Path to the file containing the video paths')
parser.add_argument('--result_path', required=True, help='Path to save the extracted keypoint files')

parser.add_argument('--rank', required=False,  type=int, default=0)
parser.add_argument('--nshard', required=False, type=int, default=1)

args = parser.parse_args()

# Initialize the mediapipe holistic keypoint detection model
mp_holistic = mp.solutions.holistic


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
		filelist.append(os.path.join(args.data_path, df.iloc[i]["filename"]+".avi"))
	
	return filelist

def load_video_frames(video_file):

	'''
	This function loads the video frames from the video file
	Args:
		- video_file (str): The path to the video file
	Returns:
		- frames (list): The list of frames from the video
	'''

	# Read the video
	try:
		vr = VideoReader(video_file, ctx=cpu(0))
	except:
		return None

	# Extract the frames
	frames = []
	for k in range(len(vr)):
		frames.append(vr[k].asnumpy())

	return frames


def get_keypoints(frames):

	'''
	This function extracts the keypoints from the video frames
	Args:
		- frames (list): The list of frames from the video
	Returns:
		- all_frame_kps (list): The list of keypoints for each frame
		- resolution (tuple): The resolution of the video
	'''

	resolution = frames[0].shape
	all_frame_kps = []

	with mp_holistic.Holistic(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	) as holistic:
		try:
			for frame in frames:

				results = holistic.process(frame)

				pose, pose_world, left_hand, right_hand, face = None, None, None, None, None
				if results.pose_landmarks is not None:
					pose = protobuf_to_dict(results.pose_landmarks)['landmark']
				if results.pose_world_landmarks is not None:
					pose_world = protobuf_to_dict(results.pose_world_landmarks)['landmark']
				if results.left_hand_landmarks is not None:
					left_hand = protobuf_to_dict(results.left_hand_landmarks)['landmark']
				if results.right_hand_landmarks is not None:
					right_hand = protobuf_to_dict(results.right_hand_landmarks)['landmark']
				if results.face_landmarks is not None:
					face = protobuf_to_dict(results.face_landmarks)['landmark']

				frame_dict = {"pose":pose, "pose_world":pose_world, "left_hand":left_hand, "right_hand":right_hand, "face":face}

				all_frame_kps.append(frame_dict)
			
		except:
			return None

	return all_frame_kps, resolution

def process(args):

	'''
	This function processes the videos, extracts and saves the keypoints
	Args:
		- args (argparse): The arguments object containing the file and data path
	'''

	# Get the filelist
	filelist = get_filelist(args)
	print("No of videos: ", len(filelist))

	# Split the filelist into shards for parallel processing
	num_per_shard = math.ceil(len(filelist)/args.nshard)
	start_id, end_id = num_per_shard*args.rank, num_per_shard*(args.rank+1)
	print("Start : End = ", start_id, ":", end_id)
	filelist = filelist[start_id: end_id]
	print("Total files: ", len(filelist))

	# Create the result folder if it doesn't exist
	if not os.path.exists(args.result_path):
		os.makedirs(args.result_path)

	# Extract the keypoints for each video
	for video in tqdm(filelist):

		print("Video file: ", video)
		folder = os.path.join(args.result_path, video.split("/")[-2])
		if not os.path.exists(folder):
			os.makedirs(folder)

		kp_file = os.path.join(folder, video.split("/")[-1].split(".")[0]+"_mediapipe_kps.pkl")
		if os.path.exists(kp_file):
			continue

		frames = load_video_frames(video)
		if frames is None:
			print("Could not extract the frames from the video {}! Skipping this video...".format(video))
			continue
		print("Frames: ", len(frames))

		try:
			all_frame_kps, resolution = get_keypoints(frames)
		except:
			print("Could not extract the keypoints from the video {}! Skipping this video...".format(video))
			continue
		print("Keypoints: ", len(all_frame_kps))

		if len(frames) != len(all_frame_kps):
			print("Frames and keypoints do not match for the video {}! Skipping this video...".format(video))
			continue

		kp_dict = {"kps":all_frame_kps, "resolution":resolution}

		with open(kp_file, 'wb') as f:
			pickle.dump(kp_dict, f)

		print("Saved the output kp file: ", kp_file)


if __name__ == '__main__':

	process(args)