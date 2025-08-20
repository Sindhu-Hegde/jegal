import os
import argparse
import subprocess
import numpy as np
import cv2
import torch
import string
import pickle

from decord import VideoReader
from decord import cpu

from models.gestsync import *
from models.jegal import *
import whisperx
from utils.audio_utils import *

from einops import rearrange
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize the mediapipe holistic keypoint detection model
mp_holistic = mp.solutions.holistic

use_cuda = torch.cuda.is_available()
print("Use CUDA: ", use_cuda)


def get_args():

	parser = argparse.ArgumentParser(description="JEGAL Inference Script")

	parser.add_argument(
		'--checkpoint_path_gestsync',
		type=str,
		required=True,
		help="Path to the pretrained checkpoint for the GestSync model."
	)
	parser.add_argument(
		'--checkpoint_path_jegal',
		type=str,
		required=True,
		help="Path to the pretrained checkpoint for the JEGAL model."
	)
	parser.add_argument(
		'--modalities',
		type=str,
		choices=['vta', 'vt', 'va', 'ta', 'v', 't', 'a'],
		default="vta",
		help=(
			"Modalities to use:\n"
			"  v  = visual (video)\n"
			"  t  = text\n"
			"  a  = audio\n"
			"Combinations allowed: vta, vt, va, ta, v, t, a"
		)
	)
	parser.add_argument(
		'--video_path',
		type=str,
		default=None,
		help="Path to the video input file (required if modality includes 'v')."
	)
	parser.add_argument(
		'--text_path',
		type=str,
		default=None,
		help="Path to the text input file (required if modality includes 't')."
	)
	parser.add_argument(
		'--audio_path',
		type=str,
		default=None,
		help="Path to the audio input file (required if modality includes 'a')."
	)
	parser.add_argument(
		'--res_dir',
		type=str,
		default="results",
		help="Directory to store output results. Default is 'results'."
	)
	args = parser.parse_args()

	return args



def load_model(checkpoint_path, model):

	'''
	This function loads the model from the checkpoint path

	Args:
		- checkpoint_path (string) : Path of the checkpoint file
		- model (torch.nn.Module) : Model to load the checkpoint into
	Returns:
		- model (torch.nn.Module) : Model 
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

	frames = np.asarray(frames)
	print("Input video frames: ", frames.shape)

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

def check_visible_gestures(kp_dict):

	keypoints = kp_dict['kps']
	keypoints = np.array(keypoints)

	if len(keypoints)<25:
		return None, None
	
	ignore_idx = [0,1,2,3,4,5,6,7,8,9,10]

	count=0
	frame_pose_kps = []
	for frame_kp_dict in keypoints:

		pose = frame_kp_dict["pose"]
		left_hand = frame_kp_dict["left_hand"]
		right_hand = frame_kp_dict["right_hand"]

		if pose is None:
			continue
		
		if left_hand is None and right_hand is None:
			count+=1

		pose_kps = []
		for idx in range(len(pose)):
			# Ignore face keypoints
			if idx not in ignore_idx:
				x, y = pose[idx]["x"], pose[idx]["y"]
				pose_kps.append((x,y))

		frame_pose_kps.append(pose_kps)


	if count/len(keypoints) > 0.7 or len(frame_pose_kps)/len(keypoints) < 0.3:
		print("The gestures in the input video are not visible! Please give a video with visible gestures as input.")
		exit(0)

	print("Successfully verified the input video - Gestures are visible!")

def load_rgb_masked_frames(input_frames, kp_dict, width=480, height=270):

	'''
	This function masks the faces using the keypoints extracted from the frames

	Args:
		- input_frames (list) : List of frames extracted from the video
		- kp_dict (dict) : Dictionary containing the keypoints and the resolution of the frames
		- width (int) : Width of the frames
		- height (int) : Height of the frames
	Returns:
		- input_frames_masked (array) : Array of masked input frames
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

def validate_text_file(text_path: str):

	'''
	This function validates the text file

	Args:
		- text_path (string) : Path of the text file
	Returns:
		- True if the text file is valid, raises an error otherwise
	'''

	with open(text_path, "r", encoding="utf-8") as f:
		lines = [line.strip() for line in f.readlines()]

	# Basic length check
	if len(lines) < 4:
		raise ValueError(f"{text_path} is too short to be valid.")

	# Check headers
	if not lines[0].startswith("Text: "):
		raise ValueError("First line must start with 'Text: '")
	if not lines[1].startswith("Lang: "):
		raise ValueError("Second line must start with 'Lang: '")
	if lines[2] != "":
		raise ValueError("Third line must be empty.")
	if lines[3] != "WORD, START, END, SCORE":
		raise ValueError("Fourth line must be 'WORD, START, END, SCORE'")

	return True
	
def preprocess_text(text):

	'''
	This function preprocesses the text

	Args:
		- text (string) : Text to preprocess
	Returns:
		- text (string) : Preprocessed text
	'''
	
	# Text preprocessing: Convert to lower case and remove punctuation
	text = text.lower()
	text = "".join([i for i in text if i not in string.punctuation])		    
	return text
	
def load_text(text_path, fps=25):

	'''
	This function loads the text from the text file

	Args:
		- text_path (string) : Path of the text file
		- fps (int) : Frames per second (default: 25)
	Returns:
		- text (string) : Text
		- word_boundaries (list) : Word boundaries
	'''

	# Validate the text file
	validate_text_file(text_path)

	# Load the text
	with open(text_path, 'r', encoding='utf-8') as file:
		lines = file.readlines()

	# Skip the first three lines
	metadata = lines[4:]

	# Process the text and get the word boundaries
	text="" 
	word_boundaries = [] 
	for i in range(len(metadata)):
		row = metadata[i].split(", ")
		word = preprocess_text(row[0])
		if word != "":
			text += word
			if i != (len(metadata)-1):
				text += " "
			start = round(float(row[1])*fps)
			end = round(float(row[2])*fps)
			word_boundaries.append([word, start, end])    

	text = [text]
	word_boundaries = [word_boundaries]
	print("Input text: ", text)
	print("Input word boundaries: ", word_boundaries)

	# Return the text and the word boundaries
	return text, word_boundaries

def get_word_boundaries_whisperx(audio_path, res_dir, batch_size=8):

	'''
	This function gets the word boundaries from the audio file using WhisperX model

	Args:
		- audio_path (string) : Path of the audio file
		- res_dir (string) : Path of the result directory
		- batch_size (int) : Batch size for the WhisperX model (default: 8)
	Returns:
		- output_fname (string) : Path of the output file
	'''

	model = whisperx.load_model("large-v3", device="cpu", compute_type="float32")
	print("Loaded the WhisperX model")

	try:
		# print("File: ", audio_file)
		output_fname = os.path.join(res_dir, "word_boundaries.txt")

		audio = whisperx.load_audio(audio_path)
		result = model.transcribe(audio, batch_size=batch_size)
		lang = result['language']

		f = open(output_fname, "w")
		f.write("Text: ")
		for idx in range(len(result["segments"])):
			f.write(result["segments"][idx]["text"])
		f.write("\nLang: " + lang)

		model_a, metadata = whisperx.load_align_model(language_code=lang, device="cpu")
		result = whisperx.align(result["segments"], model_a, metadata, audio, "cpu", return_char_alignments=False)


		f.write("\n\nWORD, START, END, SCORE\n")
		for idx in range(len(result["segments"])):
			words = result["segments"][idx]["words"]
			# print("Words: ", words)
			for line in words:
				if not "start" in list(line.keys()):
					f.write(line["word"]+"\n")
				else:
					f.write(line["word"]+", "+str(line["start"])+", "+str(line["end"])+", "+str(line["score"])+"\n")

		f.close()
		# print("Written transcript: ", output_fname_pred)
		
	except Exception as e:
		print("Error in getting word boundaries: ", e)
		return None

	return output_fname

	
def load_audio(audio_path, res_dir):

	'''
	This function loads the audio from the audio file

	Args:
		- audio_path (string) : Path of the audio/video file
		- res_dir (string) : Path of the result directory
	Returns:
		- audio (tensor) : Audio tensor
		- audio_mask (tensor) : Audio mask
	'''

	if audio_path.endswith(".mp4") or audio_path.endswith(".mkv") or audio_path.endswith(".avi"):
		try:
			wav_file  = os.path.join(res_dir, "audio.wav")

			subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -async 1 -ac 1 -vn \
						-acodec pcm_s16le -ar 16000 %s' % (audio_path, wav_file), shell=True)

		except:
			print("Oops! Could not load the audio file in the given input video. Please check the input and try again")
			return None, None
	else:
		wav_file = audio_path

	# Load the audio
	audio = load_wav(wav_file).astype('float32')
	audio = np.array(audio)
	audio = torch.FloatTensor(audio)
	# print("Wav: ", audio.shape)
	
	# Convert to mel spectrogram
	mel, _, _, _ = wav2filterbanks(audio.unsqueeze(0))
	print("Input audio mel: ", mel.shape)

	# Create a mask for the audio
	audio_mask = torch.ones((mel.shape[0]//4))
	audio_mask = audio_mask.unsqueeze(0)

	return mel, audio_mask
	

def get_gestsync_feats(frames, gestsync_model):

	'''
	This function extracts the GestSync features from RGB input

	Args:
		- frames (tensor) : Pose tensor
		- gestsync_model (torch.nn.Module) : GestSync model
	Returns:
		- visual_emb (tensor) : GestSync features
	'''

	num_frames = 25
	pose_sync = []
	for i in range(0, frames.shape[1], 1):
		if (i+num_frames <= frames.shape[1]):
			pose_sync.append(frames[:, i:i+num_frames,])

	batch_size = 48
	visual_emb = []

	with torch.cuda.amp.autocast():
		with torch.no_grad():
			for batch_idx in range(0, len(pose_sync), batch_size):

				pose_inp = pose_sync[batch_idx:batch_idx+batch_size]
				# print("Pose inp: ", len(pose_inp))

				pose_inp = torch.stack(pose_inp)
				t,b,f,h,w,c = pose_inp.shape
				pose_inp = rearrange(pose_inp, 't b f h w c -> (t b) f h w c')
				pose_inp = pose_inp.permute(0,4,1,2,3)
				# print("Model input - pose:", pose_inp.shape) # (B*T) x 3 x 25 x 270 x 480
												
				vid_emb = gestsync_model.forward_vid(pose_inp.cuda(), return_feats=False)
				vid_emb = torch.mean(vid_emb, axis=-1)
				# print("Visual embedding mean: ", vid_emb.shape)                 # (B*T)x1024
							
				visual_emb.append(vid_emb)

				torch.cuda.empty_cache()

			visual_emb = torch.cat(visual_emb, dim=0)
			visual_emb = rearrange(visual_emb, '(tv bv) vd -> bv tv vd', bv=1)
			# print("Visual embedding: ", visual_emb.shape)               # BxTx1024
			
	return visual_emb


	
def extract_embs(gestsync_model, jegal_model, res_dir, modalities="vta"):

	'''
	This function extracts the JEGAL embeddings from the input modalities

	Args:
		- gestsync_model (torch.nn.Module) : GestSync model 
		- jegal_model (torch.nn.Module) : JEGAL model
		- res_dir (string) : Path of the result directory to save the extracted embeddings
		- modalities (string) : Modalities to extract the embeddings from (default: "vta")
	'''

	if not os.path.exists(res_dir):
		os.makedirs(res_dir)

	# Initialize
	visual_feats, visual_mask, text, audio, audio_mask, word_boundaries, fname = None, None, None, None, None, None, None


	# Load the input video
	if args.video_path is not None:

		# Extract the video frames
		video_frames = load_video_frames(args.video_path)

		# Extract the keypoints from the video frames (needed to mask the faces for the GestSync model)
		kp_dict = get_keypoints(video_frames)

		# Validate if the input video has visible gestures
		check_visible_gestures(kp_dict)

		# Mask the faces for the GestSync model
		input_frames_masked = load_rgb_masked_frames(video_frames, kp_dict)
		input_frames_masked = torch.FloatTensor(np.array(input_frames_masked)).unsqueeze(0)
		print("Input masked frames: ", input_frames_masked.shape)

		print("Extracting pre-trained GestSync features...")
		visual_feats = get_gestsync_feats(input_frames_masked, gestsync_model)
		visual_feats = visual_feats.cuda()

		visual_mask = torch.ones((visual_feats.shape[1]))
		visual_mask = visual_mask.unsqueeze(0).cuda()

		fname = os.path.basename(args.video_path).split(".")[0]

		print("Input visual features: ", visual_feats.shape)
		print("Input visual mask: ", visual_mask.shape)

	# Load the input text
	if args.text_path is not None:
		text, word_boundaries = load_text(args.text_path)
		if fname is None:
			fname = os.path.basename(args.text_path).split(".")[0]
	
	# Load the input audio 
	if args.audio_path is not None:
		print("Loading audio...")

		# Load audio and audio mask
		audio, audio_mask = load_audio(args.audio_path, res_dir)
		if audio is None:
			print("Error in loading audio. Please check the input and try again")
			exit(0)
		audio = audio.cuda()
		audio_mask = audio_mask.cuda()

		# Get the word boundaries from audio using WhisperX if not provided in the input text file
		if word_boundaries is None:
			print("Getting word boundaries using WhisperX...")

			# text = ["and you have one very long pipeline"]
			# word_boundaries = [[['and', 0, 2], ['you', 2, 4], ['have', 5, 9], ['one', 12, 15], ['very', 16, 21], ['long', 22, 27], ['pipeline', 28, 44]]]

			text_file = get_word_boundaries_whisperx(args.audio_path, res_dir)
			if text_file is None:
				print("Error in getting word boundaries using WhisperX. Please check the input and try again")
				exit(0)
		
			text, word_boundaries = load_text(text_file)



		if fname is None:
			fname = os.path.basename(args.audio_path).split(".")[0]

	print("Extracting JEGAL embeddings...")
	print("------------------------------------------------")
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

			gesture_emb, content_emb = jegal_model.forward_inference(visual_feats=visual_feats, visual_mask=visual_mask, text=text, audio=audio, audio_mask=audio_mask.unsqueeze(1), word_boundaries=word_boundaries)
			

	# Normalize the embeddings
	if gesture_emb is not None:
		gesture_emb = F.normalize(gesture_emb, p=2, dim=-1) 
		gesture_emb = gesture_emb[0].cpu().numpy()
		print("Extracted gesture embeddings: ", gesture_emb.shape)
	if content_emb is not None:
		content_emb = F.normalize(content_emb, p=2, dim=-1)
		content_emb = content_emb[0].cpu().numpy()
		print("Extracted content embeddings: ", content_emb.shape)
	
	print("------------------------------------------------")
	
	# Save the embeddings
	feat_dict = {"gesture_emb":gesture_emb, "content_emb":content_emb, "info":{"fname":fname, "word_boundaries":word_boundaries[0], "text":text[0]}}
	output_fname = os.path.join(res_dir, fname + ".pkl")
	with open(output_fname, 'wb') as f:
		pickle.dump(feat_dict, f)
	print("Saved the embeddings: ", output_fname)


if __name__ == "__main__":

	# Validate modality-specific inputs
	modality_to_arg = {
		'v': 'video_path',
		'a': 'audio_path'
	}

	args = get_args()

	for m in args.modalities:
		if m in modality_to_arg:
			if getattr(args, modality_to_arg[m]) is None:
				raise ValueError(f"--{modality_to_arg[m]} must be specified when modality '{m}' is used.")
		elif m == 't':
			if args.text_path is None and args.audio_path is None:
				raise ValueError(
					"For modality 't', you must specify either --text_path or --audio_path "
					"(since text can be extracted from audio)."
				)

	# Load the GestSync model
	gestsync_model = GestSync()
	gestsync_model = gestsync_model.cuda()
	gestsync_model = load_model(args.checkpoint_path_gestsync, gestsync_model)

	# Load the JEGAL model
	jegal_model = JEGAL()
	jegal_model = jegal_model.cuda()
	jegal_model = load_model(args.checkpoint_path_jegal, jegal_model)

	# Extract and save JEGAL embeddings
	print("Modalities being used: ", args.modalities)
	extract_embs(gestsync_model, jegal_model, args.res_dir, modalities=args.modalities)
