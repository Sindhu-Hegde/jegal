import os
import numpy as np
import torch
from torch.utils import data
from torch import nn
from utils.audio_utils import *
import string

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 


class DataGenerator_Train(data.Dataset):

	def __init__(self, df, feature_dir, sample_rate=16000, fps=25):

		self.df = df
		self.feature_dir = feature_dir
		self.sample_rate = sample_rate
		self.fps = fps

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):

		data = self.df.iloc[index]

		file = data.filename
		text_fname = data.text_path
		audio_fname = data.audio_path

		if not os.path.exists(text_fname):
			# print("Text file does not exist: ", text_fname)
			return None

		if not os.path.exists(audio_fname):
			# print("Audio file does not exist: ", audio_fname)
			return None


		# Load visual features (pre-trained GestSync features)
		gestsync_fname = os.path.join(self.feature_dir, file+".npy")
		visual_feats, visual_mask = self.load_visual_feats(gestsync_fname, start_frame, end_frame)
		if visual_feats is None:
			return None
		visual_feats = torch.FloatTensor(np.array(visual_feats))

		# Load text
		text, start_frame, end_frame, word_boundaries = self.load_text(text_fname)
		if text is None:
			return None

		# Load audio		
		audio, audio_mask = self.load_audio(audio_fname, start_frame, end_frame)
		if audio is None:
			return None		

		out_dict = {
			'visual_feats': visual_feats,
			'visual_mask': visual_mask,
			'text': text,
			'audio': audio,
			'audio_mask': audio_mask,
			'file': file,
			'word_boundaries': word_boundaries,
			'info': data
		}

		return out_dict

	def load_visual_feats(self, fname, start_frame, end_frame):

		# Read the video
		try:
			feats = np.load(fname)
			# print("Initial visual features: ", feats.shape) 			
			
			if start_frame is not None and end_frame is not None:
				selected_feats = feats[start_frame:end_frame+1,]
				# print("Feats selected: ", selected_feats.shape)
			else:
				selected_feats = feats
				# print("Feats selected: ", selected_feats.shape)

			
			visual_feats = np.array(selected_feats)
			# print("Visual features: ", visual_feats.shape)

			visual_mask = torch.ones((len(visual_feats)))
			# print("Visual mask: ", visual_mask.shape)


			if visual_feats.shape[1] != 1024:
				return None, None

		except:
			# print("Error in loading GestSync feats: ", fname)
			return None, None
		
		
		return visual_feats, visual_mask

		
	def load_text(self, fname):

		def preprocess_text(text):
	
			text = text.lower()
			text = "".join([i for i in text if i not in string.punctuation])		    
			return text

		try:
			with open(fname, 'r', encoding='utf-8') as file:
				lines = file.readlines()

			# Skip the first three lines
			metadata = lines[4:]
			
			# print("Metadata: ",metadata)
			# print("Total words: ", len(metadata))
			
			if len(metadata) < 5:
				return None, None, None, None

			max_words = np.random.randint(10, 20)
			num_words = np.random.randint(5, min(len(metadata), max_words)+1)				
			# print("Num selected words: ", num_words)
			
			start_word_idx = np.random.randint(0, len(metadata)-num_words+1)
			# print("Start word idx: ", start_word_idx)

			start_time = metadata[start_word_idx].split(", ")[1]
			# print("Start time: ", start_time)

			end_time = metadata[start_word_idx+(num_words-1)].split(", ")[2]
			# print("End time: ", end_time)

			start_frame, end_frame = round(float(start_time)*self.fps), round(float(end_time)*self.fps)
			# print("DL - Start frame: {} | End frame: {} || Total frames: {}".format(start_frame, end_frame, end_frame-start_frame+1))

			text=""
			word_boundaries = []
			for i in range(start_word_idx, start_word_idx+num_words):
				row = metadata[i].split(", ")
				inp = row[0]
				word = preprocess_text(inp)
				if word == "":
					continue
				text += word
				if i != (start_word_idx+num_words-1):
					text += " "
				start = round(float(row[1])*self.fps)
				end = round(float(row[2])*self.fps)
				word_boundaries.append([word, start, end])

		except:
			return None, None, None, None
		
		# print("Text: ", text)
		# print("Word boundaries: ", word_boundaries)

		return text, start_frame, end_frame, word_boundaries
	

	def load_audio(self, wavpath, start_frame, end_frame):
			
		try:
			# print("Audio file: ", wavpath)
			audio = load_wav(wavpath).astype('float32')
			# print("Wav: ", audio.shape)
		
			if start_frame is not None and end_frame is not None:
				aud_fact = int(np.round(self.sample_rate / self.fps))
				audio_window = audio[aud_fact*start_frame : aud_fact*(end_frame+1)]
			else:
				audio_window = audio

			audio_window = np.array(audio_window)
			audio_window = torch.FloatTensor(audio_window)
			# print("Input audio: ", audio_window.shape)
			
			mel, _, _, _ = wav2filterbanks(audio_window.unsqueeze(0))
			mel = mel.squeeze(0).cpu()
			# print("Audio mel: ", mel.shape)

			audio_mask = torch.ones((mel.shape[0]//4))

			return mel, audio_mask
			
		except:
			# print("Error in loading audio: ", wavpath)
			return None, None


class DataGenerator_Test(data.Dataset):

	def __init__(self, df, video_dir, feature_dir, modalities="vta"):

		self.df = df
		self.video_dir = video_dir
		self.feature_dir = feature_dir
		self.modalities = modalities

		
	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):

		data = self.df.iloc[index]
		file = data.filename
		text = data.phrase
		word_boundaries = data.word_boundaries

		if "v" in self.modalities:
			gestsync_fname = os.path.join(self.feature_dir, file+".npy")
			if not os.path.exists(gestsync_fname):
				print("Visual feats file does not exist: ", gestsync_fname)
				return None

			visual_feats, visual_mask = self.load_visual_feats(gestsync_fname)
			visual_feats = torch.FloatTensor(np.array(visual_feats))
			if visual_feats is None:
				print("Error loading visual features, check the file: ", gestsync_fname)
				return None
		else:
			visual_feats, visual_mask = torch.FloatTensor([0]), torch.FloatTensor([0])
		

		if "a" in self.modalities:
			audio_fname = os.path.join(self.video_dir, file+".wav")
			if not os.path.exists(audio_fname):
				print("Audio file does not exist: ", audio_fname)
				return None
			audio, audio_mask = self.load_audio(audio_fname)
			if audio is None:
				print("Error loading audio, check the file: ", audio_fname)
				return None
		else:
			audio, audio_mask = torch.FloatTensor([0]), torch.FloatTensor([0])

	
		out_dict = {
			'visual_feats': visual_feats,
			'visual_mask': visual_mask,
			'text': text,
			'audio': audio,
			'audio_mask': audio_mask,
			'file': file,
			'word_boundaries': word_boundaries,			
			'info': data
		}

		return out_dict


	def load_visual_feats(self, fname):

		# Read the video
		try:
			feats = np.load(fname)
			visual_feats = np.array(feats)
			# print("Input feat: ", input_feat.shape)

			visual_mask = torch.ones((len(visual_feats)))
			# print("Pose mask: ", pose_mask.shape)

			# Validate GestSync features shape
			if visual_feats.shape[1] != 1024:
				return None, None
		except:
			# print("Error in loading visual features: ", fname)
			return None, None
		
		return visual_feats, visual_mask
	

	def load_audio(self, wavpath):
			
		try:
			audio = load_wav(wavpath).astype('float32')
			audio = np.array(audio)
			audio = torch.FloatTensor(audio)
			
			mel, _, _, _ = wav2filterbanks(audio.unsqueeze(0))
			mel = mel.squeeze(0).cpu()
			# print("Audio mel: ", mel.shape)

			audio_mask = torch.ones((mel.shape[0]//4))

			return mel, audio_mask
			
		except:
			# print("Error in loading audio: ", wavpath)
			return None, None

def collate_data(data):

	visual_feats = []
	visual_mask = []
	text = []
	audio = []
	audio_mask = []
	word_boundaries = []
	files = []	
	info = []

	for sample in data:

		if sample is None:
			continue

		feats = sample['visual_feats']
		feat_mask = sample['visual_mask']
		phrase = sample['text']
		spec = sample['audio']
		spec_mask = sample['audio_mask']
		boundaries = sample['word_boundaries']
		file = sample['file']
		inf = sample['info']
		
			
		visual_feats.append(feats)
		visual_mask.append(feat_mask)
		text.append(phrase)
		audio.append(spec)
		audio_mask.append(spec_mask)
		word_boundaries.append(boundaries)
		files.append(file)
		info.append(inf)
		
		
	if len(visual_feats) > 0:
		visual_feats = nn.utils.rnn.pad_sequence(visual_feats, batch_first=True, padding_value=0)
		visual_mask = nn.utils.rnn.pad_sequence(visual_mask, batch_first=True, padding_value=0)
		audio = nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
		audio_mask = nn.utils.rnn.pad_sequence(audio_mask, batch_first=True, padding_value=0)
	else:
		return 0


	out_dict = {
			'visual_feats': visual_feats,
			'visual_mask': visual_mask,
			'text': text,
			'audio': audio,
			'audio_mask': audio_mask,
			'word_boundaries': word_boundaries,
			'file': files,
			'info': info,
		}

	return out_dict		