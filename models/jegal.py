import torch
import torch.nn as nn
from torch.autograd import Variable
import math, copy
import random

from models.modules import *
from transformers import AutoTokenizer, XLMRobertaModel

from einops.layers.torch import Rearrange
from einops import rearrange

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
mroberta = XLMRobertaModel.from_pretrained("xlm-roberta-base")

class JEGAL(nn.Module):

	def __init__(self, fusion_strategy='concat', N=6, N_text=3, d_model=512, d_model_text=768, h=8, dropout=0.1):
		super().__init__()

		c = copy.deepcopy

		self.fusion_strategy = fusion_strategy

		self.proj_ip_rgb = nn.Sequential(nn.Linear(1024, 512), 
									nn.LayerNorm(512),
									nn.ReLU(), 
									nn.Linear(512, 512),)
		attn_rgb = MultiHeadedAttention_Transformer(h, d_model, dropout=dropout)
		ff_rgb = PositionwiseFeedForward_Transformer(d_model, d_model*4, dropout)
		self.position_rgb = PositionalEncoding_Transformer(d_model, dropout)
		self.encoder_rgb = Encoder_Transformer(EncoderLayer_Transformer(d_model, c(attn_rgb), c(ff_rgb), dropout), N)
		self.proj_op_rgb = nn.Linear(512, 512)

		attn_text = MultiHeadedAttention_Transformer(h, d_model_text, dropout=dropout)
		ff_text = PositionwiseFeedForward_Transformer(d_model_text, d_model_text*4, dropout)
		self.encoder_text = Encoder_Transformer(EncoderLayer_Transformer(d_model_text, c(attn_text), c(ff_text), dropout), N_text)
		self.proj_op_text = nn.Linear(768, 256)


		self.cnn  = nn.Sequential(
				nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
				nn.BatchNorm2d(32),
				nn.ReLU(),
				
				nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
				nn.BatchNorm2d(64),
				nn.ReLU(),

				nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
				nn.BatchNorm2d(128),
				nn.ReLU(),

				nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 3), padding=(1, 1)),
				nn.BatchNorm2d(256),
				nn.ReLU(),

				nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 3), padding=(1, 1)),
				nn.BatchNorm2d(256),
				nn.ReLU(),

				nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 3), padding=(0, 0)),
			)
		self.proj_op_audio = nn.Linear(256, 256)


		self.proj_op_fusion_content = nn.Sequential(nn.Linear(512, 512), 
									nn.ReLU(), 
									nn.Linear(512, 512))

		self.proj_op_align_gesture = nn.Sequential(nn.Linear(512, 512),
													nn.ReLU(),
													nn.Linear(512, 512))
		self.proj_op_align_content = nn.Sequential(nn.Linear(512, 512),
													nn.ReLU(),
													nn.Linear(512, 512))
	
	def forward_gestures(self, x, x_mask=None):

		x = self.proj_ip_rgb(x)
		# print("X proj: ", x.shape)

		position_enc = self.position_rgb(x)
		# print("Pos encoding: ", position_enc.shape)						# BxTx512
	
		transformer_enc = self.encoder_rgb(position_enc, x_mask)
		# print("Transformer: ", transformer_enc.shape)     				# BxTx512

		proj_op = self.proj_op_rgb(transformer_enc)
		# print("Output MLP: ", proj_op.shape)								# BxTx512
		
		return proj_op

	
	def forward_text(self, x, x_mask=None):

		transformer_enc = self.encoder_text(x, x_mask)
		# print("Transformer: ", transformer_enc.shape)     				# BxTx256

		proj_op = self.proj_op_text(transformer_enc)
		# print("Output MLP: ", proj_op.shape)								# BxTx256

		return proj_op

	def forward_audio(self, x, x_mask=None):

		cnn = self.cnn(x.unsqueeze(1)).squeeze(-1).permute(0, 2, 1)
		# print("CNN: ", cnn.shape)				

		proj_op = self.proj_op_audio(cnn)
		# print("Output MLP: ", out.shape)									# BxTx256

		return proj_op
	
	
	def get_roberta_embeddings(self, text):

		with torch.no_grad():
			text_batch = [words.split(" ") for words in text]
			text_input = tokenizer(text_batch, return_tensors="pt", padding=True, is_split_into_words=True, return_offsets_mapping=True)

			input_ids = text_input["input_ids"]
			text_mask = text_input["attention_mask"]
			offset_mapping = text_input["offset_mapping"]

			text_output = mroberta(input_ids, attention_mask=text_mask)
			text_emb = text_output.last_hidden_state.cuda()

		return text_emb, text_mask, text_batch, input_ids, offset_mapping

	def get_word_level_embs(self, text_emb, text, input_ids, offset_mapping, audio_emb=None, word_boundaries=None):

		batch_size = input_ids.size(0)

		# Find the special token IDs for <s>, </s>, and padding tokens
		special_token_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

		# Iterate over each sentence in the batch
		word_text_emb, word_audio_emb = [], []
		invalid_sample_idx = []
		for batch_idx in range(batch_size):  

			# Get tokenized words and offsets for this batch element
			batch_offset_mapping = offset_mapping[batch_idx]
			
			# Collect the start indices for each word (excluding special tokens)
			word_start_indices = []
			for i, word_offset in enumerate(batch_offset_mapping):
				if word_offset[0] == 0 and input_ids[batch_idx][i] not in special_token_ids:  # Skip special tokens
					word_start_indices.append(i)
			
			# Extract embeddings for each word in the sentence
			embs_text, embs_audio = [], []

			if audio_emb is not None:
				actual_start = int(word_boundaries[batch_idx][0][1])
			
			valid_sample=True
			for idx, word in enumerate(text[batch_idx]):

				if idx >= len(word_start_indices):
					# If there are more words than start indices, break the loop
					valid_sample=False
					invalid_sample_idx.append(batch_idx)
					break

				# Get the range of subwords corresponding to the word
				if idx < len(word_start_indices) - 1:
					subword_indices = range(word_start_indices[idx], word_start_indices[idx + 1])
				else:
					subword_indices = range(word_start_indices[idx], input_ids.shape[1])
				
				# Gather embeddings for subwords and perform averaging if necessary
				subword_embeddings = text_emb[batch_idx, list(subword_indices)]
				
				# If the word is tokenized into subwords, average the subword embeddings
				if len(subword_embeddings) > 1:
					word_embedding = subword_embeddings.mean(dim=0)
				else:
					word_embedding = subword_embeddings[0]
				
				embs_text.append(word_embedding)

				if audio_emb is not None:
					# Get the start and end frame indices for the word
					start_frame, end_frame = int(word_boundaries[batch_idx][idx][1])-actual_start, int(word_boundaries[batch_idx][idx][2])-actual_start
					
					# Extract the audio embeddings for the frames corresponding to the word
					word_audio_embeddings = audio_emb[batch_idx, start_frame:end_frame + 1]
					
					# If the word spans multiple frames, average the frame embeddings
					if len(word_audio_embeddings) > 1:
						word_audio_embedding = word_audio_embeddings.mean(dim=0)
					else:
						word_audio_embedding = word_audio_embeddings[0]
					
					embs_audio.append(word_audio_embedding)


			if valid_sample:
				if len(embs_text) <= 0:
					invalid_sample_idx.append(batch_idx)
				else:
					embs_text = torch.stack(embs_text)
					word_text_emb.append(embs_text)

					if audio_emb is not None:
						embs_audio = torch.stack(embs_audio)
						word_audio_emb.append(embs_audio)
				
		return word_text_emb, word_audio_emb, invalid_sample_idx

	def get_audio_word_level_embs(self, audio_emb, word_boundaries, invalid_sample_idx=None):

		batch_size = audio_emb.size(0)

		word_audio_emb = []
		for batch_idx in range(batch_size):  

			if invalid_sample_idx is not None:
				if batch_idx in invalid_sample_idx:
					continue

			actual_start = int(word_boundaries[batch_idx][0][1])
			embs_audio = []
			# print("Word boundaries: ", word_boundaries[batch_idx])
			for idx in range(len(word_boundaries[batch_idx])):

				# Get the start and end frame indices for the word
				start_frame, end_frame = int(word_boundaries[batch_idx][idx][1])-actual_start, int(word_boundaries[batch_idx][idx][2])-actual_start
				
				# Extract the audio embeddings for the frames corresponding to the word
				word_audio_embeddings = audio_emb[batch_idx, start_frame:end_frame + 1]				# print("Word audio embeddings: ", audio_emb[batch_idx].shape, word_audio_embeddings.shape)
				
				# If the word spans multiple frames, average the frame embeddings
				if len(word_audio_embeddings) > 1:
					word_audio_embedding = word_audio_embeddings.mean(dim=0)
				else:
					word_audio_embedding = word_audio_embeddings[0]
				
				embs_audio.append(word_audio_embedding)

			if len(embs_audio) > 0:
				embs_audio = torch.stack(embs_audio)
				word_audio_emb.append(embs_audio)
			else:
				if invalid_sample_idx is not None:
					invalid_sample_idx.append(batch_idx)
				else:
					invalid_sample_idx = [batch_idx]
				
		return word_audio_emb, invalid_sample_idx
	
	def pad_wordlevel_embs(self, wordlevel_embs):
		
		# Find the longest sequence
		max_length = max([emb.size(0) for emb in wordlevel_embs])  
		
		padded_embs, padded_mask = [], []
		for emb in wordlevel_embs:
			pad_length = max_length - emb.size(0)

			# Pad the embeddings to the maximum length
			padded_embs.append(F.pad(emb, (0, 0, 0, pad_length), mode='constant', value=0))

			# Store the valid length before padding
			padded_mask.append(emb.size(0))
			
		# Stack into tensors for batched processing
		padded_embs = torch.stack(padded_embs)

		return padded_embs, padded_mask
		
	def forward(self, visual_feats, visual_mask, text, audio, audio_mask, word_boundaries):

		gesture_emb = self.forward_gestures(visual_feats, visual_mask.unsqueeze(1))
			
		invalid_sample_idx = None
		if random.random() > 0.5:	
			if random.random() > 0.5:
				# print("Audio dropped")
				text_emb_roberta, text_mask, text_batch, input_ids, offset_mapping = self.get_roberta_embeddings(text)
				text_emb_subwords = self.forward_text(text_emb_roberta, text_mask.unsqueeze(1).cuda())
				text_emb_words, _, invalid_sample_idx = self.get_word_level_embs(text_emb_subwords, text_batch, input_ids, offset_mapping, audio_emb=None, word_boundaries=None)
				text_emb, content_mask = self.pad_wordlevel_embs(text_emb_words)
				audio_emb = torch.zeros_like(text_emb).cuda()  # Drop audio
			else:
				# print("Text dropped")
				audio_emb_frames = self.forward_audio(audio, audio_mask)
				audio_emb_words, invalid_sample_idx = self.get_audio_word_level_embs(audio_emb_frames, word_boundaries)
				audio_emb, content_mask = self.pad_wordlevel_embs(audio_emb_words)
				text_emb = torch.zeros_like(audio_emb).cuda()  # Drop text

			
		else:
			# print("No content dropped")
			text_emb_roberta, text_mask, text_batch, input_ids, offset_mapping = self.get_roberta_embeddings(text)
			text_emb_subwords = self.forward_text(text_emb_roberta, text_mask.unsqueeze(1).cuda())
			audio_emb_frames = self.forward_audio(audio, audio_mask)

			text_emb_words, audio_emb_words, invalid_sample_idx = self.get_word_level_embs(text_attn_subwords, text_batch, input_ids, offset_mapping, audio_attn_frames, word_boundaries)
			text_emb, content_mask = self.pad_wordlevel_embs(text_emb_words)
			audio_emb, _ = self.pad_wordlevel_embs(audio_emb_words)
			
		if invalid_sample_idx is not None:
			if len(invalid_sample_idx) > 0:
				valid_indices = [i for i in range(gesture_emb.size(0)) if i not in invalid_sample_idx]
				gesture_emb = gesture_emb[valid_indices]
				visual_mask = visual_mask[valid_indices]
				content_mask = content_mask[valid_indices]
				word_boundaries = [word_boundaries[i] for i in valid_indices]
			
		# print("Gesture emb: {}".format(gesture_emb.shape))							# BxTx512
		# print("Text emb: {}".format(text_emb.shape))									# BxNx256
		# print("Audio emb: {}".format(audio_emb.shape))								# BxNx256
		

		# Fuse the content embeddings
		if self.fusion_strategy == 'concat':
			content_emb = torch.cat((audio_emb, text_emb), dim=-1)
		elif self.fusion_strategy == 'avg':
			content_emb = (audio_emb + text_emb) / 2

		# Projection layer for the content fused embeddings
		content_emb = self.proj_op_fusion_content(content_emb)
		# print("Content emb: {}".format(content_emb.shape))							# BxNx512

		return gesture_emb, content_emb, visual_mask, content_mask, word_boundaries
	
	def forward_validation(self, visual_feats, visual_mask=None, text=None, audio=None, audio_mask=None, word_boundaries=None):

		gesture_emb = self.forward_gestures(visual_feats, visual_mask.unsqueeze(1))
		

		invalid_sample_idx = None
		if text is not None:
			text_emb_roberta, text_mask, text_batch, input_ids, offset_mapping = self.get_roberta_embeddings(text)
			text_emb_subwords = self.forward_text(text_emb_roberta, text_mask.unsqueeze(1).cuda())
			text_emb_words, _, invalid_sample_idx = self.get_word_level_embs(text_emb_subwords, text_batch, input_ids, offset_mapping, audio_emb=None, word_boundaries=None)
			text_attn, content_mask = self.pad_wordlevel_embs(text_emb_words)

		if audio is not None:
			audio_emb_frames = self.forward_audio(audio, audio_mask)
			audio_emb_words, invalid_sample_idx = self.get_audio_word_level_embs(audio_emb_frames, word_boundaries, invalid_sample_idx)
			audio_emb, content_mask = self.pad_wordlevel_embs(audio_emb_words)

		if text is None:
			text_emb = torch.zeros_like(audio_emb)
		elif audio is None:
			audio_emb = torch.zeros_like(text_emb)
						
		if invalid_sample_idx is not None:
			if len(invalid_sample_idx) > 0:
				valid_indices = [i for i in range(gesture_emb.size(0)) if i not in invalid_sample_idx]
				gesture_emb = gesture_emb[valid_indices]
				visual_mask = visual_mask[valid_indices]
				content_mask = content_mask[valid_indices]
				word_boundaries = [word_boundaries[i] for i in valid_indices]

		# print("Gesture emb: {}".format(gesture_emb.shape))							# BxTx512
		# print("Text emb: {}".format(text_emb.shape))									# BxNx256
		# print("Audio emb: {}".format(audio_emb.shape))								# BxNx256
		

		# Fuse the content embeddings
		if self.fusion_strategy == 'concat':
			content_emb = torch.cat((audio_emb, text_emb), dim=-1)
		elif self.fusion_strategy == 'avg':
			content_emb = (audio_emb + text_emb) / 2

		# Projection layer for the content fused embeddings
		content_emb = self.proj_op_fusion_content(content_emb)
		# print("Content emb: {}".format(content_emb.shape))							# BxNx512

		return gesture_emb, content_emb, visual_mask, content_mask, word_boundaries
	
	def forward_inference(self, visual_feats=None, visual_mask=None, text=None, audio=None, audio_mask=None, word_boundaries=None):

		if visual_feats is not None:		
			gesture_attn = self.forward_gestures(visual_feats, visual_mask.unsqueeze(1))
			gesture_attn = self.proj_op_align_gesture(gesture_attn)
			# print("Gesture attn: {}".format(gesture_attn.shape))							# BxTx512
			
			# Return gesture output if content embedding is not required
			if text is None and audio is None:
				return gesture_attn

		if text is not None:
			text_feats, text_mask, text_batch, input_ids, offset_mapping = self.get_roberta_embeddings(text)
			text_attn_subwords = self.forward_text(text_feats, text_mask.unsqueeze(1).cuda())
			word_text_emb, _, _ = self.get_word_level_embs(text_attn_subwords, text_batch, input_ids, offset_mapping, audio_emb=None, word_boundaries=None)
			text_attn, content_mask = self.pad_wordlevel_embs(word_text_emb)
			if audio is None:
				audio_attn = torch.zeros_like(text_attn)
			# print("Text attn: {}".format(text_attn.shape))								# BxNx256

		if audio is not None:
			audio_attn_frames = self.forward_audio(audio, audio_mask.unsqueeze(1))
			word_audio_emb, _ = self.get_audio_word_level_embs(audio_attn_frames, word_boundaries)
			audio_attn, content_mask = self.pad_wordlevel_embs(word_audio_emb)
			if text is None:
				text_attn = torch.zeros_like(audio_attn)				
			# print("Audio attn: {}".format(audio_attn.shape))								# BxNx256
		

		# Fuse the content embeddings
		if self.fusion_strategy == 'concat':
			content_fused_attn = torch.cat((audio_attn, text_attn), dim=-1)
		elif self.fusion_strategy == 'avg':
			content_fused_attn = (audio_attn + text_attn) / 2
		# print("Content fused attn: {}".format(content_fused_attn.shape))				# BxNx512
	
		# Content projection
		content_attn = self.proj_op_fusion_content(content_fused_attn)
		content_attn = self.proj_op_align_content(content_attn)

		if visual_feats is None:
			return content_attn
		else:
			return gesture_attn, content_attn