import os
import argparse
from glob import glob
from tqdm import tqdm
import whisperx

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True, help="Path to the folder containing audio files")
parser.add_argument('--result_dir', type=str, required=True, help="Path to the folder to save the transcripts")

parser.add_argument('--model_type', type=str, default="large-v3")
parser.add_argument('--language', default="en")
parser.add_argument('--batch_size', type=int, default=2)

args = parser.parse_args()

device = "cpu" 
compute_type = "float32"

def get_predictions(filelist, batch_size=8, lang_dict=None):

	if not os.path.exists(args.result_dir): 
		os.mkdir(args.result_dir)	
	
	for audio_file in tqdm(filelist):

		try:
			fname = os.path.join(audio_file.split("/")[-2], audio_file.split("/")[-1].split(".")[0])
			output_fname_pred = os.path.join(args.result_dir, fname+".txt")
			# print("Output file: ", output_fname_pred)

			if os.path.exists(output_fname_pred):
				continue

			if lang_dict is None:
				lang = args.language
			else:
				lang = lang_dict[audio_file]
			audio = whisperx.load_audio(audio_file)
			result = model.transcribe(audio, batch_size=batch_size, language=lang)

			folder = os.path.join(args.result_dir, audio_file.split("/")[-2]) 
			if not os.path.exists(folder): 
				os.mkdir(folder)

			f = open(output_fname_pred, "w")
			f.write("Text: ")
			for idx in range(len(result["segments"])):
				f.write(result["segments"][idx]["text"])
			f.write("\nLang: " + lang)

			model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
			result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)


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

		except:
			continue
		
	

if __name__ == '__main__':

	filelist = glob("{}/*.wav".format(args.path))
	print("Total files: ", len(filelist))

	model = whisperx.load_model(args.model_type, device, compute_type=compute_type)
	print("Loaded the WhisperX model")

	get_predictions(filelist, batch_size=args.batch_size, lang_dict=None)