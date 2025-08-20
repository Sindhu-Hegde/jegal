import argparse
import pandas as pd
import os, subprocess
from tqdm import tqdm
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["LC_ALL"]="en_US.utf-8"
os.environ["LANG"]="en_US.utf-8"

parser = argparse.ArgumentParser(description="Code to download the videos from the input csv file")
parser.add_argument('--file', required=False, default="avs_spot.csv")
parser.add_argument('--video_root', type=str, default="videos")
args = parser.parse_args()

def is_valid_video(file_path):
	
	'''
	This function validates the video file using the following checks:
		(i) Check if duration > 0
		(ii) Check if audio is present
	
	Args:
		- file_path (str): Path to the video file.
	Returns:
		- True if the video is valid, False otherwise.
	'''

	if not os.path.exists(file_path):
		return False  # File does not exist

	# Use ffmpeg to get duration
	try:
		result = subprocess.run(
			["ffmpeg", "-i", file_path, "-f", "null", "-"],
			stdout=subprocess.DEVNULL, 
			stderr=subprocess.DEVNULL  
		)
		valid = result.returncode == 0  # If return code is 0, it's a valid video
		if not valid:
			print(f"Invalid video (ffmpeg failed): {file_path}")
			os.remove(file_path)
			return False
	except Exception:
		return False  # If ffmpeg fails, assume it's invalid

	return True


def mp_handler(i, df, result_dir):

	'''
	This function handles the multiprocessing of the video download

	Args:
		- i (int): Index of the video.
		- df (pd.DataFrame): DataFrame containing the video information.
		- result_dir (str): Directory to save the video.
	'''

	cookies_files = "cookies.txt"

	try:
		data = df.iloc[i]
		vid = data['video_id']
		video_link = "https://www.youtube.com/watch?v={}".format(vid)
		start = data['start_time']
		end = data['end_time']
		start = format(float(start), '.6f')
		end = format(float(end), '.6f')
		time = "*{}-{}".format(start,end)
		# print(vid, video_link, start, end, time)

		output_fname = os.path.join(result_dir, "{}_{}-{}.mp4".format(vid, start, end))
	
		if os.path.exists(output_fname):
			# Validate file
			# if is_valid_video(output_fname):
			# 	return
			return

		# Download the video
		# cmd = "yt-dlp --geo-bypass --download-sections {} --format=mp4 -o {} {}".format(time, output_fname, video_link)
		cmd = "yt-dlp --no-cache-dir --cookies {} --geo-bypass --download-sections {} --format=mp4 -o {} {}".format(cookies_files, time, output_fname, video_link)
		subprocess.call(cmd, shell=True)

		# Validate file and delete if invalid
		if not is_valid_video(output_fname):
			print(f"Invalid file detected: {output_fname}. Deleting...")
			# os.remove(output_fname)

	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

def download_data(args):

	'''
	This function downloads the videos from the given csv file

	Args:
		- args (argparse.Namespace): Arguments.
	'''

	# Read the csv file
	df = pd.read_csv(args.file)[::-1]
	print("Total files: ", len(df))

	# Create the result directory
	if not os.path.exists(args.video_root):
		os.makedirs(args.video_root)

	# Create the multiprocessing pool and submit the jobs to download the videos
	jobs = [idx for idx in range(len(df))]
	p = ThreadPoolExecutor(8)
	futures = [p.submit(mp_handler, j, df, args.video_root) for j in jobs]
	res = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

if __name__ == '__main__':

	# Download the videos
	download_data(args)