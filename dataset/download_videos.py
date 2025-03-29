import argparse
import pandas as pd
import numpy as np
import os, subprocess
from tqdm import tqdm
import math

import traceback
import executor
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

os.environ["LC_ALL"]="en_US.utf-8"
os.environ["LANG"]="en_US.utf-8"

parser = argparse.ArgumentParser(description="Code to download the videos from the input csv file")
parser.add_argument('--file', type=str, required=True, help="Path to the input csv file")
parser.add_argument('--video_root', type=str, required=True, help="Path to the directory to save the videos")

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
        return result.returncode == 0  # If return code is 0, it's a valid video
    except Exception:
        return False  # If ffmpeg fails, assume it's invalid

	# Check if audio exists
    cmd_audio = f'ffmpeg -i "{video_path}" -map 0:a:0 -f null - 2>&1 | grep "Output"'
    try:
        audio_output = subprocess.check_output(cmd_audio, shell=True, text=True)
        if not audio_output.strip():
            print(f"Invalid video (no audio): {video_path}")
            os.remove(video_path)
            return False
    except Exception:
        print(f"Error checking audio: {video_path}")
        os.remove(video_path)
        return False


def mp_handler(i, df, result_dir):

	'''
	This function handles the multiprocessing of the video download

	Args:
		- i (int): Index of the video.
		- df (pd.DataFrame): DataFrame containing the video information.
		- result_dir (str): Directory to save the video.
	'''

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
			if is_valid_video(output_fname):
				return

		# Download the video
		cmd = "yt-dlp --geo-bypass -f b --download-sections {} --format=mp4 -o {} {}".format(time, output_fname, video_link)
		subprocess.call(cmd, shell=True)

		# Validate file and delete if invalid
		if not is_valid_video(output_fname):
			print(f"Invalid file detected: {output_fname}. Deleting...")
			os.remove(output_fname)

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

    # Load the dataset csv file with annotations
    df = pd.read_csv(args.file)
	print("Total files: ", len(df))

	# Create the result directory
	if not os.path.exists(args.result_dir):
		os.makedirs(args.result_dir)

	# Create the multiprocessing pool and submit the jobs to download the videos
	jobs = [idx for idx in range(len(df))]
	p = ThreadPoolExecutor(8)
	futures = [p.submit(mp_handler, j, df, args.video_root) for j in jobs]
	res = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

if __name__ == '__main__':

	# Download the videos
	download_data(args)