import sys, os, argparse, pickle, subprocess, cv2, math
import numpy as np
import random
from shutil import rmtree, copy

from tqdm import tqdm
from glob import glob

from scipy.interpolate import interp1d
from scipy import signal

from ultralytics import YOLO
from decord import VideoReader
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", message="Feedback manager requires a model with a single signature inference")

parser = argparse.ArgumentParser(description="Code to preprocess the videos and obtain target speaker crops")
parser.add_argument('--file', type=str, required=True, help="Path to the input csv file")
parser.add_argument('--data_root', type=str, required=True, default="", help='Path of the folder containing full uncropped videos')
parser.add_argument('--preprocessed_root', type=str, required=True, help='Path to save the output crops')
parser.add_argument('--merge_dir', type=str, required=True, help='Path to save the audio-video merged output')
parser.add_argument('--temp_dir', type=str, required=True, help='Path to save intermediate results')
parser.add_argument('--metadata_root', type=str, required=True, help='Path to save metadata files')

parser.add_argument('--crop_scale', type=float, default=0, help='Scale bounding box')
parser.add_argument('--min_track', type=int, default=10, help='Minimum facetrack duration')
parser.add_argument('--min_frame_size', type=int, default=64, help='Minimum frame size in pixels')
parser.add_argument('--num_failed_det', type=int, default=25, help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--frame_rate', type=int, default=25, help='Frame rate')
opt = parser.parse_args()

# Set random seeds
random.seed(42) 

# Load the YOLO model
yolo_model = YOLO("yolov9c.pt")

# Initialize the mediapipe holistic keypoint detection model
mp_holistic = mp.solutions.holistic

# Global variables
save_count, err_count = 0, 0

def bb_intersection_over_union(boxA, boxB):

	'''
	This function calculates the intersection over union of two bounding boxes

	Args:
		- boxA (list): Bounding box A.
		- boxB (list): Bounding box B.
	Returns:
		- iou (float): Intersection over union of the two bounding boxes
	'''

	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxB[3], boxB[3])

	interArea = max(0, xB - xA) * max(0, yB - yA)

	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

	try:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	except:
		return None

	return iou

def track_speakers(opt, dets):

	'''
	This function tracks the speakers in the video using the detected bounding boxes

	Args:
		- opt (argparse): Argument parser.
		- dets (list): List of detections.
	Returns:
		- tracks (list): List of tracks.
	'''

	# Minimum IOU between consecutive face detections
	iouThres = 0.5
	tracks = []

	while True:
		track = []
		for frames in dets:
			for face in frames:
				if track == []:
					track.append(face)
					frames.remove(face)
				elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou is None:
						track.append(face)
						frames.remove(face)
						continue
					elif iou > iouThres:
						track.append(face)
						frames.remove(face)
						continue
				else:
					break

		if track == []:
			break
		elif len(track) > opt.min_track:
			framenum = np.array([f['frame'] for f in track])
			bboxes = np.array([np.array(f['bbox']) for f in track])

			frame_i = np.arange(framenum[0], framenum[-1] + 1)

			bboxes_i = []
			for ij in range(0, 4):
				interpfn = interp1d(framenum, bboxes[:, ij])
				bboxes_i.append(interpfn(frame_i))
			bboxes_i = np.stack(bboxes_i, axis=1)

			if max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])) > opt.min_frame_size:
				tracks.append({'frame': frame_i, 'bbox': bboxes_i})

	return tracks


def get_keypoints(frames):

	'''
	This function extracts the keypoints from the frames using MediaPipe Holistic pipeline

	Args:
		- frames (list) : List of frames extracted from the video
	Returns:
		- all_frame_kps (list) : List of keypoints for each frame
	'''

	resolution = frames[0].shape
	# print("Resolution:", resolution)
	
	all_frame_kps= []
	with mp_holistic.Holistic(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	) as holistic:
		for frame in frames:

			results = holistic.process(frame)

			pose, left_hand, right_hand, face = None, None, None, None
			if results.pose_landmarks is not None:
				pose, pose_world, left_hand, right_hand, face = None, None, None, None, None
				if results.pose_landmarks is not None:
					pose = protobuf_to_dict(results.pose_landmarks)['landmark']

				pose_kps = []
				for kps in pose:
					if kps is not None:
						pose_kps.append([kps["x"]*resolution[1], kps["y"]*resolution[0], kps["visibility"]])
				pose_kps = np.array(pose_kps)

				all_frame_kps.append(pose_kps)


	all_frame_kps = np.array(all_frame_kps)
	if len(all_frame_kps) == 0:
		return None
	else:
		return all_frame_kps

def adjust_bbox_kps(frames, all_frame_kps, padding_x=25, padding_y=-15):

	'''
	This function crops frames based on the detected keypoints
	
	Args:
		- frames (list of numpy arrays): List of cropped frames.
		- all_frame_kps (numpy array): T x num_keypoints x 3 (x, y, confidence).
		- padding_x (int): Extra padding for cropping.
		- padding_y (int): Extra padding for cropping.
	
	Returns:
		- cropped_frames (list of numpy arrays): Cropped frames according to adjusted bounding box from keypoints
	'''
	
	LEFT_KP_INDICES = [12, 14, 16, 18, 20, 22, 24]
	RIGHT_KP_INDICES = [11, 13, 15, 17, 19, 21, 23]
	LEFT_HIP_IDX = 23
	RIGHT_HIP_IDX = 24

	left_xs = []
	right_xs = []
	waist_ys = []

	for keypoints in all_frame_kps:
		# Extract left and right keypoints that have confidence > 0.7
		left_kps = [keypoints[i] for i in LEFT_KP_INDICES if keypoints[i][2] > 0.7]
		right_kps = [keypoints[i] for i in RIGHT_KP_INDICES if keypoints[i][2] > 0.7]

		# Compute spatial limits if keypoints exist
		if left_kps:
			left_xs.append(min(kp[0] for kp in left_kps))  # Leftmost x
		if right_kps:
			right_xs.append(max(kp[0] for kp in right_kps))  # Rightmost x

		# Compute waistline from hips if both are detected with confidence > 0.7
		left_hip, right_hip = keypoints[LEFT_HIP_IDX], keypoints[RIGHT_HIP_IDX]
		if left_hip[2] > 0.7 and right_hip[2] > 0.7:
			waist_ys.append((left_hip[1] + right_hip[1]) / 2)

	# Compute global cropping limits
	frame_height, frame_width = frames[0].shape[:2]

	if len(left_xs) > 0 and len(left_xs)/len(all_frame_kps) > 0.7:
		left_x = int(min(left_xs)) - padding_x
	else:
		left_x = 0

	if len(right_xs) > 0 and len(right_xs)/len(all_frame_kps) > 0.7:
		right_x = int(max(right_xs)) + padding_x
	else:
		right_x = frame_width

	if len(waist_ys) > 0 and len(waist_ys)/len(all_frame_kps) > 0.7:
		upper_body_estimate = int(np.mean(waist_ys))  # Use average waist position
		new_y2 = upper_body_estimate + padding_y
	else:
		new_y2 = frame_height

	# Ensure within frame bounds
	left_x = max(0, left_x)
	right_x = min(frame_width, right_x)
	new_y2 = min(new_y2, frame_height)


	# Crop all frames
	cropped_frames = [frame[:new_y2, left_x:right_x] for frame in frames]

	return cropped_frames

def detect_speaker(opt, padding=5, video_fname="person", work_dir=None):

	'''
	This function detects the speaker in the video using YOLOv9 model

	Args:
		- opt (argparse): Argument parser.
		- padding (int): Extra padding for cropping.
		- video_fname (str): Name of the video file.
		- work_dir (str): Directory to save the person.pkl file.
	Returns:
		- alltracks (list): List of tracks.
	'''
	
	videofile = os.path.join(opt.avi_dir, 'video.avi')
	vidObj = cv2.VideoCapture(videofile)

	dets = []
	fidx = 0
	alltracks = []

	while True:
		success, image = vidObj.read()
		if not success:
			break

		image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Perform person detection
		results = yolo_model(image_np, verbose=False)
		detections = results[0].boxes

		dets.append([])
		for i, det in enumerate(detections):
			x1, y1, x2, y2 = det.xyxy[0]
			cls = det.cls[0]
			conf = det.conf[0]  
			if int(cls) == 0 and conf>0.6:  # Class 0 is 'person' in COCO dataset
				x1 = max(0, int(x1) - padding)
				y1 = max(0, int(y1) - padding)
				x2 = min(image_np.shape[1], int(x2) + padding)
				y2 = min(image_np.shape[0], int(y2) + padding)
				dets[-1].append({'frame': fidx, 'bbox': [x1, y1, x2, y2], 'conf': conf})

		fidx += 1


	if len(dets) >= opt.min_track and np.abs(len(dets) - fidx) <= 5:
		alltracks.extend(track_speakers(opt, dets[0:len(dets)]))
	
		savepath = os.path.join(work_dir, 'person.pkl')
		with open(savepath, 'wb') as fil:
			pickle.dump(dets, fil)
		
		return alltracks

	else:
		# print("Num. of frames = {} | Num. of detections = {}".format(fidx, len(dets)))
		return None


def crop_video(opt, track, cropfile, tight_scale=0.9, padding=-15):

	'''
	This function crops the video based on the detected bounding boxes

	Args:
		- opt (argparse): Argument parser.
		- track (dict): Person tracks obtained.
		- cropfile (str): Path to save the cropped video.
		- tight_scale (float): Tight scale for cropping.
		- padding (int): Extra padding for cropping.
	Returns:
		- dets (dict): Detections.
	'''
	
	dets = {'x': [], 'y': [], 's': []}

	for det in track['bbox']:
		# Scale the bounding box by a small factor to obtain a tight crop
		width = (det[2] - det[0]) * tight_scale
		height = (det[3] - det[1]) * tight_scale
		center_x = (det[0] + det[2]) / 2
		center_y = (det[1] + det[3]) / 2

		dets['s'].append(max(height, width) / 2)
		dets['y'].append(center_y)  # crop center y
		dets['x'].append(center_x)  # crop center x

	# Smooth detections
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

	videofile = os.path.join(opt.avi_dir, 'video.avi')
	frame_no_to_start = track['frame'][0]
	
	video_stream = cv2.VideoCapture(videofile)
	video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_no_to_start)
	cropped_frames, all_heights, all_widths = [], [], []

	for fidx, _ in enumerate(track['frame']):
		cs = opt.crop_scale
		bs = dets['s'][fidx]  # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
	
		image = video_stream.read()[1]
		frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))

		my = dets['y'][fidx] + bsi  # BBox center Y
		mx = dets['x'][fidx] + bsi  # BBox center X

		crop = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
		
		cropped_frames.append(crop)
		all_heights.append(crop.shape[0])
		all_widths.append(crop.shape[1])

	# Get the maximum height and width of the cropped frames
	shape = (max(all_heights), max(all_widths))

	# Resize the cropped frames to the maximum height and width
	cropped_frames_fixed_size = []
	for frame in cropped_frames:
		frame = cv2.resize(frame, (shape[0], shape[1]))
		cropped_frames_fixed_size.append(frame)

	# Get the keypoints for the cropped frames
	all_frame_kps = get_keypoints(cropped_frames_fixed_size.copy(), cropfile+"_kps.pkl")
	if all_frame_kps is None:
		upper_body_cropped_frames = cropped_frames_fixed_size
	else:
		upper_body_cropped_frames = adjust_bbox_kps(cropped_frames_fixed_size, all_frame_kps)

	# Write the cropped frames to a video file
	shape_video = (upper_body_cropped_frames[0].shape[1], upper_body_cropped_frames[0].shape[0])
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	vOut = cv2.VideoWriter(cropfile + '.avi', fourcc, opt.frame_rate, shape_video)

	for frame in upper_body_cropped_frames:
		frame = np.uint8(frame)
		vOut.write(frame)

	video_stream.release()
	audiotmp = os.path.join(opt.tmp_dir, 'audio.wav')
	audiostart = (track['frame'][0]) / opt.frame_rate
	audioend = (track['frame'][-1] + 1) / opt.frame_rate

	vOut.release()

	# ========== CROP AUDIO FILE ==========

	command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir, 'audio.wav'), audiostart, audioend, audiotmp))
	output = subprocess.call(command, shell=True, stdout=None)

	copy(audiotmp, cropfile + '.wav')

	# print('Mean pos: x %.2f y %.2f s %.2f' % (np.mean(dets['x']), np.mean(dets['y']), np.mean(dets['s'])))

	return {'track': track, 'proc_track': dets}


def process_video(file, initial_data_path):

	'''
	This function processes the video

	Args:
		- file (str): Path to the video file.
		- initial_data_path (str): Path to save the initial data.
	'''

	global save_count, err_count

	video_name = file.split('/')[-1]
	sd_dest_folder = os.path.join(initial_data_path, video_name[:-4])
	work_dest_folder = os.path.join(opt.metadata_root, video_name[:-4])

	if os.path.exists(os.path.join(work_dest_folder, 'person.pkl')) and os.path.exists(sd_dest_folder):
		save_count += 1
		return


	print("Processing video: ", file)
	setattr(opt, 'videofile', file)

	if os.path.exists(opt.work_dir):
		rmtree(opt.work_dir)

	if os.path.exists(opt.crop_dir):
		rmtree(opt.crop_dir)

	if os.path.exists(opt.avi_dir):
		rmtree(opt.avi_dir)

	if os.path.exists(opt.frames_dir):
		rmtree(opt.frames_dir)

	if os.path.exists(opt.tmp_dir):
		rmtree(opt.tmp_dir)

	os.makedirs(opt.work_dir)
	os.makedirs(opt.crop_dir)
	os.makedirs(opt.avi_dir)
	os.makedirs(opt.frames_dir)
	os.makedirs(opt.tmp_dir)
	os.makedirs(sd_dest_folder, exist_ok=True)
	os.makedirs(work_dest_folder, exist_ok=True)

	# Extract the video and convert it to 25 FPS from the input video file
	command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (opt.videofile,
																os.path.join(opt.avi_dir,
																'video.avi')))
	output = subprocess.call(command, shell=True, stdout=None)
	if output != 0:
		print("Failed to extract video from ", file)
		err_count += 1
		return


	command = ("ffmpeg -hide_banner -loglevel panic -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,
																		 'video.avi'),
																		 os.path.join(opt.avi_dir,
																		'audio.wav')))
	output = subprocess.call(command, shell=True, stdout=None)
	if output != 0:
		print("Failed to extract audio from ", file)
		err_count += 1
		return

	# Detect the speaker in the video using YOLOv9 model
	spk_tracks = detect_speaker(opt, video_fname=video_name[:-4], work_dir=work_dest_folder)
	if spk_tracks is None:
		print("No tracks found for ", file)
		err_count += 1
		return


	# Crop the video based on the detected bounding boxes
	vidtracks = []
	for ii, track in enumerate(spk_tracks):
		vidtracks.append(crop_video(opt, track, os.path.join(sd_dest_folder, '%05d' % ii)))

	savepath = os.path.join(work_dest_folder, 'tracks.pkl')
	with open(savepath, 'wb') as fil:
		pickle.dump(vidtracks, fil)

	# Clean up the temporary directories
	rmtree(opt.tmp_dir)
	rmtree(opt.avi_dir)
	rmtree(opt.frames_dir)

	print("Saved processed video...")
	save_count += 1


def filter_and_merge(csv_path, initial_data_path, final_data_path, merged_path, fps=25):

	'''
	This function filters and merges the audio and video

	Args:
		- csv_path (str): Path to the dataset csv file.
		- initial_data_path (str): Path to the initial data.
		- final_data_path (str): Path to save the final data (saves .avi and .wav files).
		- merged_path (str): Path to save the merged data (saves .mp4 files).
		- fps (int): Frames per second.
	'''
	
    # Load the dataset csv file with annotations
    df = pd.read_csv(csv_path)

	print("Merging audio and video for visualization...")

	for idx in tqdm(range(len(df))):
		row = df.iloc[idx]
		video = os.path.join(initial_data_path, row["filename"] + ".avi")
		audio = video.replace(".avi", ".wav")
		kps = video.replace(".avi", "_kps.pkl")

		video_output = video.replace(initial_data_path, final_data_path)
		audio_output = audio.replace(initial_data_path, final_data_path)
		fname = video.split("/")[-2] + "_" + video.split("/")[-1].split(".")[0]
		merged_output = os.path.join(merged_path, str(idx)+"_"+fname+".mp4")

		# Merge the audio and video
		status = subprocess.call('ffmpeg -hide_banner -loglevel panic -threads 1 -y -i %s -i %s -strict -2 -q:v 1 %s' %
					(audio, video, merged_output), shell=True)
		if status != 0:
			print("Failed to merge audio and video for ", video)
			continue

		# Copy the target video to the final data path
		video_output_folder = os.path.join(final_data_path, video_output.split("/")[-2])
		if not os.path.exists(video_output_folder):
			os.makedirs(video_output_folder)
		status = subprocess.call('rsync -az %s %s' % (video, video_output), shell=True)
		if status != 0:
			print("Failed to copy video for ", video)
			continue

		# Copy the target audio to the final data path
		status = subprocess.call('rsync -az %s %s' % (audio, audio_output), shell=True)
		if status != 0:
			print("Failed to copy audio for ", audio)
			continue


if __name__ == "__main__":

	# Get the list of videos to process
	filelist = pd.read_csv(opt.file)["filename"].tolist()
	files = [os.path.join(opt.data_root, f.split("/")[0] + ".mp4") for f in filelist]
	print(f"A total of {len(files)} files found.")

	# Create the necessary directories
	initial_data_path = opt.preprocessed_root+"_all"
	os.makedirs(initial_data_path, exist_ok=True)
	os.makedirs(opt.preprocessed_root, exist_ok=True)
	os.makedirs(opt.metadata_root, exist_ok=True)

	# Set the necessary attributes
	setattr(opt, 'avi_dir', os.path.join(opt.temp_dir, 'pyavi'))
	setattr(opt, 'tmp_dir', os.path.join(opt.temp_dir, 'pytmp'))
	setattr(opt, 'work_dir', os.path.join(opt.temp_dir, 'pywork'))
	setattr(opt, 'crop_dir', os.path.join(opt.temp_dir, 'pycrop'))
	setattr(opt, 'frames_dir', os.path.join(opt.temp_dir, 'pyframes'))
	setattr(opt, 'audio_dir', None)


	# Process the videos
	prog_bar = tqdm(files)
	for file in prog_bar:
		process_video(file, initial_data_path)
		prog_bar.set_description(f"Saved files: {save_count} | Failed files: {err_count}")

    # Create the directory and merge the audio-video files
    os.makedirs(opt.merge_dir, exist_ok=True)
    filter_and_merge(opt.file, initial_data_path, opt.preprocessed_root, opt.merge_dir)