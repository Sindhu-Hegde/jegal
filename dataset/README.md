# Dataset: AVS-Spot

AVS-Spot is a benchmark for evaluating the task of **gestured word-spotting**. It contains **500 videos**, sampled from the AVSpeech official test dataset. Each video contains at least one clearly gestured word, annotated as the "target word". Additionally, we provide other annotations, including the text phrase, word boundaries, and speech-stress labels for each sample.

**Task:** Given a target word, an input gesture video with a transcript/speech, the goal is to localize the occurrence of the target word in the video based on gestures.

## Download instructions

`avs_spot.csv` contains the video ids along with other annotations. They can easily be read by:

```python
# Switch to the dataset folder
cd dataset

# Load the dataset csv file
import pandas as pd
df = pd.read_csv("avs_spot.csv")
```

To download and pre-process the videos, run the following commands:

```bash
# Download the videos from YouTube-ids and timestamps
python download_videos.py --file=avs_spot.csv --video_root=<dataset-path>

# Obtain the crops with the target speaker (this step will take some time)
python preprocess_videos.py --file=avs_spot.csv --data_root=<dataset-path> --preprocessed_root=<path-to-save-the-preprocessed-data> --merge_dir=<path-to-save-audio-video-merged-results> --temp_dir=<path-to-save-intermediate-results> --metadata_root=<path-to-save-the-metadata>
```

Once the dataset is downloaded and pre-processed, the structure of the folders will be as follows:

```
video_root (path of the downloaded videos) 
├── *.mp4 (videos)
```
    
```
preprocessed_root (path of the pre-processed videos) 
├── list of video-ids
│   ├── *.avi (extracted person track video for each sample)
|	├── *.wav (extracted person track audio for each sample)
```

```
merge_dir (path of the merged videos) 
├── *.mp4 (target-speaker videos with audio)
```


AVS-Spot dataset is also hosted on [🤗](https://huggingface.co/datasets/sindhuhegde/avs-spot). For further exploration, please check the 🤗 page.