
# Datasets and Benchmarks

### [1] Gesture Word Spotting: AVS-Spot dataset

AVS-Spot is a benchmark for evaluating the task of **gestured word-spotting**. It contains **500 videos**, sampled from the AVSpeech official test dataset. Each video contains at least one clearly gestured word, annotated as the "target word". Additionally, we provide other annotations, including the text phrase, word boundaries, and speech-stress labels for each sample.

**Task:** Given a target word, an input gesture video with a transcript/speech, the goal is to localize the occurrence of the target word in the video based on gestures.

### [2] Cross-modal Retrieval: AVS-Ret dataset

AVS-Ret is a benchmark for evaluating the task of **cross-modal retrieval**. It comprises **500 videos**, sampled from the AVSpeech official test dataset. Each video contains isolated clean speech along with accurate text transcripts. All the videos are also verified to have reasonable gesture activities.

**Task:** Given a gallery of gesture-speech-text samples, the task is to retrieve a gesture clip given a speech segment and/or text and vice-versa.


### [3] Active Speaker Detection: AVS-Asd dataset
 
AVS-Asd is a benchmark for evaluating the task of **active speaker detection**. It contains **500 videos**, sampled from the AVSpeech official test dataset. For each of these 500 video clips, there are 5 other video clips, chosen from different speakers in the dataset.  Specifically, we create three evaluation subsets, where we choose $P - 1$ clips from different speakers, where $P = 2, 4, 6$. 

**Task:** Given gesture clips of multiple speakers, and a query speech and/or text segment, the goal is to predict the active speaker who is uttering the queried speech/text.
 
## Download instructions 

The files `avs_spot.csv`, `avs_ret.csv` and `avs_asd.csv` are the test datasets for the three downstream tasks. These csv files contain the video-ids along with other annotations. They can easily be read by:

```python
# Switch to the dataset folder
cd dataset

# Load the dataset csv file
import pandas as pd
df_spot = pd.read_csv("avs_spot.csv")	# Gesture Word Spotting dataset
df_ret = pd.read_csv("avs_ret.csv")		# Cross-modal Retrieval dataset
df_asd = pd.read_csv("avs_asd.csv")		# Active Speaker Detection dataset
```

To download and pre-process the videos, run the following commands:

```bash
# Download the videos from YouTube-ids and timestamps
python download_videos.py --file=<avs_spot/avs_ret/avs_asd.csv> --video_root=<dataset-path>

# Obtain the crops with the target speaker (this step will take some time)
python preprocess_videos.py --file=<avs_spot/avs_ret/avs_asd.csv>  --data_root=<dataset-path> --preprocessed_root=<path-to-save-the-preprocessed-data> --merge_dir=<path-to-save-audio-video-merged-results> --temp_dir=<path-to-save-intermediate-results> --metadata_root=<path-to-save-the-metadata>
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
|   ├── *.wav (extracted person track audio for each sample)
```

```
merge_dir (path of the merged videos)
├── *.mp4 (target-speaker videos with audio)
```

### Exploration
  
These datasets are also hosted on 🤗 datasets. The links are shown below:
- [AVS-Spot](https://huggingface.co/datasets/sindhuhegde/avs-spot)
- [AVS-Ret](https://huggingface.co/datasets/sindhuhegde/avs-ret)
- [AVS-Asd](https://huggingface.co/datasets/sindhuhegde/avs-asd)

For further exploration of these datasets, please check the 🤗 pages.