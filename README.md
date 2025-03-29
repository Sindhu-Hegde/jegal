# JEGAL: ***J***oint ***E***mbedding for ***G***estures, ***A***udio, and ***L***anguage

This code is for our paper titled: **Understanding Co-speech Gestures in-the-wild**.<br />
**Authors**: [Sindhu Hegde](https://sindhu-hegde.github.io), [K R Prajwal](https://www.robots.ox.ac.uk/~prajwal/), [Taein Kwon](https://taeinkwon.com/), [Andrew Zisserman](https://scholar.google.com/citations?hl=en&user=UZ5wscMAAAAJ) 

|   📝 Paper   |   📑 Project Page    |  📦 AVS-Spot Dataset | 🛠 Demo  | 
|:-----------:|:-------------------:|:------------------:|:------------------:|
| [Paper]() | [Website](https://www.robots.ox.ac.uk/~vgg/research/jegal/) | [Dataset](https://huggingface.co/datasets/sindhuhegde/avs-spot) | Coming soon | 
<br />

<p align="center">
    <img src="assets/teaser.gif", width="450"/>
</p>

We present **JEGAL**, a Joint Embedding space for Gestures, Audio and Language. Our semantic gesture representations can be used to perform multiple downstream tasks such as cross-modal retrieval, spotting gestured words, and identifying who is speaking solely using gestures.

## News 🚀🚀🚀

- **[2025.03.29]** 🔥 Our new gesture-spotting dataset: **AVS-Spot** has been released!


## Dataset

```bash
# Switch to the dataset folder
cd dataset

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



## Updates

Thank you for visiting, we appreciate your interest in our work! We plan to release the inference script along with the trained models soon, likely within the next few weeks. Until then, stay tuned and watch the repository for updates.
