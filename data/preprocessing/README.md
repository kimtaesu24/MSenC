# English Conversation 

## Preprocessing

### 1. Download Raw video
- You can download raw video from below link:
    - link: https://www.youtube.com/playlistlist=PLzVm1SmjPKc_OnC56MbILOmdXvGV_3kE9
    - video set: 1~12, 14~57, 66~81
- Save raw video under `./raw_video`. The directory structure has to:

```
english_conversation
    ├── README.md
    ├── requirments.txt
    ├── ...
    └── raw_video
        ├── English Conversation 01.mp4
        ├── ...
```

### 2. Split Raw video according to Dialogue
- Split the raw videos to dialogue unit. You can use `timestamp.json` which save start-end seconds of each dialogue.
- With the below command, the dialogue videos will saved at `./dia_video`.

```
python timestamp_dialogue.py
```

### 3. Split Dialogue video to Scene video
- Because Whisper model was train to predict up to 30 second audio, Whisper model used to fail to record timestamp by utterance over 30 second. So we split video with visual information. That is "When speaeker talking, scene would not change."
- Because Whisper model cannot record timestamp by utterance over 30 second, we conduct scene detection.
- With the below command, the scene video will saved at `./scene_video`.

```
python dialogue_scene.py
```

### 4. Split Scene video to Utterance
- Split the dialogue videos to utterance unit. 
- With the below command, the dialogue csv file will saved as `utterance.csv`. We utilize '[distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)' model which train openAI Whisper-large-v3 with english only data and show more accurate and fast inference speed. (recommend using GPU)
- Even though we utilize scene detection, there is long duration video existing for exmaple phone call scene with out opponent (only with voice). So we decide to utlize speaker diarization just for long duration video. We utilize '[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)' model.
    - The reason why we do not use spearker diarization from first is speaker diarization split conversation too much including interjection. This may occure confusion during training phase.
- With the below command, the dialogue videos will saved at `./utterance`.

```
python scene_utterance.py
```

### 5. Assign Speaker ID on Utterance video
- we assign speaker ID for all video clip. For this, we get speaker embedding using [wespeaker](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) and cluster them with HDBSCAN algorithm (cosine distance) along to dialogue.
- With the below command, the speakerID will append at `utterance.csv` file.

```
python speakerID.py
```

### 6. Train, Validation, Test set split
- With the below command, utterance video will distributed into `./train`, `./valid`, `./test` directory.

```
python data_split.py
```

### Additional) remove file
- After process the video, you can remove redundant files.

```
rm -r raw_video
rm -r raw_audio

rm -r dia_video
rm -r dia_audio

rm -r scene_video
rm -r scene_audio

rm -r sd_video
rm -r sd_audio
```
