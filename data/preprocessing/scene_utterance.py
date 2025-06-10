import torch
import time
import os
import numpy as np
import moviepy.editor as mp
import pandas as pd
import natsort

from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import video2audio_save
from dialogue_scene import split_video
from diarization import speaker_diarization

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
# model_id = "openai/whisper-medium.en"
model_id = "distil-whisper/distil-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=25,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

video_load_path = './scene_video'
audio_save_path = './scene_audio'

video_save_path = './utt_video'
temp_video_save_path = './sd_video'
temp_audio_save_path = './sd_audio'

os.makedirs(audio_save_path, exist_ok=True)
os.makedirs(video_save_path, exist_ok=True)
os.makedirs(temp_video_save_path, exist_ok=True)
os.makedirs(temp_audio_save_path, exist_ok=True)

diaID_list = []
uttID_list = []
utterance_list = []
timestamp_list = []

utt_idx = 0
pre_dia_idx = 0

def get_video_duration(file_path):
    clip = VideoFileClip(file_path)
    duration = clip.duration
    clip.close()
    return duration

def whisper_after_diarization(audio, video_load_path, video_name):
    global utt_idx
    global pre_dia_idx
    
    global diaID_list
    global uttID_list
    global utterance_list
    global timestamp_list
    
    sd_timestamp = speaker_diarization(audio)
    
    
    dia_idx = video_name.split("-")[0].split('dia')[1]
    
    for j, time in enumerate(sd_timestamp):
        input_video = os.path.join(video_load_path, f'{video_name}.mp4')
        output_video = os.path.join(temp_video_save_path, f'{video_name}:{j}.mp4')
        # print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        #     i+1,
        #     scene[0].get_timecode(), scene[0].get_frames(),
        #     scene[1].get_timecode(), scene[1].get_frames(),))
        
        if time[1] == None:  # end of video but dead
            time[1] = get_video_duration(input_video)
            
        split_video(input_video, output_video, time[0], time[1])
    
    for k in range(len(sd_timestamp)):
        try:
            video2audio_save(load_path=temp_video_save_path,
                            video=f'{video_name}:{k}.mp4',
                            save_path=temp_audio_save_path)
        except:  # diarization split too short [KeyError:video_fps]
            with open('error.txt', 'a') as f:
                f.write(f'{video_name}:{k}.mp4\n')
            f.close()
            continue
        
        audio = os.path.join(temp_audio_save_path, f'{video_name}:{k}.wav')
        
        # result = pipe(audio, generate_kwargs={"language": "english"}, return_timestamps=True)
        result = pipe(audio, return_timestamps=True)
        
        for data in result['chunks']:        
            split_video(input_file=os.path.join(temp_video_save_path, f'{video_name}:{k}.mp4'),
                        output_file=os.path.join(video_save_path, f"dia{dia_idx}_utt{utt_idx}.mp4"),
                        start_time=data['timestamp'][0],
                        end_time=data['timestamp'][1])
            
            diaID_list.append(dia_idx)
            uttID_list.append(utt_idx)
            utterance_list.append(data['text'])
            timestamp_list.append(data['timestamp'])
            
            utt_idx += 1

        pre_dia_idx = dia_idx
    
        
    df = pd.DataFrame({'Dialogue_ID': diaID_list,
                        'Utterance_ID': uttID_list,
                        'Utterance': utterance_list,
                        'timestamp': timestamp_list})
    try:
        total_df = pd.read_csv('utterance.csv')
        total_df = pd.concat([total_df, df])
        total_df.to_csv('utterance.csv', index=False)
    except FileNotFoundError:
        df.to_csv('utterance.csv', index=False)
    diaID_list = []
    uttID_list = []
    utterance_list = []
    timestamp_list = []
    
    
scene_videos = natsort.natsorted(os.listdir(video_load_path))

for i, sample in enumerate(scene_videos):
    # if sample == 'dia684-0.mp4':
    #     print(i)
    #     exit()
    # else:
    #     continue
    
    # if i < 8791:
    #     continue
    # sample = './dia_audio/dia1000.wav'
    print(sample)
    
    
    try:
        video2audio_save(load_path=video_load_path,
                        video=sample,
                        save_path=audio_save_path)
    except:  # scene split too short [KeyError:video_fps]
        with open('error.txt', 'a') as f:
            f.write(f'{sample}\n')
        f.close()
        continue
    
    start = time.time()
    video_name = sample.split('.mp4')[0]
    audio = os.path.join(audio_save_path, f'{video_name}.wav')
    
    dia_idx = video_name.split("-")[0].split('dia')[1]
    
    if pre_dia_idx != dia_idx:
        utt_idx = 0


    # get duration whether give to whisper directly or not.
    duration = get_video_duration(os.path.join(video_load_path, sample))  
    
    if duration < 25:
        # result = pipe(audio, generate_kwargs={"language": "english"}, return_timestamps=True)
        result = pipe(audio, return_timestamps=True)
        # result = pipe(audio, return_timestamps=True)
        # print(result['chunks'])
        # print('time spent:{:.02f}'.format(time.time() - start))
    else:
        print(f"video duration is over 25 second at {sample}")
        whisper_after_diarization(audio, video_load_path, video_name)
        continue
    
    for data in result['chunks']:
        # print(data)
        # print("timestamp:", data['timestamp'])
        # print("text:", data['text'])
        
        start_time = data['timestamp'][0]
        end_time = data['timestamp'][1]
        
        if end_time == None:  # end of video but dead
            end_time = duration
            
        split_video(input_file=os.path.join(video_load_path, sample),
                    output_file=os.path.join(video_save_path, f"dia{dia_idx}_utt{utt_idx}.mp4"),
                    start_time=start_time,
                    end_time=end_time)
        
        diaID_list.append(dia_idx)
        uttID_list.append(utt_idx)
        utterance_list.append(data['text'])
        timestamp_list.append(data['timestamp'])
        
        utt_idx += 1
        
    pre_dia_idx = dia_idx
    

    df = pd.DataFrame({'Dialogue_ID': diaID_list,
                        'Utterance_ID': uttID_list,
                        'Utterance': utterance_list,
                        'timestamp': timestamp_list})
    try:
        total_df = pd.read_csv('utterance.csv')
        total_df = pd.concat([total_df, df])
        total_df.to_csv('utterance.csv', index=False)
    except FileNotFoundError:
        df.to_csv('utterance.csv', index=False)
    diaID_list = []
    uttID_list = []
    utterance_list = []
    timestamp_list = []
    
