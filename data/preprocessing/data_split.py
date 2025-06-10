import pandas as pd
import random
import os
import natsort
import shutil
import json

random.seed(123)

def split_dict():
    df = pd.read_csv('utterance_speakerID.csv')

    dia_list = list(set(df['Dialogue_ID']))

    train_list = []
    valid_list = []
    test_list = []
    for dia in dia_list:
        temp = random.random()
        if temp < 0.8:
            train_list.append(dia)
        elif temp > 0.9:
            valid_list.append(dia)
        else:
            test_list.append(dia)
            
    print(train_list)
    print(len(train_list))
    print(valid_list)
    print(len(valid_list))
    print(test_list)
    print(len(test_list))

    train_dictionary = {train_list[i] : i  for i in range(len(train_list))}
    valid_dictionary = {valid_list[i] : i for i in range(len(valid_list))}
    test_dictionary = {test_list[i] : i for i in range(len(test_list))}

    print(train_dictionary)
    print(valid_dictionary)
    print(test_dictionary)

    with open("train_dictionary.json", 'w') as f:
        json.dump(train_dictionary, f)
        
    with open("valid_dictionary.json", 'w') as ff:
        json.dump(valid_dictionary, ff)
        
    with open("test_dictionary.json", 'w') as fff:
        json.dump(test_dictionary, fff)
        
        

def data_split(mode='train'):
    print(f'{mode} data split begin')
    os.makedirs(os.path.join(f'../MSC/{mode}','video'), exist_ok=True)
    os.makedirs(os.path.join(f'../MSC/{mode}','audio'), exist_ok=True)
    
    with open(f"{mode}_dictionary.json", "r") as st_json:
        dic = json.load(st_json)
        dic = {int(k):v for k,v in dic.items()}  # string to integer
        print(dic)
    dia_from = list(dic.keys())
    dia_to = list(dic.values())
        
    # text
    print(f'{mode} text data split begin')
    text_df = pd.read_csv('utterance_speakerID.csv')
    text_sample_df = text_df[text_df['Dialogue_ID'].isin(dia_from)]  # extract 'mode' sample
    text_sample_aligned_df = text_sample_df.replace({'Dialogue_ID':dic})  # sequential dialogue number
    text_sample_aligned_df['Utterance_ID'] = text_sample_aligned_df.groupby('Dialogue_ID').cumcount()  # sequential utterance number
    text_sample_aligned_df.to_csv(f'../MSC/{mode}/text_data.csv', index=False)
    
    
    # audio
    print(f'{mode} audio data split begin')
    audio_list = natsort.natsorted(os.listdir('./utt_audio'))
    # print("audio_list: ", audio_list)
    pre_dia_num = dia_from[0]
    sequential_utt_num = 0
    for audio in audio_list:
        dia_num = int(audio.split('_')[0][3:])
        
        if pre_dia_num != dia_num:
            pre_dia_num = dia_num
            sequential_utt_num = 0
            
        if dia_num in dia_from:
            from_file_path = f'./utt_audio/{audio}' # extract 'mode' sample
            to_file_path = f'../MSC/{mode}/audio/dia{dic[dia_num]}_utt{sequential_utt_num}.wav' # sequential dialogue utterance number
            shutil.copyfile(from_file_path, to_file_path)
            
        sequential_utt_num += 1

        
    # video
    print(f'{mode} video data split begin')
    video_list = natsort.natsorted(os.listdir('./utt_video'))
    # print("video_list:", video_list)
    pre_dia_num = dia_from[0]
    sequential_utt_num = 0
    for video in video_list:
        dia_num = int(video.split('_')[0][3:])
        
        if pre_dia_num != dia_num:
            pre_dia_num = dia_num
            sequential_utt_num = 0
            
        if dia_num in dia_from:
            from_file_path = f'./utt_video/{video}'
            to_file_path = f'../MSC/{mode}/video/dia{dic[dia_num]}_utt{sequential_utt_num}.mp4'
            shutil.copyfile(from_file_path, to_file_path)
            
        sequential_utt_num += 1
        
        
if __name__=='__main__':
    # split_dict()
    data_split(mode='train')
    data_split(mode='valid')
    data_split(mode='test')