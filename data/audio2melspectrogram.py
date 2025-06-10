import torch
import os
import torchaudio
import torchaudio.transforms as T
import natsort

torch.random.manual_seed(0)
spectrogram = T.MelSpectrogram(n_fft=1024)

def a2s(audio_path):
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(audio_path)
    spec = spectrogram(SPEECH_WAVEFORM)
    return spec

def a2mel(data_name):
    for mode in ['train', 'test', 'valid']:
        file_root = f"./{data_name}/{mode}"
        save_path = f'{file_root}/melspectrogram'
        
        os.makedirs(save_path, exist_ok=True)
    
        for audio_file in natsort.natsorted(os.listdir(f"{file_root}/audio")):
            try:
                audio_name = os.path.basename(audio_file).split('.wav')[0]    
                audio_feature = a2s(f"{file_root}/audio/{audio_file}")
                
                torch.save(audio_feature, f"{save_path}/{audio_name}.pt")
            except:
                print(f"This file occurs ERROR =====>> {file_root}/audio/{audio_file} \n")

if __name__ == "__main__":
    # a2mel('MELD')
    a2mel('MSC')