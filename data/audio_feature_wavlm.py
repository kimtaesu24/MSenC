import torch
import torchaudio
import os
import natsort

from WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load('./WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()


def wavlm_feature(SPEECH_FILE):    
    wav, sr = torchaudio.load(SPEECH_FILE)
    wav_input_16khz = torchaudio.functional.resample(wav, sr, 16000)
    
    # wav_input_16khz = torch.randn(1,10000)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    rep = model.extract_features(wav_input_16khz)[0]

    return rep

def get_audio_feature(data_name):
    print(data_name)
    for mode in ['test']:
        print(mode)
        file_root = f"./{data_name}/{mode}"
        
        save_path = f'{file_root}/audio_feature'
        os.makedirs(save_path, exist_ok=True)
        
        os.makedirs(f"./{data_name}/error", exist_ok=True)
        
        for audio in natsort.natsorted(os.listdir(f'./{file_root}/audio/')):
            try:
                SPEECH_FILE = f"{file_root}/audio/{audio}"
                audio_feature = wavlm_feature(SPEECH_FILE)
                
                audio_name = os.path.basename(audio).split('.wav')[0]
                torch.save(audio_feature, f'./{save_path}/{audio_name}.pt')
            except:
                with open(f"./{data_name}/error/{mode}_audio_feature_error.txt",'a') as f:
                    f.write(f"This file occurs ERROR =====>> {file_root}/audio/{audio} \n")
                f.close()


if __name__ == "__main__":
    get_audio_feature('MELD')
    # get_audio_feature('MSC')
    '''
    command : PYTHONPATH=unilm/wavlm python audio_feature_wavlm.py 
    '''
