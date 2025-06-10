import torch
import random
import copy

from torch.utils.data import Dataset
from . import modules

class Conversation_Dataset(Dataset):
    def __init__(self, dataset, args):
        self.device = args.device
        
        self.dataset = dataset
        self.pre_dia_idx = list(set(self.dataset['Dialogue_ID']))
        
        self.max_history = args.max_history
        self.audio_type = args.audio_type
        self.audio_pad_size = args.audio_pad_size
        self.video_pad_size = args.video_pad_size
        self.target = args.target
        self.fps = args.fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dia = self.dataset.iloc[idx]['Dialogue_ID']
        utt = self.dataset.iloc[idx]['Utterance_ID']
        # print(f'dia{dia}_utt{utt}')
        
        ######################################################
        # text histoy
        text_history = self.dataset[(self.dataset['Dialogue_ID']==int(dia)) & (self.dataset['Utterance_ID']<=int(utt))]['Utterance']  # caution on shuffle
        text_history = text_history.to_list()
        
        
        # audio history
        if self.audio_type == 'wavlm':
            audio_paths = self.dataset[(self.dataset['Dialogue_ID']==int(dia)) & (self.dataset['Utterance_ID']<=int(utt))]['audio_feature_path']
        elif self.audio_type == 'spectrogram':
            audio_paths = self.dataset[(self.dataset['Dialogue_ID']==int(dia)) & (self.dataset['Utterance_ID']<=int(utt))]['melspectrogram_path']
        audio_paths = audio_paths.to_list()
        
        audio_history = audio_process(audio_paths, self.audio_type, self.audio_pad_size)
        
        # video history
        video_paths = self.dataset[(self.dataset['Dialogue_ID']==int(dia)) & (self.dataset['Utterance_ID']<=int(utt))]['video_feature_path']
        video_paths = video_paths.to_list()
        
        video_history = video_process(video_paths, self.fps, self.video_pad_size)
        
        ######################################################
        # align the input number for each modalities
        text_history, audio_history, video_history, num_hist = multimodal_align(text_history, audio_history, video_history)
        
        assert torch.sum(video_history['mask']) != 0 and torch.sum(audio_history['mask']) != 0, f'too long video at dia{dia}_utt{utt}'
        
        ######################################################
        # speaker history
        speaker = self.dataset[(self.dataset['Dialogue_ID']==int(dia)) & (self.dataset['Utterance_ID']<=int(utt)+1)]['Speaker_ID']  # for prompt of R(response)
        speaker = speaker.to_list()
        speaker_history = speaker[-(num_hist+1):]  # for prompt of R(response)
        
        ######################################################
        # label
        label_text = self.dataset.iloc[idx]['label_Utterance']
        label_audio = self.dataset.iloc[idx]['label_audio_path']
        if 'description' in self.target:
            label_description = f"({self.dataset.iloc[idx]['label_description']})"
        else:
            label_description = ""
            
        ######################################################
        with torch.no_grad():
            inputs = [text_history,
                    audio_history,
                    video_history,
                    speaker_history,
                    ]

            labels = [label_text,
                    label_audio,  # stereo -> mono,
                    label_description,
                    # label_speaker
                    ]
        identifier = f'dia{dia}_utt{utt}'
        
        return inputs, labels, identifier
    
    def shuffle(self):
        post_dia_list = copy.deepcopy(self.pre_dia_idx)
        random.shuffle(post_dia_list)
        
        idx_map = dict(zip(self.pre_dia_idx, post_dia_list))
        
        self.dataset = self.dataset.replace({"Dialogue_ID": idx_map})
        self.dataset = self.dataset.sort_values(by=['Dialogue_ID', 'Utterance_ID'])
        
        
def audio_process(audio_paths, audio_type, audio_pad_size):
    audio_feature_list = []
    audio_mask_list = []
    for i, audio_path in enumerate(audio_paths):
        audio_feature = torch.load(audio_path).detach()
        if audio_type == 'spectrogram':
            audio_feature = torch.transpose(audio_feature[0], 0, 1)
        if audio_type == 'wavlm':
            audio_feature = audio_feature[0] # stereo -> mono
        
        audio_feature_list.append(audio_feature)
        audio_mask_list.append(torch.ones([audio_feature.shape[0]], dtype=torch.int8)*(i+1))
    audio_history = {'feature': modules.pad(torch.cat(audio_feature_list, dim=0), padding_size=audio_pad_size),
                    'mask': modules.pad(torch.cat(audio_mask_list, dim=0), padding_size=audio_pad_size)
                    }
    
    return audio_history
            
def video_process(video_paths, fps, video_pad_size):
    video_feature_list = []
    video_mask_list = []
    for i, video_path in enumerate(video_paths):
        video_feature = torch.load(video_path).detach()  # ([seq_len, hidden_dim])
        
        indices = get_indices(input_feature=video_feature, fps=fps)
        video_feature = torch.index_select(video_feature, dim=0, index=indices)
            
        video_feature_list.append(video_feature)
        video_mask_list.append(torch.ones([video_feature.shape[0]], dtype=torch.int8)*(i+1))
            
    video_history = {'feature': modules.pad(torch.cat(video_feature_list, dim=0), padding_size=video_pad_size),
                     'mask': modules.pad(torch.cat(video_mask_list, dim=0), padding_size=video_pad_size)
                    }
    
    return video_history

def get_indices(input_feature, fps):
    ''' video is 30 fps '''
    index = torch.tensor([i for i in range(input_feature.shape[0]) if i % (30//fps) == 0])
    return index


def multimodal_align(text_history, audio_history, video_history):
    max_align = torch.max(torch.min(video_history['mask']), torch.min(audio_history['mask']))
    
    if max_align != 0:
        video_history['mask'] = align_length(video_history['mask'], max_align)
        audio_history['mask'] = align_length(audio_history['mask'], max_align)
        
    num_hist = len(torch.unique(video_history['mask']))
    if 0 in torch.unique(video_history['mask']):
        num_hist -= 1  # except 0 (padding)
    
    text_history = text_history[-num_hist:]
    
    return text_history, audio_history, video_history, num_hist

def align_length(mask, max_align):
    a = (max_align <= mask)  # 1 <= mask
    return mask * a


def collate_fn(samples):
    inputs, labels, identifiers = zip(*samples)
    
    # Collate inputs
    text_histories, audio_histories, video_histories, speaker_histories = zip(*inputs)
    audio_feature_batch = torch.stack([sample['feature'] for sample in audio_histories])
    audio_mask_batch = torch.stack([sample['mask'] for sample in audio_histories])
    video_feature_batch = torch.stack([sample['feature'] for sample in video_histories])
    video_mask_batch = torch.stack([sample['mask'] for sample in video_histories])

    inputs_batch = [
        text_histories,
        {'feature': audio_feature_batch,
        'mask': audio_mask_batch},
        {'feature': video_feature_batch,
        'mask': video_mask_batch},
        speaker_histories
    ]

    # Collate labels
    label_text_batch = [label[0] for label in labels]
    label_audio_batch = [label[1] for label in labels]
    label_description_batch = [label[2] for label in labels]

    labels_batch = [
        label_text_batch,
        label_audio_batch,
        label_description_batch,
    ]


    return inputs_batch, labels_batch, identifiers