#https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPModel
import torch
import os
import cv2
import natsort

from transformers import CLIPModel, CLIPProcessor
from PIL import Image

video_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
video_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    
    frames = []
    success,image = vidcap.read()
    while success:
        image = OpenCV2PIL(image)
        frames.append(image)
        
        success,image = vidcap.read()
        
    return frames
    
def OpenCV2PIL(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)
    return pil_image

def get_image_features(video_path):
    frames = load_frames(video_path)
    
    inputs = video_processor(images=frames, return_tensors="pt")
    # video_feature = video_model(inputs['pixel_values']).last_hidden_state  # ([seq_len, 50, hidden_dim])
    video_feature = video_model.get_image_features(**inputs)
    # print(video_feature.shape)  # [seq_len, hidden_dim=512]
    
    return video_feature

def video_feature_extract(data_name):
    print(data_name)
    for mode in ['train', 'valid', 'test']:
        print(mode)
        file_root = f"./{data_name}/{mode}"
        
        save_path = f'{file_root}/video_feature'
        os.makedirs(save_path, exist_ok=True)
    
        os.makedirs(f"./{data_name}/error", exist_ok=True)
        
        for video in natsort.natsorted(os.listdir(f"./{file_root}/video")):
            try:
                video_path = f"{file_root}/video/{video}"
                video_feature = get_image_features(video_path)
                
                video_name = os.path.basename(video).split('.mp4')[0]
                torch.save(video_feature, f'{save_path}/{video_name}.pt')
            except:
                with open(f"./{data_name}/error/{mode}_video_feature_error.txt",'a') as f:
                    f.write(f"This file occurs ERROR =====>> {file_root}/audio/{audio} \n")
                f.close()         

if __name__=="__main__":
    # video_feature_extract('MELD')
    video_feature_extract('MSC')