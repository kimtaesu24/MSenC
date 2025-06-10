import torch
import cv2
from PIL import Image

def pad(inputs, padding_size):
    '''
    Used: dataset, model.inference()
    '''
    # print("inputs.shape: ", inputs.shape)
    if len(inputs.shape) == 1:  # 1D
        tmp = torch.zeros(padding_size)
        if len(inputs) > padding_size:
            tmp = inputs[-padding_size:]  # truncation
        else:
            tmp[-len(inputs):] = inputs  # padding
            
    elif len(inputs.shape) == 2:  # 2D
        tmp = torch.zeros(padding_size, inputs.shape[1])
        if inputs.shape[0] > padding_size:
            tmp = inputs[-padding_size:]  # truncation
        else:
            tmp[-inputs.shape[0]:] = inputs  # padding
    
    # print("tmp.shape: ", tmp.shape)
    return tmp

def speech_token_pad(inputs, padding_size):
    '''
    Used: dataset, model.inference()
    '''
    # print("inputs.shape: ", inputs.shape)
    tmp = torch.zeros(inputs.shape[0], padding_size, inputs.shape[-1])
    if inputs.shape[1] > padding_size:
        tmp = inputs[:,-padding_size:]  # truncation
        mask = torch.ones(inputs.shape[0], padding_size, inputs.shape[-1])
    else:
        tmp[:,:inputs.shape[1]] = inputs  # padding
        mask = torch.zeros(inputs.shape[0], padding_size, inputs.shape[-1])
        mask[:,:inputs.shape[1]] = 1
        
    # print("tmp.shape: ", tmp.shape)
    return tmp, mask


def history_pad(inputs, padding_size):
    '''
    Used: dataset, model.inference()
    '''
    if len(inputs) > padding_size:
        tmp = inputs[-padding_size:]  # truncation
    else:
        tmp = ["" for i in range(padding_size)]
        tmp[-len(inputs):] = inputs  # padding
    return tmp

# select frames per 'fps' frames
def select_frames(video_path, fps):
    vidcap = cv2.VideoCapture(video_path)
    
    frames = []
    success,image = vidcap.read()
    count = 0
    while success:
        if count % fps == 0:
            image = OpenCV2PIL(image)
            frames.append(image)
        
        success,image = vidcap.read()
        count += 1
        
    return frames
        
def OpenCV2PIL(opencv_image):
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_coverted)
    return pil_image