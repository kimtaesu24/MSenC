import os
import moviepy.editor as mp
import subprocess
import shutil

from datetime import datetime
from datetime import timedelta
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from utils import split_video

video_load_path = './dia_video'
scene_save_path = './scene_video'


def split_video_into_scenes(video_path, diaID, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    # split_video_ffmpeg(video_path, scene_list, show_progress=True)
    if len(scene_list) == 0:
        shutil.copy(video_path, f'{scene_save_path}/dia{diaID}-0.mp4')
    for i, scene in enumerate(scene_list):
        output_video = f'{scene_save_path}/dia{diaID}-{i}.mp4'
        # print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        #     i+1,
        #     scene[0].get_timecode(), scene[0].get_frames(),
        #     scene[1].get_timecode(), scene[1].get_frames(),))
        # print('    Scene %2d: Start %.02f / Frame %d, End %.02f / Frame %d' % (
        #     i+1,
        #     scene[0].get_seconds(), scene[0].get_frames(),
        #     scene[1].get_seconds(), scene[1].get_frames(),))
        
        split_video(video_path, output_video, scene[0].get_timecode(), scene[1].get_timecode())
        
        
if __name__=="__main__":
    os.makedirs(scene_save_path, exist_ok=True)
    dia_videos = os.listdir(video_load_path)
    for video in dia_videos:
        print(video)
        diaID = video.split(".mp4")[0].split('dia')[1]
        split_video_into_scenes(os.path.join(video_load_path, video), diaID)