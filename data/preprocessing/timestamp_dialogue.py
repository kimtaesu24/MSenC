import json
import moviepy.editor as mp
import os

with open("timestamp.json", "r") as fp:
    timestamp = json.load(fp)
    
dia_idx = 0
for key, value in zip(timestamp.keys(), timestamp.values()):
    for time_pair in value:
        if time_pair[1] - time_pair[0] <= 0:
            print(f"[{key}] subclip from {time_pair[0]}'s to {time_pair[1]}'s as named dia{dia_idx}.mp4 has problem")
            continue
        # file_name = f"English Conversation {prev_video_num:02d}.mp4"
        
        my_clip = mp.VideoFileClip(os.path.join('./raw_video', key))

        sub_clip = my_clip.subclip(time_pair[0], time_pair[1])
        sub_clip.write_videofile(f"./dia_video/dia{dia_idx}.mp4", verbose=False, logger=None)
        
        print(f"[{key}] make subclip from {time_pair[0]}'s to {time_pair[1]}'s as named dia{dia_idx}.mp4")
        
        dia_idx += 1