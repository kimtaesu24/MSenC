from moviepy.editor import *
import os


def video2audio_save(load_path, video, save_path):
    # Load the video
    video_name = video.split('.mp4')[0]
    video = VideoFileClip(os.path.join(load_path, f"{video_name}.mp4"))

    # Extract audio
    audio = video.audio

    # Save audio as WAV
    os.makedirs(save_path, exist_ok=True)
    audio.write_audiofile(os.path.join(save_path, f"{video_name}.wav"), verbose=False, logger=None)

    # Close the video file
    video.close()
    
def video2audio_file(load_path, video):
    # Load the video
    # video_name = video.split('.mp4')[0]
    videofile = VideoFileClip(os.path.join(load_path, video))

    # Extract audio
    audio = videofile.audio.to_soundarray()[:,1] ## stereo -> mono
    
    # Close the video file
    videofile.close()

    return audio

def calculate_duration(start_time, end_time):
    """
    Calculate duration between start_time and end_time.

    Parameters:
    - start_time: Start time (format: "hh:mm:ss" or "mm:ss" or "ss").
    - end_time: End time (format: "hh:mm:ss" or "mm:ss" or "ss").

    Returns:
    - Duration (format: "hh:mm:ss").
    """
    # Determine the correct format based on the length of the input time
    if not isinstance(start_time, str):
        start_time = str(timedelta(seconds=start_time, microseconds=1))
        end_time = str(timedelta(seconds=end_time, microseconds=1))

    if len(end_time.split(':')[-1]) > 2:
        fmt = '%H:%M:%S.%f' if len(start_time.split(':')) == 3 else '%M:%S.%f'
    else:
        fmt = '%H:%M:%S' if len(start_time.split(':')) == 3 else '%M:%S'
        
    # print(end_time)
    # print(start_time)
    # print(datetime.strptime(end_time, fmt))
    # print(datetime.strptime(start_time, fmt))
    
    tdelta = datetime.strptime(end_time, fmt) - datetime.strptime(start_time, fmt)

    return str(tdelta)

def split_video(input_file, output_file, start_time, end_time):
    """
    Split a video using FFmpeg based on start and end times.

    Parameters:
    - input_file: Path to the input video file.
    - output_file: Path to the output video file.
    - start_time: Start time for the segment (format: "hh:mm:ss" or "mm:ss" or "ss").
    - end_time: End time for the segment (format: "hh:mm:ss" or "mm:ss" or "ss").
    """
        
    duration = calculate_duration(start_time, end_time)
    # print("start_time:",start_time)
    # print("end_time:",end_time)
    # print("duration:",duration)
    
    command = [
        'ffmpeg',
        '-ss', str(start_time),
        '-t', str(duration),
        '-i', input_file,
        # '-to', str(end_time),
        # '-c', 'copy',
        # '-c:v', "libx264",
        # '-c:a', 'copy',
        '-b:v', '744k',
        '-b:a', '128k',
        '-loglevel', 'quiet',
        output_file
    ]
    
    subprocess.run(command, check=True)
    
if __name__=='__main__':
    # video2audio_save(load_path='./utt_video',
    #                 video='dia1_utt5.mp4',
    #                 save_path='./test_audio')
    for i in range(13):
        video2audio_save(load_path='./utt_video',
                        video=f'dia1119_utt{i}.mp4',
                        save_path='./utt_audio')