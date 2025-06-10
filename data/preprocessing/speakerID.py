import hdbscan
import numpy as np
import os
import natsort
import pandas as pd
import re

from pyannote.audio import Model
from pyannote.audio import Inference
from utils import video2audio_save
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")

inference = Inference(model, window="whole")
# inference.to(torch.device("cuda"))

# clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric=cosine_distance)  # You can adjust the min_cluster_size parameter
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')  # You can adjust the min_cluster_size parameter

    
def demo():
    # embedding1 = inference("./test/dia0_utt1.wav")
    # embedding2 = inference("./test/dia0_utt3.wav")
    # # `embeddingX` is (1 x D) numpy array extracted from the file as a whole.

    # embedding1 = embedding1.reshape((1,-1))
    # embedding2 = embedding2.reshape((1,-1))

    # print(embedding1.shape)
    # print(embedding2.shape)

    
    # distance = cdist(embedding1, embedding2, metric="cosine")[0,0]
    # print(distance)
    # `distance` is a `float` describing how dissimilar speakers 1 and 2 are.
    
    video_load_path = "./test"
    audio_save_path = "./test_audio"
    
    video_list = natsort.natsorted(os.listdir(video_load_path))
    
    emb_dict = {}
    embeddings = []
    for mp4 in video_list:
        file_name = mp4.split('.mp4')[0]
        # video2audio_save(video_load_path, mp4, audio_save_path)
        
        embedding = inference(os.path.join(audio_save_path, f'{file_name}.wav'))
        # embedding = embedding.reshape((1,-1))
        # print(embedding.shape)
        embeddings.append(embedding)
        
    # Step 1: Compute the cosine similarity matrix
    embeddings = np.array(embeddings, dtype=np.float64)
    print('embeddings.shape', embeddings.shape)
    cosine_sim_matrix = cosine_similarity(embeddings)
    print('cosine_sim_matrix.shape', cosine_sim_matrix.shape)

    # Step 2: Convert the similarity matrix to a distance matrix
    cosine_dist_matrix = 1 - cosine_sim_matrix
    
    cluster_labels = clusterer.fit_predict(cosine_dist_matrix)
    
    # print("len(cluster_labels):", len(cluster_labels))
    print(cluster_labels)
    
    for k, v in zip(video_list, cluster_labels):
        emb_dict[k] = v
    print(emb_dict)
    
    
def error_log(video_load_path, audio_save_path):
    df = pd.read_csv('utterance.csv')
    
    num_dia = max(df['Dialogue_ID'])
    
    speaker_id_list = []
    for dia_id in range(num_dia):
        print(dia_id)
        dia_df = df[df['Dialogue_ID']==dia_id]
        utt_id_list = list(dia_df['Utterance_ID'])
        
        emb_dict = {}
        embeddings = []
        for utt_id in utt_id_list:
            try:
                video2audio_save(video_load_path, f'dia{dia_id}_utt{utt_id}.mp4', audio_save_path)
            except:  # video splitted with speaker diarization method may unable to open. (too short)
                with open('error_speaker.txt', 'a') as f:
                    f.write(f'dia{dia_id}_utt{utt_id}.mp4\n')
                f.close()
            
# Function to rename files within each 'dia' group
def rename_files(dia, files, extension, directory):
    files.sort()  # Sort by 'utt' number
    for new_utt, (old_utt, old_filename) in enumerate(files):
        new_filename = f"{dia}_utt{new_utt}{extension}"
        old_filepath = os.path.join(directory, old_filename)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed {old_filepath} to {new_filepath}")
            
            
def rename_dir_file(directory, extension=".mp4"):
    # Directory containing the video files
    # directory = "path/to/your/directory"

    # Regular expression to match the filenames
    if extension == ".mp4":
        pattern = re.compile(r"(dia\d+)_utt(\d+)\.mp4")
    elif extension == ".wav":
        pattern = re.compile(r"(dia\d+)_utt(\d+)\.wav")

    # Read all filenames in the directory
    filenames = [f for f in natsort.natsorted(os.listdir(directory)) if f.endswith(extension)]

    # Group filenames by 'dia' number
    grouped_files = {}
    for filename in filenames:
        match = pattern.match(filename)
        if match:
            dia, utt = match.groups()
            if dia not in grouped_files:
                grouped_files[dia] = []
            grouped_files[dia].append((int(utt), filename))

    # Rename files in each group
    for dia, files in grouped_files.items():
        rename_files(dia, files, extension, directory)

    print("Renaming complete.")
    
    
# Function to renumber 'utt' column within each 'dia' group
def renumber_utt(group):
    group['Utterance_ID'] = range(len(group))
    return group


def rename_csv(csv_path, error_list):
    df = pd.read_csv(csv_path)
    
    for error_file_name in error_list:
        dia = error_file_name.split('.mp4')[0].split('_')[0].split('dia')[1]
        utt = error_file_name.split('.mp4')[0].split('_')[1].split('utt')[1]
        # print(error_file_name)
        # print(dia)
        # print(utt)
        df = df.drop(df[(df.Dialogue_ID == int(dia)) & (df.Utterance_ID == int(utt))].index)
    # print(df)
    
    df = df.groupby('Dialogue_ID', group_keys=False).apply(renumber_utt)
    
    df.to_csv(csv_path, index=False)

    print("Renumbering complete.")
    
            
def arrange_utterance():
    f = open("error_speaker.txt", 'r')
    error_list = []
    while True:
        line = f.readline()[:-1]
        if not line: break
        error_list.append(line)
    f.close()
    # print(error_list)
    
    # for file_name in error_list:
    #     os.remove(os.path.join('./utt_video2', file_name))
    # rename_dir_file(directory='./utt_video2', extension=".mp4")
    
    # rename_dir_file(directory='./utt_audio2', extension=".wav")
    
    rename_csv('utterance copy.csv', error_list)
    
    
def assign_speakerID(video_load_path, audio_save_path):
    df = pd.read_csv('utterance.csv')
    
    num_dia = max(df['Dialogue_ID'])
    
    speaker_id_list = []
    for dia_id in range(num_dia + 1):
        # print(dia_id)
        dia_df = df[df['Dialogue_ID']==dia_id]
        utt_id_list = list(dia_df['Utterance_ID'])
        
        embeddings = []
        except_list = []
    
        for utt_id in utt_id_list:
            embedding = inference(os.path.join(audio_save_path, f'dia{dia_id}_utt{utt_id}.wav'))
            # print(type(embedding))  # 'numpy.ndarray'
            if np.isnan(embedding).any():
                except_list.append(utt_id)
                print(f'dia{dia_id}_utt{utt_id}.wav has nan value')
            else:
                embeddings.append(embedding)
            
            
        if len(embeddings) > 1:
            embeddings = np.array(embeddings, dtype=np.float64)
            
            # Step 1: Compute the cosine similarity matrix
            # try:
            cosine_sim_matrix = cosine_similarity(embeddings)
            
            # Step 2: Convert the similarity matrix to a distance matrix
            cosine_dist_matrix = 1 - cosine_sim_matrix
            # print(cosine_dist_matrix)
            # print(cosine_dist_matrix.shape)
            
            cluster_labels = clusterer.fit_predict(cosine_dist_matrix)
            # except:
            #     print(f'cannot calculate cosine similarity (Input contains NaN.) at Dialouge_ID {dia_id}')
            #     cluster_labels = [-1 for _ in range(len(utt_id_list))]
            
            # print(cluster_labels)
            cluster_labels = list(cluster_labels)
            for idx in except_list:
                cluster_labels.insert(idx, -1)
            
        else:  # dialogue with single utterance Or only can calculate at least single embedding
            cluster_labels = [0 for _ in range(len(utt_id_list))]
            print(f'DialogueID {dia_id} has at least single utterance')
            
        speaker_id_list = speaker_id_list + cluster_labels
        
        assert len(utt_id_list) == len(cluster_labels), f'utterance length is {len(utt_id_list)}, but cluster_labels length is {len(cluster_labels)} at dialogue {dia_id}'
        
    df["Speaker_ID"] = speaker_id_list
    df.to_csv('utterance_speakerID.csv', index=False)

if __name__=="__main__":
    # demo()
    error_log("./utt_video", "./utt_audio")
    arrange_utterance()
    assign_speakerID("./utt_video", "./utt_audio")