import pandas as pd
import os
import torchaudio
import numpy as np


MIN_AUDIO_LEN = 2.0
MAX_AUDIO_LEN = 10.0
TARGET_SAMPLE_RATE = 48000
PAD_AUDIO_LEN = 1.5 #overlap for segments over 10s


def get_segmentation_info(audio_length, fs):
    n_segs = int(np.ceil((audio_length - PAD_AUDIO_LEN*fs)/(MAX_AUDIO_LEN*fs - PAD_AUDIO_LEN*fs)))
    length = np.floor(((audio_length - PAD_AUDIO_LEN*fs)/n_segs) - PAD_AUDIO_LEN*fs)
    seg_samples = [
        (
            int(i*(length + PAD_AUDIO_LEN*fs)), 
            int((i+1)*(length + PAD_AUDIO_LEN*fs) + PAD_AUDIO_LEN*fs)
        ) for i in range(int(n_segs))
    ]
    return n_segs, seg_samples


def get_input_dataset(config_file):
    """
    Validates input audio files are within restrictions and chops samples
    if they are longer than 10 seconds or have multiple channels

    Parameters
    ----------
    config_file (dict) :  Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding

    Returns
    ----------
    input_df (pandas.DataFrame) : Dataframe conatining all input nformation


    """

    input_dir = os.path.join(config_file["path"], config_file["dataset_name"])

    input_df = pd.DataFrame(
        data=[], 
        columns=[
            "db", "file_num", "file_path", "file_fs", 
            "total_wav_channels", "wav_channel", "total_wav_segments", "wav_segment",
            "wav_start", "wav_end"
        ]
    )
    
    wav_path = None
    data_dir = None

    if str(input_dir).endswith('.wav'): 
        wav_path = input_dir
        data_dir = None
    else: 
        data_dir = input_dir + "/"

    added_files = 0

    if data_dir:
        db_name = os.path.basename(data_dir)
        file_paths = [os.path.basename(f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    else: 
        db_name = ""
        data_dir = wav_path
        file_paths = [""]

    for file_index in range(len(file_paths)):
        file_path = file_paths[file_index]
        audio, sample_rate = torchaudio.load(os.path.join(data_dir, file_path))

        audio_len = audio.shape[1]
        if audio_len >= sample_rate*MIN_AUDIO_LEN:
            for channel in range(audio.shape[0]):
                if np.any(audio[channel].numpy() != 0):
                    if audio_len > sample_rate*MAX_AUDIO_LEN:
                        n_segs, sg_info = get_segmentation_info(audio_len, sample_rate)
                        for i_seg in range(n_segs):
                            input_df.loc[added_files] = [
                                db_name, file_index, file_path, sample_rate,
                                int(audio.shape[0]), int(channel), int(n_segs), int(i_seg)+1,
                                int(sg_info[i_seg][0]), int(sg_info[i_seg][1])
                            ]
                            added_files += 1

                    else:
                        input_df.loc[added_files] = [
                            db_name, file_index, file_path, sample_rate,
                            audio.shape[0], int(channel), 1, 0,
                            0, int(audio_len)
                        ]
                        added_files += 1
                else:
                    print(f"{file_path}: channel: {channel} is empty, skipping channel")
        
        else:
            print(f"{file_path} is not above minimum length: {MIN_AUDIO_LEN} seconds")

    return input_df

