import torch
import ppgs
import numpy as np

def get_ppgs_dict():
    phoneme_dict_in = ppgs.PHONEME_TO_INDEX_MAPPING
    phoneme_dict_out = {}
    for phoneme in phoneme_dict_in.keys():
        phoneme_dict_out[str(phoneme_dict_in[phoneme])] = phoneme
    return phoneme_dict_out


def get_ppgs(audio_files, config_file):

    # use device from config and check if GPU is available
    gpu = None
    if config_file["device"] == "gpu":
        gpu = 0

    audio_files = audio_files

    # get ppgs by the batch of audio files
    batch_audio = torch.zeros((len(audio_files), 1, 10*ppgs.SAMPLE_RATE))

    for _, (index, filename) in enumerate(audio_files):
        audio = ppgs.load.audio(filename)

        # trim to max length of 10 seconds for ppgs extraction
        if audio.shape[1] > 10*ppgs.SAMPLE_RATE:
            audio = audio[:, :10*ppgs.SAMPLE_RATE]

        batch_audio[index, 0, :audio.shape[1]] = audio

    ppgs_out = torch.zeros((len(audio_files), 1, len(ppgs.PHONEMES), 1000))
    
    # batch seems to not be working (TODO: come back to this)
    for _, (index, filename) in enumerate(audio_files):
        ppgs_out[index, :, :] = ppgs.from_audio(batch_audio[index, :, :], ppgs.SAMPLE_RATE, gpu=gpu)

    ppgs_out = ppgs_out.cpu().detach().numpy()

    # apply time smoothing (30ms window recommended in doi: 10.1109/ICSPCS.2010.5709770)
    ppgs_out_avg = np.zeros_like(ppgs_out)

    average_window_num = config_file["ppg"]["avg_window_num"]

    if average_window_num > 1:
        for col in range(np.shape(ppgs_out_avg)[0]):
            for row in range(np.shape(ppgs_out_avg)[2]):
                ppgs_out_row = ppgs_out[col, 0, row, :]
                ppgs_out_avg[col, :, row, :] = np.convolve(ppgs_out_row, np.ones(average_window_num)/average_window_num, mode='same')
    else:
        ppgs_out_avg = ppgs_out

    ppgs_out_avg = ppgs_out_avg[:, :, :, 1:-1] # remove padding from ppgs
    ppgs_out_avg = (ppgs_out_avg - ppgs_out_avg.min()) / (ppgs_out_avg.max() - ppgs_out_avg.min() + 1e-8)
    
    return ppgs_out_avg

