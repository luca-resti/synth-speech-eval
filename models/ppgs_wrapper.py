import numpy as np
import whisperx
from phonemizer import phonemize
import numpy as np
import os
from datetime import datetime
from models import sq_ast_mod
import torchaudio

import logging
logger = logging.getLogger()

# Set for espeak requirement (default location)
if os.environ.get('OS','') == 'Windows_NT':
    logger.info(f" === WINDOWS === ")
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:/Program Files/eSpeak NG/libespeak-ng.dll" 
else:
    logger.info(f" === LINUX === ")
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1" 
    os.environ['PHONEMIZER_ESPEAK_PATH'] = "/usr/bin/espeak-ng"

PAD_IN_SECONDS = 1.0
TOTAL_AUDIO_LENGTH = sq_ast_mod.MAX_AUDIO_LEN
WHISPERX_FS = 16000
DESIRED_FRAME_DURATION = sq_ast_mod.SQ_HOP_SIZE # 10ms to match input mel spectrogram


def whisperx_get_ppgs(input_df, config_file):
    '''
    Runs all files in input_df through WhisperX, to phoneme and word alignment

    Parameters
    ----------
    input_df (pandas.dataframe) : 
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding

    Returns
    ----------
    ppgs_out (numpy.array) : All of the PPGS returned from WhisperX force alignment for each file
    phoneme_dict_out (dict) : Returns the dictionary linking the index and phonemes outputted by WhisperX
    output_word_alignment (list) : The aligned word-transcription for each file
    '''

    input_dir = os.path.join(config_file["path"], config_file["dataset_name"])

    current_time = datetime.now()

    # use device from config and check if GPU is available
    device = config_file["device"]
    compute_type = "int8"
    if config_file["device"] == "cuda": 
        device = "cuda"
        compute_type = "float16" # should run float16 if on cuda

    model = whisperx.load_model("small", device=device, compute_type=compute_type)
    model_a, metadata = whisperx.load_align_model(
        language_code=config_file["language"], 
        device=device, 
        model_name="facebook/wav2vec2-lv-60-espeak-cv-ft"
    )

    # get ppgs by the batch of audio files
    batch_audio = np.zeros((1, int((len(input_df)*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS))*WHISPERX_FS)), dtype=np.float32) # only works for float32?

    for index, row in input_df.iterrows():
        filename = row["file_path"]
        audio, sample_rate = torchaudio.load(os.path.join(input_dir, filename))

        # get channel of waveform
        if audio.shape[0] > 1:
            audio = audio[row['wav_channel'], :]
        else:
            audio = audio.squeeze()

        # get segment of waveform
        audio = audio[row['wav_start']:row['wav_end']]

        # resample before segment
        if sample_rate != WHISPERX_FS:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=WHISPERX_FS
            )
            audio = resampler(audio)
            sample_rate = WHISPERX_FS

        audio = audio.numpy().astype(np.float32)

        # trim to max length of 10 seconds for ppgs extraction
        if audio.shape[0] > TOTAL_AUDIO_LENGTH*WHISPERX_FS:
            audio = audio[:TOTAL_AUDIO_LENGTH*WHISPERX_FS]

        start_time = int(index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS)*WHISPERX_FS)
        batch_audio[:, start_time:start_time+audio.shape[0]] = audio

    vocab = metadata["dictionary"] # get the phoneme list
    phoneme_to_idx = {p: i for i, p in enumerate(vocab)}
    ppgs_out = np.zeros((len(input_df), 1, len(vocab), int(TOTAL_AUDIO_LENGTH/DESIRED_FRAME_DURATION)))
    sil_idx = vocab["</s>"]

    preproc_time = datetime.now()
    logger.info(f"WhisperX Preprocessing completed in time: {preproc_time - current_time}")
    logger.info(f"Running WhisperX Inference on {len(input_df)} audio files.")

    audio = batch_audio[0, :]
    result = model.transcribe(audio, batch_size=int(config_file["whisperx"]["batch_size"]), chunk_size=2) # chunk size determined to recover silences

    result_aligned_words = whisperx.align(
        result["segments"], model_a, metadata, audio, device, return_char_alignments=False
    )
    # then phonemize for phoneme alignment
    for segment in result["segments"]:
        segment["text_old"] = segment["text"]
        segment["text"] = phonemize(segment["text"], language='en-us', backend='espeak')

    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, device, return_char_alignments=True
    )

    output_word_alignment = [[] for i in range(len(input_df))]
    for segment in result_aligned_words["segments"]:
        for word_segment in segment["words"]:
            if ("start" in word_segment.keys()) and ("end" in word_segment.keys()):
                file_index = int(np.floor((word_segment["start"]+PAD_IN_SECONDS/2)/(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS)))
                word_segment["start"] = np.max([word_segment["start"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), 0.0])
                word_segment["end"] = np.min([word_segment["end"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), TOTAL_AUDIO_LENGTH])
                output_word_alignment[file_index].append(word_segment)

    result_aligned_all = [[] for i in range(len(input_df))]
    for segment in result_aligned["segments"]:
        for char_data in segment["chars"]:
            if ("start" in char_data.keys()) and ("end" in char_data.keys()):
                file_index = int(np.floor((char_data["start"]+PAD_IN_SECONDS/2)/(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS)))
                char_data["start"] = np.max([char_data["start"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), 0.0])
                char_data["end"] = np.min([char_data["end"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), TOTAL_AUDIO_LENGTH])
                result_aligned_all[file_index].append(char_data)

    for index, df_row in input_df.iterrows():

        for char_data in result_aligned_all[index]:
            p_label = char_data["char"]
            if p_label in phoneme_to_idx:
                # convert seconds to frame indices
                start_f = int(char_data["start"] / (DESIRED_FRAME_DURATION))
                end_f = int(char_data["end"] / (DESIRED_FRAME_DURATION))
                
                # generate discretized ppg
                p_idx = phoneme_to_idx[p_label]
                ppgs_out[index, :, p_idx, start_f:end_f] = 1.0

        silences = (np.sum(ppgs_out[index, :, :, :], axis=1) == 0)[0].astype(int)
        ppgs_out[index, 0, silences, sil_idx] = 1.0

    phoneme_dict_out = {}
    for phoneme in phoneme_to_idx.keys():
        phoneme_dict_out[str(phoneme_to_idx[phoneme])] = phoneme

    logger.info(f"WhisperX Transcription and Phonemization completed in time: {datetime.now() - preproc_time}")
    logger.info(f"WhisperX Overall Usage completed in time: {datetime.now() - current_time}")

    return (
        ppgs_out[:, :, :, 1:-1], # remove padding
        phoneme_dict_out,
        output_word_alignment
    )


def get_agg_asr_confidence(word_alignment):
    """
    Gets the aggregate ASR confidence over whole audio segment

    Parameters
    ----------
    word_alignment (dict) : word alignment data from whisperx

    Returns
    ----------
    asr_confidence (float) : Mean and Median percentage of ASR confidence per word (0 if no words aligned)
    """
    mean_asr_confidence = 0.0
    median_asr_confidence = 0.0

    if len(word_alignment) > 0:
        asr_confidence = np.zeros(shape=(len(word_alignment)), dtype=np.float32)
        for index, word in enumerate(word_alignment):
            asr_confidence[index] = word["score"]
        mean_asr_confidence = np.mean(asr_confidence)
        median_asr_confidence = np.median(asr_confidence)


    return (mean_asr_confidence, median_asr_confidence)