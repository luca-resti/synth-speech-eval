import numpy as np
import whisperx
from phonemizer import phonemize
import numpy as np
import os
from datetime import datetime


os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:/Program Files/eSpeak NG/libespeak-ng.dll" 

PAD_IN_SECONDS = 1.0
TOTAL_AUDIO_LENGTH = 10
WHISPERX_FS = 16000
DESIRED_FRAME_DURATION = 0.01 # 10ms to match input mel spectrogram

def whisperx_get_ppgs(output_ind_df, config_file): # (audio_file, device)

    current_time = datetime.now()

    audio_files = output_ind_df["file_path"]

    # use device from config and check if GPU is available
    device = "cpu"
    compute_type = "int8"
    if config_file["device"] == "gpu": 
        device = "cuda"
        compute_type = "float16"

    model = whisperx.load_model("small", device=device, compute_type=compute_type)
    model_a, metadata = whisperx.load_align_model(
        language_code=config_file["language"], 
        device=device, 
        model_name="facebook/wav2vec2-lv-60-espeak-cv-ft"
    )

    # get ppgs by the batch of audio files
    batch_audio = np.zeros((1, int((len(audio_files)*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS))*WHISPERX_FS)), dtype=np.float32) # only works for float32?

    for index, filename in enumerate(audio_files):
        audio = whisperx.load_audio(filename)

        # trim to max length of 10 seconds for ppgs extraction
        if audio.shape[0] > TOTAL_AUDIO_LENGTH*WHISPERX_FS:
            audio = audio[:TOTAL_AUDIO_LENGTH*WHISPERX_FS]

        start_time = int(index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS)*WHISPERX_FS)
        batch_audio[:, start_time:start_time+audio.shape[0]] = audio

    vocab = metadata["dictionary"] # get the phoneme list
    phoneme_to_idx = {p: i for i, p in enumerate(vocab)}
    ppgs_out = np.zeros((len(audio_files), 1, len(vocab), int(TOTAL_AUDIO_LENGTH/DESIRED_FRAME_DURATION)))
    sil_idx = vocab["</s>"]

    preproc_time = datetime.now()
    print(f"WhisperX Preprocessing completed in time: {preproc_time - current_time}")
    print(f"Running WhisperX Inference on {len(audio_files)} audio files.")

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

    output_word_alignment = [[] for i in range(len(audio_files))]
    for segment in result_aligned_words["segments"]:
        for word_segment in segment["words"]:
            if ("start" in word_segment.keys()) and ("end" in word_segment.keys()):
                file_index = int(np.floor((word_segment["start"]+PAD_IN_SECONDS/2)/(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS)))
                word_segment["start"] = np.max([word_segment["start"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), 0.0])
                word_segment["end"] = np.min([word_segment["end"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), TOTAL_AUDIO_LENGTH])
                output_word_alignment[file_index].append(word_segment)

    result_aligned_all = [[] for i in range(len(audio_files))]
    for segment in result_aligned["segments"]:
        for char_data in segment["chars"]:
            if ("start" in char_data.keys()) and ("end" in char_data.keys()):
                file_index = int(np.floor((char_data["start"]+PAD_IN_SECONDS/2)/(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS)))
                char_data["start"] = np.max([char_data["start"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), 0.0])
                char_data["end"] = np.min([char_data["end"] - file_index*(TOTAL_AUDIO_LENGTH+PAD_IN_SECONDS), TOTAL_AUDIO_LENGTH])
                result_aligned_all[file_index].append(char_data)

    for index, df_row in output_ind_df.iterrows():

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

    print(f"WhisperX Transcription and Phonemization completed in time: {datetime.now() - preproc_time}")
    print(f"WhisperX Overall Usage completed in time: {datetime.now() - current_time}")

    return (
        ppgs_out[:, :, :, 1:-1], # remove padding
        phoneme_dict_out,
        output_word_alignment
    )

