import numpy as np
import whisperx
from phonemizer import phonemize
import numpy as np
import os
from datetime import datetime


os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:/Program Files/eSpeak NG/libespeak-ng.dll" 


def whisperx_get_ppgs(audio_files, config_file): # (audio_file, device)

    current_time = datetime.now()

    # use device from config and check if GPU is available
    device = "cpu"
    if config_file["device"] == "gpu": 
        device = "cuda"
    
    model = whisperx.load_model("small", device=device)
    model_a, metadata = whisperx.load_align_model(
        language_code=config_file["language"], 
        device=device, 
        model_name="facebook/wav2vec2-lv-60-espeak-cv-ft"
    )

    # get ppgs by the batch of audio files
    batch_audio = np.zeros((len(audio_files), 1, 10*16000), dtype=np.float32)

    for _, (index, filename) in enumerate(audio_files):
        audio = whisperx.load_audio(filename)

        # trim to max length of 10 seconds for ppgs extraction
        if audio.shape[0] > 10*16000:
            audio = audio[:10*16000]

        batch_audio[index, :, :audio.shape[0]] = audio

    vocab = metadata["dictionary"] # get the phoneme list
    phoneme_to_idx = {p: i for i, p in enumerate(vocab)}
    desired_frame_duration = 0.01 # 10ms to match input mel spectrogram
    ppgs_out = np.zeros((len(audio_files), 1, len(vocab), 1000))
    sil_idx = vocab["</s>"]

    output_word_alignment = []

    preproc_time = datetime.now()
    print(f"WhisperX Preprocessing completed in time: {preproc_time - current_time}")

    for _, (index, filename) in enumerate(audio_files):

        audio = batch_audio[index, 0, :]
        result = model.transcribe(audio, batch_size=16, chunk_size=2)

        result_aligned_words = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )
        output_word_alignment.append(result_aligned_words["word_segments"])

        for segment in result["segments"]:
            segment["text_old"] = segment["text"]#.replace(" ", "<s>").replace(".", "<s>")
            segment["text"] = phonemize(segment["text"], language='en-us', backend='espeak')

        result_aligned = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=True
        )

        for segment in result_aligned["segments"]:
            for char_data in segment["chars"]:
                p_label = char_data["char"]
                if p_label in phoneme_to_idx:
                    # convert seconds to frame indices
                    start_f = int(char_data["start"] / (desired_frame_duration))
                    end_f = int(char_data["end"] / (desired_frame_duration))
                    
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

