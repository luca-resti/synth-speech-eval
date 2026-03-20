# Wafaa Wardah, TU-Berlin, 2025
# Modified by Ben Heritage 2026 to extract attention flow

import multiprocessing as mp, threading, sys, gc, logging, os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from transformers import ASTConfig, ASTModel, ASTFeatureExtractor, logging as hf_logging
import numpy as np

hf_logging.set_verbosity_error()  # Silence unnecessary warnings from huggingface
torch.multiprocessing.set_sharing_strategy('file_system')


ALL_DIMS = ["mos", "noi", "dis", "col", "loud"]
COL_IDX = {d: i for i, d in enumerate(ALL_DIMS)}
SALIENCY_INTERP_SIZE = (128, 1024)

MAX_AUDIO_LEN = 10
SQ_FRAME_SIZE = 0.025
SQ_HOP_SIZE = 0.01
SQ_MEL_FREQ = SALIENCY_INTERP_SIZE[0]

# Values from SQ_AST code
DB_MEAN = -10.25446422 # calculated from the validation datasets from July 2025
DB_STD = 4.205750774 # calculated from the validation datasets from July 2025

class ASTXL(torch.nn.Module):
    """Model-XL with individual AST for each dimension."""

    PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, pretrained_model: str = PRETRAINED_MODEL) -> None:
        super(ASTXL, self).__init__()
        try:
            config = ASTConfig.from_pretrained(pretrained_model, output_attentions=True)
            self.ast = ASTModel.from_pretrained(pretrained_model, config=config)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained model from {pretrained_model}") from e

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor, output_attentions: bool = False) -> torch.Tensor:
        if output_attentions:
            outputs = self.ast(features, output_attentions=True)
            hidden_state = outputs.pooler_output            
            attentions = outputs.attentions

            pred = self.fc(hidden_state).squeeze()
            
            return pred, attentions[0]
        else:
            hidden_state = self.ast(features, output_attentions=False).pooler_output
            pred = self.fc(hidden_state).squeeze()
            return pred, attentions[0]
    

class ASTVal(Dataset):
    def __init__(self, df, data_dir, db_mean, db_std):
        self.df = df
        self.data_dir = data_dir
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor.sampling_rate = 48000  # Set to 48 kHz for our 48k audio
        self.feature_extractor.max_length = 1024      # Truncate inputs after 1024 patches
        self.feature_extractor.num_mel_bins = 128     # Customize if needed; keep original as default
        self.feature_extractor.return_attention_mask = True
        self.feature_extractor.mean = db_mean
        self.feature_extractor.std = db_std

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        TARGET_SR = 48_000

        file_name = os.path.join(self.data_dir, self.df['file_path'].iloc[index])
        waveform, sample_rate = torchaudio.load(file_name)

        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze()

        if sample_rate != TARGET_SR:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SR
            )
            waveform = resampler(waveform)
            sample_rate = TARGET_SR

        features = self.feature_extractor(
            waveform, 
            sampling_rate=sample_rate, 
            return_attention_mask=True, 
            return_tensors="pt"
        )['input_values']

        features = features.squeeze()
        return index, features


def get_expected_fbank_shape(waveform, sample_rate):
    """
    Returns expected shape of frequency bank mel_spectrogram

    Parameters
    ----------
    waveform (np.array) : Raw audio file after reading
    sample_rate (int) : Audio sample rate (should be 48k)
    
    Returns
    ----------
    dl (dataloader) : The dataloader needed by SQ_AST
    ds (ASTVal) : The SQ_AST dataset to be used for later analysis and spectrogram access
    fbank_lengths (np.array) : Lengths of spectrograms for each audio file in dataset
    audio_files_txt (list) : List of strings to file paths
    """

    num_mel_bins = SALIENCY_INTERP_SIZE[0]
    frame_length = int(sample_rate * SQ_FRAME_SIZE)  # 25 ms
    frame_step = int(sample_rate * SQ_HOP_SIZE)    # 10 ms
    num_frames = int(np.floor((waveform.shape[1] - frame_length) / frame_step)) + 1
    return (num_mel_bins, num_frames)


def prepare_dataloader(data_dir, wav_path, bs, num_workers):
    """
    Gathers the data in the input directory ready for the SQ_AST forward pass

    Parameters
    ----------
    data_dir (os.path) : Dataset directory, if dataset is a directory, else None
    wav_path (os.path) : Audio file path, if dataset is a single wav file, else None
    bs (int) : SQ_AST batch size
    num_workers (int) : SQ_AST num_workers arguement
    
    Returns
    ----------
    dl (dataloader) : The dataloader needed by SQ_AST
    ds (ASTVal) : The SQ_AST dataset to be used for later analysis and spectrogram access
    fbank_lengths (np.array) : Lengths of spectrograms for each audio file in dataset
    audio_files_txt (list) : List of strings to file paths
    """

    fbank_lengths = []
    audio_files_txt = []

    dtype_dict = {
        'db': str,
        'file_path': str,
        'file_num': float,
        'db_mean': float,
        'db_std': float
    }

    if data_dir:
        db_name = os.path.basename(data_dir)

        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_dict.items()})
        file_paths = [os.path.basename(f) for f in os.listdir(data_dir) if f.endswith('.wav')]

        df['file_path'] = file_paths
        df['db'] = db_name # data_dir basename
        df['file_num'] = range(1, len(file_paths) + 1)
        df['db_mean'] = DB_MEAN
        df['db_std'] = DB_STD
   
        for data_index in range(len(file_paths)):
            data_file = file_paths[data_index]
            audio, sample_rate = torchaudio.load(os.path.join(data_dir, data_file))
            audio_files_txt.append([data_index, os.path.join(data_dir, data_file)])
            fbank_lengths.append([data_index, get_expected_fbank_shape(audio, sample_rate)[1]])
   
        ds = ASTVal(df, data_dir, DB_MEAN, DB_STD)

    else: # single file
        db_name = os.path.basename(str(wav_path).replace('.wav',''))

        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_dict.items()})

        df = pd.DataFrame({
            'db': [db_name],
            'file_path': [os.path.basename(wav_path)],
            'file_num': [1],
            'db_mean': [DB_MEAN],
            'db_std': [DB_STD]
        })

        audio, sample_rate = torchaudio.load(wav_path)
        audio_files_txt.append([0, wav_path])
        fbank_lengths.append([0, get_expected_fbank_shape(audio, sample_rate)[1]])

        ds = ASTVal(df, os.path.dirname(wav_path), DB_MEAN, DB_STD)

    print(f"Running SQ_AST Inference on {len(audio_files_txt)} audio files.")

    dl = DataLoader(dataset=ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    fbank_lengths = np.array(fbank_lengths)

    return dl, ds, fbank_lengths, audio_files_txt


def get_scaled_saliency_map(attn_map, saliency_interp_method):
    """
    Interpolates and scales the input saliency map to the size of the input spectrogram of SQ_AST

    Parameters
    ----------
    attn_map (np.array) : Input saliency map to interpolate up to size
    saliency_interp_method (str) : extracted attention for each layer
    
    Returns
    ----------
    scaled_attn_map (np.array) : Interpolated saliency map to fit size of the input spectrogram to SQ_AST

    """
    attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)
    scaled_attn_map = F.interpolate(attn_map, size=SALIENCY_INTERP_SIZE, mode=saliency_interp_method)
    return scaled_attn_map
    

def tensor_attention_flow(attn_maps):
    """
    Approximates attention flow (using only CLS on each layer for efficiency)

    Parameters
    ----------
    attn_maps (np.array) : extracted attention for each layer
    
    Returns
    ----------
    cls_flow (torch.Tensor) : Flattened tensor for attention flow on CLS (removed <CLS> and <DISTILL> tokens)

    """
    num_layers, seq_len, _ = attn_maps.shape

    I = torch.eye(seq_len).to(attn_maps.device)
    A_adj = 0.5 * attn_maps + 0.5 * I # add identity for residual connections
    
    cls_flow = A_adj[0, 0, :] # CLS attention only!
    for i in range(1, num_layers):
        cls_flow = torch.min(cls_flow, A_adj[i]).sum(dim=-1)
        cls_flow /= cls_flow.sum() # renormalise
    return cls_flow[2:] # remove <CLS> and <DISTILL>


def get_pred_attn(dims, dl, device, bs, threshold_value):
    """
    Evaluates the SQ_AST outputs for given input and dimensions, and returns the attention flow

    Parameters
    ----------
    dims (list) : Dimensions to run in SQ_AST
    dl (dataloader) : Dataloader for the ASTVal input dataset
    device (str) : "cpu"/"cuda"
    bs (int) : Batch size to run on SQ_AST
    threshold_value (float) : User defined threshold to filter predictions below
    
    Returns
    ----------
    predictions (numpy.array) : SQ_AST prediction outputs shape (files, dimensions)
    attention_flows (numpy.array) : SQ_AST extracted attention flow (files, dimensions, transformer layers, input tokens)

    """
    
    attention_flows = torch.Tensor(np.zeros(shape=(dl.__len__()*bs, len(dims), 12, 101)))
    predictions = torch.Tensor(np.zeros(shape=(dl.__len__()*bs, len(dims))))

    for dim_index in range(len(dims)):
        dim = dims[dim_index]
            
        with torch.no_grad():
            with torch.inference_mode():
                model = ASTXL()
                model.load_state_dict(torch.load(f"models/weights/{dim}.pth", map_location=torch.device(device), weights_only=True))
                model.to(device)
                model.eval()

                for batch_index, (index, batch_features) in enumerate(dl):
                    batch_features = batch_features.float().to(device)
                    pred, attentions = model(batch_features, output_attentions=True)
                    attentions = attentions.cpu().detach()
                    if bs == 1:
                        predictions[index, dim_index] = 4*pred + 1 # store prediction for this SQ dimension
                        if 4*pred + 1 <= threshold_value:
                            attention_flow = tensor_attention_flow(attentions[0].cpu().detach())
                            attention_flows[index, dim_index, :, :] = attention_flow.reshape(12, 101)
                    else:
                        for i in index:
                            predictions[i, dim_index] = 4*pred[i-int(bs*batch_index)] + 1 # store prediction for this SQ dimension
                            if 4*pred[i-int(bs*batch_index)] + 1 <= threshold_value:
                                attention_flow = tensor_attention_flow(attentions[i-int(bs*batch_index)].cpu().detach())
                                attention_flows[i, dim_index, :, :] = attention_flow.reshape(12, 101)

    return predictions, attention_flows


def sq_ast_fw(config_file):
    """
    Completes the SQ_AST forward pass for whole dataset and gathers saliency map below threshold defined in config_file

    Parameters
    ----------
    config_file (dict) :  Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding

    Returns
    ----------
    ds (ASTVal) : this is the dataset that contains the preprocessing steps for SQ_AST
    predictions (np.array) : SQ_AST output predictions in the shape (files, dimension)
    saliency_map (np.array) : SQ_AST saliency output in the shape (files, dimensions, transformer layers, input tokens)
    fbank_lengths (np.array) : Time-dimension lengths of the input spectrograms (used in later plots)
    output_ind_df (pandas.dataframe) : Output dataframe for predictions
    """

    current_time = datetime.now()
    dims = ALL_DIMS
    bs = int(config_file["sq_ast"]["batch_size"])
    num_workers = int(0)

    input_dir = os.path.join(config_file["path"], config_file["dataset_name"])

    # check whether is a single file or a directory and set paths accordingly
    if str(input_dir).endswith('.wav'): 
        wav_path = input_dir
        data_dir = None
    else: 
        wav_path = None
        data_dir = input_dir + "/"
    
    # get device from config and check if GPU is available
    if config_file["device"] == "gpu" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Starting SQ_AST Preprocessing")

    if wav_path is not None:
        bs = 1

    # create dataset for sq_ast
    dl, ds, fbank_lengths, audio_files_txt = prepare_dataloader(data_dir, wav_path, bs, num_workers)

    # get the predictions and attention flows for each dimension
    predictions, saliency_map = get_pred_attn(dims, dl, device, bs, config_file["score_threshold"])

    print(f"SQ_AST inference completed in time: {datetime.now() - current_time}")

    # develop dataframe
    output_ind_df = pd.DataFrame(columns=["file_path", "sq_mos", "sq_noi", "sq_dis", "sq_col", "sq_loud"])
    for i, (_, file_path) in enumerate(audio_files_txt):
        output_ind_df.loc[i] = {
            "index": i,
            "file_path": file_path,
            "sq_mos": predictions.cpu().numpy()[i, 0],
            "sq_noi": predictions.cpu().numpy()[i, 1],
            "sq_dis": predictions.cpu().numpy()[i, 2],
            "sq_col": predictions.cpu().numpy()[i, 3],
            "sq_loud": predictions.cpu().numpy()[i, 4]
        }

    predictions = predictions.cpu().numpy()
    saliency_map = saliency_map.cpu().numpy()

    return (
        ds, predictions, saliency_map, fbank_lengths, output_ind_df
    )


def get_thresholded_df(input_df, score_threshold):
    """
    Gets the filtered Dataframe with a given score threshold
    Returns the dataframe with all data lying below this threshold

    Parameters
    ----------
    input_df (pandas.dataframe) : Dataframe of SQ_AST outputs
    score_threshold (float) : User defined threshold to get all below
    
    Returns
    ----------
    output_df_thresh (pandas.dataframe) : Thresholded dataframe

    """
    # get thresholded dataframe under the score threshold
    output_df_thresh = input_df[
        (input_df["sq_mos"] <= score_threshold) |
        (input_df["sq_noi"] <= score_threshold) |
        (input_df["sq_dis"] <= score_threshold) |
        (input_df["sq_col"] <= score_threshold) |
        (input_df["sq_loud"] <= score_threshold)
    ]
    output_df_thresh["index"] = np.arange(len(output_df_thresh))
    output_df_thresh = output_df_thresh.reset_index(names=["pre_threshold_index"])
    return output_df_thresh

