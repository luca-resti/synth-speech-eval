# Wafaa Wardah, TU-Berlin, 2025
# Modified by Ben Heritage 2026 to extract attention flow

import multiprocessing as mp, threading, sys, gc, logging, os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from transformers import ASTConfig, ASTModel, ASTFeatureExtractor, logging as hf_logging
import numpy as np

#hf_logging.set_verbosity_error()  # Silence unnecessary warnings from huggingface

torch.multiprocessing.set_sharing_strategy('file_system')


ALL_DIMS = ["mos", "noi", "dis", "col", "loud"]
COL_IDX = {d: i for i, d in enumerate(ALL_DIMS)}
SALIENCY_INTERP_SIZE = (128, 1024)


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
            
            return pred, attentions
        else:
            hidden_state = self.ast(features, output_attentions=False).pooler_output
            pred = self.fc(hidden_state).squeeze()
            return pred, attentions
    

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
    num_mel_bins = 128
    frame_length = int(sample_rate * 0.025)  # 25 ms
    frame_step = int(sample_rate * 0.010)    # 10 ms
    num_frames = int(np.floor((waveform.shape[1] - frame_length) / frame_step)) + 1
    return (num_mel_bins, num_frames)


def prepare_dataloader(data_dir, wav_path, db_mean, db_std, bs, num_workers):
    
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
        df['db_mean'] = db_mean
        df['db_std'] = db_std
   
        for data_index in range(len(file_paths)):
            data_file = file_paths[data_index]
            audio, sample_rate = torchaudio.load(os.path.join(data_dir, data_file))
            audio_files_txt.append([data_index, os.path.join(data_dir, data_file)])
            fbank_lengths.append([data_index, get_expected_fbank_shape(audio, sample_rate)[1]])
   
        ds = ASTVal(df, data_dir, db_mean, db_std)

    else: # single file
        db_name = os.path.basename(str(wav_path).replace('.wav',''))

        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtype_dict.items()})

        df = pd.DataFrame({
            'db': [db_name],
            'file_path': [os.path.basename(wav_path)],
            'file_num': [1],
            'db_mean': [db_mean],
            'db_std': [db_std]
        })

        audio, sample_rate = torchaudio.load(os.path.basename(wav_path))
        audio_files_txt.append([0, os.path.basename(wav_path)])
        fbank_lengths.append([0, get_expected_fbank_shape(audio, sample_rate)[1]])

        ds = ASTVal(df, os.path.dirname(wav_path), db_mean, db_std)

    dl = DataLoader(
        dataset=ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers
    )

    return dl, ds, fbank_lengths, audio_files_txt


def get_scaled_saliency_map(attn_map, saliency_interp_method):
    attn_map = torch.tensor(attn_map).unsqueeze(0).unsqueeze(0)
    return F.interpolate(attn_map, size=SALIENCY_INTERP_SIZE, mode=saliency_interp_method)
    

def tensor_attention_flow(attn_maps):
    """
    Approximates Attention Flow using tensor operations.
    attn_maps: [layers, seq_len, seq_len]
    """
    num_layers, seq_len, _ = attn_maps.shape

    I = torch.eye(seq_len).to(attn_maps.device)
    A_adj = 0.5 * attn_maps + 0.5 * I # add identity for residual connections
    
    cls_flow = A_adj[0, 0, :] # CLS attention only!
    for i in range(1, num_layers):
        cls_flow = torch.min(cls_flow, A_adj[i]).sum(dim=-1)
        cls_flow /= cls_flow.sum() # renormalise
    return cls_flow[2:] # remove <CLS> and <DISTILL>


def get_pred_attn(dims, dl, device):
    
    attention_flows = torch.Tensor(np.zeros(shape=(dl.__len__(), len(dims), 12, 101)))
    predictions = torch.Tensor(np.zeros(shape=(dl.__len__(), len(dims))))

    for dim_index in range(len(dims)):
        dim = dims[dim_index]
            
        with torch.no_grad():
            model = ASTXL()
            model.load_state_dict(torch.load(f"models/weights/{dim}.pth", map_location=torch.device(device), weights_only=True))
            model.to(device)
            model.eval()

            for _, (index, batch_features) in enumerate(dl):
                
                batch_features = batch_features.float().to(device)

                pred, attentions = model(batch_features, output_attentions=True)

                predictions[index, dim_index] = 4*pred + 1 # store prediction for this SQ dimension

                attention_flow = tensor_attention_flow(attentions[0][0].cpu().detach())
                attention_flows[index, dim_index, :, :] = attention_flow.reshape(12, 101)

    return predictions, attention_flows


def sq_ast_fw(config_file):

    current_time = datetime.now()
    output_dir = config_file["output_dir"]
    dims = ALL_DIMS
    bs = int(1)
    num_workers = int(0)

    # check whether is a single file or a directory and set paths accordingly
    if str(config_file["path"]).endswith('.wav'): 
        wav_path = config_file["path"]
        data_dir = None
    else: 
        data_dir = config_file["path"]
        wav_path = None

    db_mean = -10.25446422 # calculated from the validation datasets from July 2025
    db_std = 4.205750774 # calculated from the validation datasets from July 2025
    
    # get device from config and check if GPU is available
    if config_file["device"] == "gpu" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # create dataset for sq_ast
    dl, ds, fbank_lengths, audio_files_txt = prepare_dataloader(data_dir, wav_path, db_mean, db_std, bs, num_workers)
    
    # get the predictions and attention flows for each dimension
    predictions, attention_flows = get_pred_attn(dims, dl, device)

    print(f"SQ_AST inference completed in time: {datetime.now() - current_time}")

    return (
        ds,
        predictions.cpu().numpy(), 
        attention_flows.cpu().numpy(), 
        fbank_lengths, 
        audio_files_txt
    )
