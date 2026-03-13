from models import sq_ast_mod
from models import ppgs_wrapper
from models import pdsm

import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import torchaudio.compliance.kaldi as ta_kaldi


if __name__ == "__main__":

    if len(sys.argv) > 1:
        config_path = "configs/" + sys.argv[1] + ".yaml"
    else:
        config_path = "configs/default.yaml"

    with open(config_path, 'r') as f:
        config_file = yaml.safe_load(f)

    output_dir = config_file["output_dir"] + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_ind_csv_path = os.path.join(output_dir, "output_individual.csv")

    output_ind_df = pd.DataFrame(columns=["file_path", "sq_mos", "sq_noi", "sq_dis", "sq_col", "sq_loud"])

    sq_ast_ds, sq_ast_pred, attention_flows, fbank_lengths, audio_files_txt = sq_ast_mod.sq_ast_fw(config_file)

    for i, (file_idx, file_path) in enumerate(audio_files_txt):
        output_ind_df.loc[i] = {
            "index": i,
            "file_path": file_path,
            "sq_mos": sq_ast_pred[i, 0],
            "sq_noi": sq_ast_pred[i, 1],
            "sq_dis": sq_ast_pred[i, 2],
            "sq_col": sq_ast_pred[i, 3],
            "sq_loud": sq_ast_pred[i, 4]
        }

    ppgs_pred = ppgs_wrapper.get_ppgs(audio_files_txt, config_file)
    ppgs_dict = ppgs_wrapper.get_ppgs_dict()

    pdsm_info = {
        "pdsm_sq_mos":[],
        "pdsm_sq_noi":[],
        "pdsm_sq_dis":[],
        "pdsm_sq_col":[],
        "pdsm_sq_loud":[],
    }

    if config_file["saliency"]["output_saliency_overlay"]:
        output_saliency_dir = output_dir + "/" + "saliency_overlay" + "/"
        if not os.path.exists(output_saliency_dir):
            os.makedirs(output_saliency_dir)

    for i, (file_idx, file_path) in enumerate(audio_files_txt):

        ppgs_pred_file = ppgs_pred[file_idx, 0, :, :fbank_lengths[file_idx][1]]

        for dim_index in range(len(sq_ast_mod.ALL_DIMS)):

            dim = sq_ast_mod.ALL_DIMS[dim_index]

            attn_rescaled = sq_ast_mod.get_scaled_saliency_map(
                attention_flows[file_idx, dim_index, :, :], 
                config_file["saliency"]["saliency_interp_method"]
            )
            attn_rescaled = attn_rescaled.squeeze().squeeze().detach().numpy()
            attn_rescaled = attn_rescaled[:, :fbank_lengths[file_idx][1]]
            attn_rescaled = (attn_rescaled - attn_rescaled.min()) / (attn_rescaled.max() - attn_rescaled.min() + 1e-8)

            dim_pdsm, dim_phon = pdsm.PDSM(
                attn_rescaled, 
                ppgs_pred_file,
                ppgs_dict,
                np.abs, 
                np.sum, 
                config_file["pdsm"]["k_method"],
                config_file["pdsm"]["k"]
            )

            pdsm_info[f"pdsm_sq_{dim}"].append(dim_phon) # store the phoneme information
        
            # save the output images for scores that don't meet the threshold
            if config_file["saliency"]["output_saliency_overlay"]:
                if sq_ast_pred[file_idx, dim_index] < config_file["score_threshold"]:

                    _, mel_spec = sq_ast_ds.__getitem__(file_idx)

                    plt.figure(figsize=(15, 10))
                    plt.subplot(2, 1, 1)
                    plt.imshow(mel_spec.T, aspect='auto', origin='lower', cmap='gray')
                    plt.imshow(dim_pdsm, alpha=0.6, aspect='auto', origin='lower', cmap='turbo')
                    plt.xlim(0, fbank_lengths[file_idx][1])
                    plt.ylim(0, 128)
                    for phon in dim_phon:
                        plt.text((phon[2]+phon[3])*0.5, 128*0.9, phon[1], fontdict={"fontsize":7, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"})
                    plt.xlabel("Time in 10ms Frames")
                    plt.ylabel("Mel Frequency Bin")

                    if config_file["pdsm"]["k_method"] == "threshold":
                        plt.title(f"File: \'{file_path}\', SQ_AST ({dim} score): {np.round(sq_ast_pred[file_idx, dim_index], 1):.1f} (With the {config_file["pdsm"]["k"]:.0f} Most Important Phonemes Highlighted)")
                    elif config_file["pdsm"]["k_method"] == "percent":
                        plt.title(f"File: \'{file_path}\', SQ_AST ({dim} score): {np.round(sq_ast_pred[file_idx, dim_index], 1):.1f} (With the {100*config_file["pdsm"]["k"]:.0f}% Most Important Phonemes Highlighted)")

                    plt.subplot(2, 1, 2)
                    plt.imshow(attn_rescaled, alpha=0.6, aspect='auto', origin='lower', cmap='jet')
                    plt.xlim(0, fbank_lengths[file_idx][1])
                    plt.ylim(0, 128)
                    plt.title(f"Attention Rollout for {dim}")
                    plt.xlabel("Time in 10ms Frames")
                    plt.ylabel("Mel Frequency Bin")
                    plt.savefig(os.path.join(output_saliency_dir, f"{file_idx}_{dim}.png"))
        
    for dim in sq_ast_mod.ALL_DIMS:
        output_ind_df[f"pdsm_sq_{dim}"] = pdsm_info[f"pdsm_sq_{dim}"]
        output_ind_df[f"pdsm_sq_{dim}_num"] = len(pdsm_info[f"pdsm_sq_{dim}"])

    output_ind_df.to_csv(output_ind_csv_path, index=False)