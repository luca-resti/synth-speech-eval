from models import sq_ast_mod
from models import ppgs_wrapper
from models import pdsm
from util import plot_helper, kde_tools

import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime


if __name__ == "__main__":

    if len(sys.argv) > 1:
        config_path = "configs/" + sys.argv[1] + ".yaml"
    else:
        config_path = "configs/default.yaml"

    with open(config_path, 'r') as f:
        config_file = yaml.safe_load(f)

    start_time = datetime.now()

    output_dir = config_file["output_dir"] + "/" + config_file["dataset_name"] + "_" +  start_time.strftime("%Y%m%d_%H%M") + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_ind_csv_path = os.path.join(output_dir, "output_sq_ast.csv")
    output_ind_csv_path_thresh = os.path.join(output_dir, "output_thresholded.csv")

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

    output_sysfig_dir = output_dir + "/sys_analysis/"
    if not os.path.exists(output_sysfig_dir):
        os.makedirs(output_sysfig_dir)

    plot_helper.plot_sys_violin_plot(config_file, sq_ast_mod.ALL_DIMS, output_ind_df, output_sysfig_dir)

    output_ind_df_thresh = output_ind_df[
        (output_ind_df["sq_mos"] <= config_file["score_threshold"]) |
        (output_ind_df["sq_noi"] <= config_file["score_threshold"]) |
        (output_ind_df["sq_dis"] <= config_file["score_threshold"]) |
        (output_ind_df["sq_col"] <= config_file["score_threshold"]) |
        (output_ind_df["sq_loud"] <= config_file["score_threshold"])
    ]
    output_ind_df_thresh["index"] = np.arange(len(output_ind_df_thresh))
    output_ind_df_thresh = output_ind_df_thresh.reset_index(names=["pre_threshold_index"])

    ppgs_pred, ppgs_dict, word_alignments = ppgs_wrapper.whisperx_get_ppgs(output_ind_df_thresh, config_file)

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

    kde_x_info = np.zeros(shape=(len(output_ind_df_thresh), len(sq_ast_mod.ALL_DIMS), int(10/0.01)), dtype=np.float32)
    kde_y_info = np.zeros(shape=(len(output_ind_df_thresh), len(sq_ast_mod.ALL_DIMS), 128), dtype=np.float32)

    pdsm_current_time = datetime.now()

    for file_idx, df_row in output_ind_df_thresh.iterrows():
        file_path = df_row["file_path"]
        old_index = df_row["pre_threshold_index"]

        ppgs_pred_file = ppgs_pred[file_idx, 0, :, :fbank_lengths[old_index][1]]

        for dim_index in range(len(sq_ast_mod.ALL_DIMS)):

            dim = sq_ast_mod.ALL_DIMS[dim_index]

            attn_rescaled = sq_ast_mod.get_scaled_saliency_map(
                attention_flows[old_index, dim_index, :, :], 
                config_file["saliency"]["saliency_interp_method"]
            )
            attn_rescaled = attn_rescaled.squeeze().squeeze().detach().numpy()
            attn_rescaled = attn_rescaled[:, :fbank_lengths[old_index][1]]
            attn_rescaled = (attn_rescaled - attn_rescaled.min()) / (attn_rescaled.max() - attn_rescaled.min() + 1e-8)

            if config_file["pdsm"]["preprocess"] == "abs":
                pdsm_preprocess = np.abs

            if config_file["pdsm"]["pool"] == "mean":
                pdsm_pool = np.mean
            elif config_file["pdsm"]["pool"] == "sum":
                pdsm_pool = np.sum
            elif config_file["pdsm"]["pool"] == "l2_norm":
                pdsm_pool = pdsm.l2_norm

            dim_pdsm, dim_phon = pdsm.PDSM(
                attn_rescaled, 
                ppgs_pred_file,
                ppgs_dict,
                pdsm_preprocess, 
                pdsm_pool, 
                config_file["pdsm"]["k_method"],
                config_file["pdsm"]["k"]
            )

            pdsm_info[f"pdsm_sq_{dim}"].append(dim_phon) # store the phoneme information
        
            kde_x, kde_y = kde_tools.get_kde_from_saliency(attn_rescaled)
            kde_x_info[file_idx, dim_index, :len(kde_x)] = kde_x
            kde_y_info[file_idx, dim_index, :] = kde_y

            # save the output images for scores that don't meet the threshold
            if config_file["saliency"]["output_saliency_overlay"]:
                if sq_ast_pred[old_index, dim_index] < config_file["score_threshold"]:

                    _, mel_spec = sq_ast_ds.__getitem__(old_index)

                    plot_helper.plot_saliency_with_pdsm(
                        old_index, mel_spec, dim_pdsm, fbank_lengths[old_index][1], 
                        dim_phon, config_file, file_path, dim, sq_ast_pred[old_index, dim_index], 
                        output_saliency_dir, attn_rescaled
                    )

                    plot_helper.plot_saliency_jointgrid(
                        config_file, attn_rescaled, mel_spec, 
                        kde_x_info[file_idx, dim_index, :len(kde_x)], kde_y_info[file_idx, dim_index, :], 
                        word_alignments[file_idx], old_index, file_path, dim, sq_ast_pred[old_index, dim_index], output_saliency_dir
                    )
        
        plot_helper.plot_kde_along_waveform(
            config_file, kde_x_info[file_idx, :, :fbank_lengths[old_index][1]], sq_ast_mod.ALL_DIMS, file_path, word_alignments[file_idx], 
            sq_ast_pred[old_index, :], output_saliency_dir, old_index
        )

    print(f"PDSM/KDE processing completed in time: {datetime.now() - pdsm_current_time}")

    for dim in sq_ast_mod.ALL_DIMS:
        output_ind_df_thresh[f"pdsm_sq_{dim}"] = pdsm_info[f"pdsm_sq_{dim}"]
        output_ind_df_thresh[f"pdsm_sq_{dim}_num"] = len(pdsm_info[f"pdsm_sq_{dim}"])

    output_ind_df.to_csv(output_ind_csv_path, index=False)
    output_ind_df_thresh.to_csv(output_ind_csv_path_thresh, index=False)

    for dim in sq_ast_mod.ALL_DIMS:
        thresholded_df = output_ind_df_thresh[output_ind_df_thresh[f"sq_{dim}"] < config_file["score_threshold"]]
        single_hist, double_hist = pdsm.get_bulk_hists_for_system(thresholded_df, dim)
        plot_helper.plot_phoneme_hists(
            single_hist, double_hist, config_file["dataset_name"], output_sysfig_dir, dim, config_file["score_threshold"]
        )

    print(f"Completed analysis in {str((datetime.now() - start_time))}")

