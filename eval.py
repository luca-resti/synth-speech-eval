from models import sq_ast_mod
from models import ppgs_wrapper
from models import pdsm
from util import plot_helper, kde_tools, config_util

import torch
import sys
import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def run_eval(config_file):
    '''
    Aims to give an interpretable understanding of subjective speech quality metrics
    Use SQ_AST model as a backbone, and combines this with the interperetability of WhisperX
    Outputs uttererance and system-level reports for the most "troublesome" utterances below user defined threshold

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding

    Returns
    ----------
    0 : 
    '''
    start_time = datetime.now()

    # create output directories
    output_dir = ""
    if str(config_file["dataset_name"]).endswith(".wav"): 
        dataset_name = os.path.basename(str(config_file["dataset_name"]).replace('.wav',''))
        output_dir = config_file["output_dir"] + "/" + dataset_name + "_" +  start_time.strftime("%Y%m%d_%H%M") + "/"
    else:
       output_dir = config_file["output_dir"] + "/" + config_file["dataset_name"] + "_" +  start_time.strftime("%Y%m%d_%H%M") + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_sysfig_dir = output_dir + "/sys_analysis/"
    if not os.path.exists(output_sysfig_dir):
        os.makedirs(output_sysfig_dir)
    output_individual_dir = output_dir + "/" + "individual_plots" + "/"
    if not os.path.exists(output_individual_dir):
        os.makedirs(output_individual_dir)
    output_ind_csv_path = os.path.join(output_dir, "output_sq_ast.csv")
    output_ind_csv_path_thresh = os.path.join(output_dir, "output_thresholded.csv")

    # unpack sq_ast outputs
    sq_ast_ds, sq_ast_pred, saliency_maps, fbank_lengths, output_ind_df = sq_ast_mod.sq_ast_fw(config_file)

    # plot system violin plots
    if config_file["plots"]["output_sys_violin"]:
        plot_helper.plot_sys_violin_plot(
            config_file, output_ind_df, sq_ast_mod.ALL_DIMS, 
            output_sysfig_dir
        )

    # threshold dataframe
    output_ind_df_thresh = sq_ast_mod.get_thresholded_df(output_ind_df, config_file["score_threshold"])

    # only run analysis if the thresholded dataframe is non empty
    if len(output_ind_df_thresh) > 0:

        # remove saliency, fbank lengths and sq_ast_pred rows not in thresholded dataframe
        fbank_lengths = fbank_lengths[output_ind_df_thresh["pre_threshold_index"].to_numpy(), :]
        sq_ast_pred = sq_ast_pred[output_ind_df_thresh["pre_threshold_index"].to_numpy(), :]
        saliency_maps = saliency_maps[output_ind_df_thresh["pre_threshold_index"].to_numpy(), :, :, :]

        # run through WhisperX model for ppg and aligned transcriptions
        ppgs_pred, ppgs_dict, word_alignments = ppgs_wrapper.whisperx_get_ppgs(output_ind_df_thresh, config_file)

        pdsm_start_time = datetime.now()

        # get pdsm dict for output dataframe
        pdsm_info = pdsm.PDSM_INFO

        # get kernel density estimate for both time and frequency domain
        kde_x_info = np.zeros(shape=(len(output_ind_df_thresh), len(sq_ast_mod.ALL_DIMS), int(sq_ast_mod.MAX_AUDIO_LEN/sq_ast_mod.SQ_HOP_SIZE)), dtype=np.float32)
        kde_y_info = np.zeros(shape=(len(output_ind_df_thresh), len(sq_ast_mod.ALL_DIMS), sq_ast_mod.SQ_MEL_FREQ), dtype=np.float32)

        # gather information on each thresholded audio file
        for file_idx, df_row in output_ind_df_thresh.iterrows():
            file_path = df_row["file_path"]
            pre_thresh_index = df_row["pre_threshold_index"]

            ppgs_pred_file = ppgs_pred[file_idx, 0, :, :fbank_lengths[file_idx][1]]

            for dim_index in range(len(sq_ast_mod.ALL_DIMS)):
                dim = sq_ast_mod.ALL_DIMS[dim_index]

                if sq_ast_pred[file_idx, dim_index] <= config_file["score_threshold"]:

                    # get saliency maps interpolated up to size
                    saliency_rescaled = sq_ast_mod.get_scaled_saliency_map(
                        saliency_maps[file_idx, dim_index, :, :], 
                        config_file["saliency"]["saliency_interp_method"]
                    )
                    saliency_rescaled = saliency_rescaled.squeeze().squeeze().detach().numpy()
                    saliency_rescaled = saliency_rescaled[:, :fbank_lengths[file_idx][1]]
                    saliency_rescaled = (saliency_rescaled - saliency_rescaled.min()) / (saliency_rescaled.max() - saliency_rescaled.min() + 1e-8)

                    # get the preprocess and pooling functions denoted in config file
                    pdsm_preprocess, pdsm_pool = pdsm.get_preprocess_and_pool(config_file)

                    # run the pdsm algorithm
                    dim_pdsm, dim_phon = pdsm.PDSM(
                        saliency_rescaled, 
                        ppgs_pred_file,
                        ppgs_dict,
                        pdsm_preprocess, 
                        pdsm_pool, 
                        config_file["pdsm"]["k_method"],
                        config_file["pdsm"]["k"]
                    )

                    # save the important phoneme information
                    pdsm_info[f"pdsm_sq_{dim}"].append(dim_phon) # store the phoneme information
                
                    # get the saliency kde for time and frequency dimensions
                    kde_x, kde_y = kde_tools.get_kde_from_saliency(saliency_rescaled, config_file["kde"]["bw_method"])
                    kde_x_info[file_idx, dim_index, :len(kde_x)] = kde_x
                    kde_y_info[file_idx, dim_index, :] = kde_y

                    # save the output images for scores that don't meet the threshold
                    if (config_file["plots"]["output_pdsm_saliency_overlay"]) or (config_file["plots"]["output_joint_kde"]):

                        # get spectrogram from original sq_ast dataset
                        _, mel_spec = sq_ast_ds.__getitem__(pre_thresh_index)

                        # save pdsm overlay
                        if (config_file["plots"]["output_pdsm_saliency_overlay"]):
                            plot_helper.plot_saliency_with_pdsm(
                                config_file, pre_thresh_index, file_path,
                                mel_spec, saliency_rescaled, fbank_lengths[file_idx][1], 
                                dim, dim_phon, dim_pdsm, sq_ast_pred[file_idx, dim_index], 
                                output_individual_dir
                            )

                        # save kde overlay with saliency and spectrogram
                        if (config_file["plots"]["output_joint_kde"]):
                            plot_helper.plot_saliency_jointgrid(
                                config_file, pre_thresh_index, file_path,
                                mel_spec, saliency_rescaled, kde_x_info[file_idx, dim_index, :len(kde_x)], kde_y_info[file_idx, dim_index, :], 
                                dim, word_alignments[file_idx], sq_ast_pred[file_idx, dim_index], 
                                output_individual_dir
                            )

                else:
                    # store dummy phoneme information to save lengths
                    pdsm_info[f"pdsm_sq_{dim}"].append([]) 
            
            # save kde over time
            if (config_file["plots"]["output_time_kde"]):
                plot_helper.plot_kde_along_waveform(
                    config_file, pre_thresh_index, file_path,
                    kde_x_info[file_idx, :, :fbank_lengths[file_idx][1]], word_alignments[file_idx], 
                    sq_ast_mod.ALL_DIMS, sq_ast_pred[file_idx, :],
                    output_individual_dir, 
                )

            # save asr confidence for each word in transcription
            if (config_file["plots"]["output_asr_confidence"]):
                plot_helper.plot_asr_confidence_along_waveform(
                    config_file, file_path, pre_thresh_index,
                    word_alignments[file_idx], output_individual_dir
                )

        # save system level frequency kde
        if (config_file["plots"]["output_freq_kde"]):
            plot_helper.plot_kde_for_freq_sys(
                config_file, kde_y_info, sq_ast_mod.ALL_DIMS,
                output_sysfig_dir
            )

        print(f"PDSM/KDE processing completed in time: {datetime.now() - pdsm_start_time}")

        for dim in sq_ast_mod.ALL_DIMS:

            # Save most important phonemes
            output_ind_df_thresh[f"pdsm_sq_{dim}"] = pdsm_info[f"pdsm_sq_{dim}"]
            output_ind_df_thresh[f"pdsm_sq_{dim}_num"] = len(pdsm_info[f"pdsm_sq_{dim}"])

            if (config_file["plots"]["output_phoneme_hist"]) or (config_file["plots"]["output_double_phoneme_hist"]):
                # get threshold for current dimension
                thresholded_df = output_ind_df_thresh[output_ind_df_thresh[f"sq_{dim}"] <= config_file["score_threshold"]]

                # retrieve histograms on single and double phonemes
                single_hist, double_hist = pdsm.get_bulk_hists_for_system(thresholded_df, dim)
                plot_helper.plot_phoneme_hists(
                    config_file, dim,
                    single_hist, double_hist, 
                    output_sysfig_dir
                )
    
        output_ind_df_thresh.to_csv(output_ind_csv_path_thresh, index=False, sep="\t")

    output_ind_df.to_csv(output_ind_csv_path, index=False, sep="\t")

    with open(f'{output_dir}/config_used.yaml', 'w') as outfile:
        yaml.dump(config_file, outfile)

    print(f"Completed analysis in {str((datetime.now() - start_time))}")

    return 0


if __name__ == "__main__":
    
    # Overwrite settings from default if given
    config_file = {}

    config_path = "configs/default.yaml"
    with open(config_path, 'r') as f:
        config_file = yaml.safe_load(f)

    # load config file and overwrite any settings given
    if len(sys.argv) > 1:
        config_file_alt = {}
        config_path_alt = "configs/" + sys.argv[1] + ".yaml"
        with open(config_path_alt, 'r') as f:
            config_file_alt = yaml.safe_load(f)
        config_file = config_util.deep_merge(config_file, config_file_alt)

    # get gpu availability
    if torch.cuda.is_available():
        config_file["device"] = "cuda"
    else:
        config_file["device"] = "cpu"

    run_eval(config_file)

