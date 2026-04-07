import logging
import sys

# set logging info
LOGGING_LEVEL = logging.INFO
logger = logging.getLogger()
# get rid of matplotlib warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s | %(message)s')
logger.setLevel(LOGGING_LEVEL)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(LOGGING_LEVEL)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

from models import sq_ast_mod
from models import ppgs_wrapper
from models import pdsm
from util import kde_tools, config_util, audio_segmentation, plot_helper_ind, plot_helper_sys

import threading
import torch
import os
import yaml
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.setrecursionlimit(10**9)
threading.stack_size(10**8)

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
    return_code : 0 if no errors, error message if one raised
    '''
    return_code = 0
    try:
        start_time = config_file["datetime"]

        # create output directories
        output_dir = ""
        if str(config_file["dataset_name"]).endswith(".wav"): 
            dataset_name = os.path.basename(str(config_file["dataset_name"]).replace('.wav',''))
            output_dir_base = config_file["output_dir"] + "/" + dataset_name + "/"
        else:
            output_dir_base = config_file["output_dir"] + "/" + config_file["dataset_name"] + "/"
        if not os.path.exists(output_dir_base):
            os.makedirs(output_dir_base)
        output_dir =  output_dir_base +  start_time.strftime("%Y%m%d_%H%M") + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_sysfig_dir = output_dir + "/system_plots/"
        if not os.path.exists(output_sysfig_dir):
            os.makedirs(output_sysfig_dir)
        output_individual_dir = output_dir + "/" + "individual_plots" + "/"
        if not os.path.exists(output_individual_dir):
            os.makedirs(output_individual_dir)
        output_ind_tsv_path = os.path.join(output_dir, "output_sq_ast.tsv")
        output_ind_tsv_path_thresh = os.path.join(output_dir, "output_thresholded.tsv")
        output_dir_tsv_sorted = os.path.join(output_dir, "sorted_tsvs/")
        if not os.path.exists(output_dir_tsv_sorted):
            os.makedirs(output_dir_tsv_sorted)

        file_handler = logging.FileHandler(output_dir + '/logs.log')
        file_handler.setLevel(LOGGING_LEVEL)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        # get gpu availability
        if torch.cuda.is_available():
            logger.info(f" === USING GPU === ")
            config_file["device"] = "cuda"
        else:
            logger.info(f" === USING CPU === ")
            config_file["device"] = "cpu"

        input_df = audio_segmentation.get_input_dataset(config_file)

        if len(input_df) == 0:
            logger.info("Input Dataset has no valid audio files")
            exit()

        # Validate dims
        config_file["sq_ast_dims"] = sq_ast_mod.sq_ast_validate_dims(config_file["sq_ast_dims"])

        # unpack sq_ast outputs
        sq_ast_ds, sq_ast_pred, saliency_maps, fbank_lengths, output_ind_df = sq_ast_mod.sq_ast_fw(config_file, input_df)

        # plot system violin plots
        if config_file["plots"]["output_sys_violin"]:
            plot_helper_sys.plot_sys_violin_plot(
                config_file, output_ind_df, config_file["sq_ast_dims"], 
                output_sysfig_dir
            )

        # plot system pair plots
        if config_file["plots"]["output_sys_corr"]:
            plot_helper_sys.plot_metric_corr_plot(
                config_file, output_ind_df, config_file["sq_ast_dims"], False,
                output_sysfig_dir
            )

        # threshold dataframe
        output_ind_df_thresh = sq_ast_mod.get_thresholded_df(config_file, output_ind_df)

        # only run analysis if the thresholded dataframe is non empty
        if len(output_ind_df_thresh) > 0:
            # plot system bar chart
            if config_file["plots"]["output_sys_bar"]:
                plot_helper_sys.plot_sys_bar_chart(
                    config_file, output_ind_df, config_file["sq_ast_dims"], 
                    output_sysfig_dir
                )

            if config_file["plots"]["output_sys_corr"]:
                plot_helper_sys.plot_metric_corr_plot(
                    config_file, output_ind_df_thresh, config_file["sq_ast_dims"], True,
                    output_sysfig_dir
                )

            if config_file["plots"]["output_sys_wav_chan"]:
                plot_helper_sys.plot_sys_chan_bar_chart(
                    config_file, output_ind_df_thresh, int(np.max(output_ind_df["total_wav_channels"])), 
                    output_sysfig_dir
                )

            # make output directories
            for index, df_row in output_ind_df_thresh.iterrows():
                individual_dir = output_individual_dir + os.path.basename(df_row["file_path"])[:-4] + "/" #remove ".wav"
                if not os.path.exists(individual_dir):
                    os.makedirs(individual_dir)
                if df_row["total_wav_channels"] > 1:
                    individual_dir = output_individual_dir + os.path.basename(df_row["file_path"])[:-4] + "/" + "ch" + str(df_row["wav_channel"]) + "/"
                    if not os.path.exists(individual_dir):
                        os.makedirs(individual_dir)

            for dim in config_file["sq_ast_dims"]:
                output_ind_df_thresh_for_row = output_ind_df_thresh[output_ind_df_thresh[f"sq_{dim}"] <= config_file["score_threshold"]]
                if len(output_ind_df_thresh_for_row) > 0:
                    output_ind_df_thresh_for_row = output_ind_df_thresh_for_row.sort_values(f"sq_{dim}", ascending=True)
                    output_ind_df_thresh_for_row.to_csv(os.path.join(output_dir_tsv_sorted, f"thredholded_sorted_sq_{dim}.tsv"), index=False, sep="\t")

            # remove saliency, fbank lengths and sq_ast_pred rows not in thresholded dataframe
            fbank_lengths = fbank_lengths[output_ind_df_thresh["pre_threshold_index"].to_numpy(), :]
            sq_ast_pred = sq_ast_pred[output_ind_df_thresh["pre_threshold_index"].to_numpy(), :]
            saliency_maps = saliency_maps[output_ind_df_thresh["pre_threshold_index"].to_numpy(), :, :, :]

            # run through WhisperX model for ppg and aligned transcriptions
            ppgs_pred, ppgs_dict, word_alignments = ppgs_wrapper.whisperx_get_ppgs(output_ind_df_thresh, config_file)

            agg_asr_confidence = np.zeros(shape=(len(output_ind_df_thresh), 2))
            for seg_index, seg_word_alignments in enumerate(word_alignments):
                agg_asr_confidence[seg_index, :] = ppgs_wrapper.get_agg_asr_confidence(seg_word_alignments)
            output_ind_df_thresh["mean_asr_conf"] = agg_asr_confidence[:, 0]
            output_ind_df_thresh["median_asr_conf"] = agg_asr_confidence[:, 1]

            if config_file["plots"]["output_asr_confidence_violin"]:
                plot_helper_sys.plot_sys_asr_conf_violin_plot(
                    config_file, output_ind_df_thresh, output_sysfig_dir
                )

            pdsm_start_time = datetime.now()

            pdsm_info = {
                "pdsm_sq_mos":[],
                "pdsm_sq_noi":[],
                "pdsm_sq_dis":[],
                "pdsm_sq_col":[],
                "pdsm_sq_loud":[],
            }

            # get kernel density estimate for both time and frequency domain
            kde_x_info = np.zeros(shape=(len(output_ind_df_thresh), len(config_file["sq_ast_dims"]), int(sq_ast_mod.SALIENCY_INTERP_SIZE[1])), dtype=np.float32)
            kde_y_info = np.zeros(shape=(len(output_ind_df_thresh), len(config_file["sq_ast_dims"]), sq_ast_mod.SQ_MEL_FREQ), dtype=np.float32)

            # gather information on each thresholded audio file
            for file_idx, df_row in output_ind_df_thresh.iterrows():
                file_tot_channels = df_row["total_wav_channels"]
                file_channel = df_row["wav_channel"]
                pre_thresh_index = df_row["pre_threshold_index"]

                if file_tot_channels == 1:
                    individual_base_folder = output_individual_dir + os.path.basename(df_row["file_path"])[:-4] + "/"
                else:
                    individual_base_folder = output_individual_dir + os.path.basename(df_row["file_path"])[:-4] + "/ch" + str(file_channel) + "/"

                ppgs_pred_file = ppgs_pred[file_idx, 0, :, :fbank_lengths[file_idx][1]]

                for dim_index in range(len(config_file["sq_ast_dims"])):
                    dim = config_file["sq_ast_dims"][dim_index]

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
                                plot_helper_ind.plot_saliency_with_pdsm(
                                    config_file, df_row,
                                    mel_spec, saliency_rescaled, fbank_lengths[file_idx][1], 
                                    dim_index, config_file["sq_ast_dims"], dim_phon, dim_pdsm, 
                                    individual_base_folder
                                )

                            # save kde overlay with saliency and spectrogram
                            if (config_file["plots"]["output_joint_kde"]):
                                plot_helper_ind.plot_saliency_jointgrid(
                                    config_file, df_row,
                                    mel_spec, saliency_rescaled, kde_x_info[file_idx, dim_index, :len(kde_x)], kde_y_info[file_idx, dim_index, :], 
                                    dim_index, config_file["sq_ast_dims"], word_alignments[file_idx],
                                    individual_base_folder
                                )

                    else:
                        # store dummy phoneme information to save lengths
                        pdsm_info[f"pdsm_sq_{dim}"].append([]) 
                
                # save kde over time
                if (config_file["plots"]["output_time_kde"]):
                    plot_helper_ind.plot_kde_along_waveform(
                        config_file, df_row,
                        kde_x_info[file_idx, :, :fbank_lengths[file_idx][1]], word_alignments[file_idx], 
                        config_file["sq_ast_dims"], individual_base_folder
                    )

                # save asr confidence for each word in transcription
                if (config_file["plots"]["output_asr_confidence"]):
                    plot_helper_ind.plot_asr_confidence_along_waveform(
                        config_file, df_row, word_alignments[file_idx],
                        individual_base_folder
                    )

            # save system level frequency kde
            if (config_file["plots"]["output_freq_kde"]):
                plot_helper_sys.plot_kde_for_freq_sys(
                    config_file, kde_y_info, config_file["sq_ast_dims"],
                    output_sysfig_dir
                )

            logger.info(f"PDSM/KDE processing completed in time: {datetime.now() - pdsm_start_time}")

            for dim in config_file["sq_ast_dims"]:

                logger.info(f"pdsm_sq_{dim}: {pdsm_info[f'pdsm_sq_{dim}']}")

                # Save most important phonemes
                output_ind_df_thresh[f"pdsm_sq_{dim}"] = pdsm_info[f"pdsm_sq_{dim}"]
                output_ind_df_thresh[f"pdsm_sq_{dim}_num"] = len(pdsm_info[f"pdsm_sq_{dim}"])

                if (config_file["plots"]["output_phoneme_hist"]) or (config_file["plots"]["output_double_phoneme_hist"]):
                    # get threshold for current dimension
                    thresholded_df = output_ind_df_thresh[output_ind_df_thresh[f"sq_{dim}"] <= config_file["score_threshold"]]

                    # retrieve histograms on single and double phonemes
                    single_hist, double_hist = pdsm.get_bulk_hists_for_system(thresholded_df, dim)
                    plot_helper_sys.plot_phoneme_hists(
                        config_file, dim,
                        single_hist, double_hist, 
                        output_sysfig_dir
                    )
        
            output_ind_df_thresh.to_csv(output_ind_tsv_path_thresh, index=False, sep="\t")

        output_ind_df.to_csv(output_ind_tsv_path, index=False, sep="\t")

        with open(f'{output_dir}/config_used.yaml', 'w') as outfile:
            yaml.dump(config_file, outfile)

        logger.info(f"Completed analysis in {str((datetime.now() - start_time))}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return_code = e

    # remove loggers
    logger.removeHandler(file_handler)

    return return_code


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

    config_file["datetime"] = datetime.now()

    run_eval(config_file)

