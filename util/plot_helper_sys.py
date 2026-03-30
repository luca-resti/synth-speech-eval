import matplotlib
import matplotlib.style as mplstyle
# stop runtime errors and speed up plotting
matplotlib.use('Agg')
mplstyle.use('fast')

import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

MAX_PHONEME_PLOT = 10

sq_ast_dim_str = {
    "mos" : "MOS",
    "noi" : "Noisiness", 
    "dis" : "Discontinuity", 
    "col" : "Colouration", 
    "loud" : "Loudness"
}

# SYS plots

def plot_sys_violin_plot(
        config_file, output_df, all_dims, output_dir
    ):
    '''
    Plots a figure of the distribution of the whole dataset along all SQ_AST dimensions (with threshold)

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    output_df (pandas.dataframe) : dataset containing the output scores from SQ_AST across all dimensions
    all_dims (list) : SQ_AST dimensions to run
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''

    plt.figure(figsize=(8, 8))

    plot_data = [output_df[f"sq_{dim}"] for dim in all_dims]

    if len(output_df) == 1:
        plt.scatter(range(len(all_dims)), plot_data, s=25, c="blue", marker="x")
    else:
        parts = plt.violinplot(plot_data, positions=range(len(all_dims)), 
                        showmeans=True, showmedians=False, showextrema=True)

        for pc in parts['bodies']:
            pc.set_facecolor('blue')
            pc.set_edgecolor('black')
            pc.set_alpha(0.3)

    plt.axhline(y=config_file["score_threshold"], color="red", 
                linestyle="--", alpha=0.5, label="Threshold")
    
    label_dims = [sq_ast_dim_str[dim] for dim in all_dims]
    plt.xticks(range(len(all_dims)), labels=label_dims)
    plt.xlabel("Sound Quality Output Categories")
    plt.ylabel("Score (1-5)")
    plt.xlim(-0.5, len(all_dims) - 0.5)
    plt.ylim(0, 5.5)
    
    if len(output_df) == 1:
        plt.title(f"{config_file['dataset_name']}: Scatter Plot\nfor Sound Quality Outputs Categories", fontsize=14)
    else:
        plt.title(f"{config_file['dataset_name']}: Violin Plot\nfor Sound Quality Categories Over Dataset", fontsize=14)

    plt.legend(loc='lower right')

    plt.savefig(os.path.join(output_dir, "sq_violin.png"), dpi=100)
    plt.clf()
    plt.close()

    return 0


def plot_sys_bar_chart(
        config_file, output_df, all_dims, output_dir
    ):
    '''
    Plots a bar chart of percentages of samples that are below the threshold

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    output_df (pandas.dataframe) : dataset containing the output scores from SQ_AST across all dimensions
    all_dims (list) : SQ_AST dimensions to run
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''

    plt.figure(figsize=(8, 8))

    label_dims = [sq_ast_dim_str[dim] for dim in all_dims]
    plot_data = np.array([len(output_df[f"sq_{dim}"][output_df[f"sq_{dim}"] <= config_file["score_threshold"]]) for dim in all_dims])
    plot_data = plot_data/len(output_df)
    plt.bar(range(len(all_dims)), plot_data, color='skyblue')

    plt.xticks(range(len(all_dims)), labels=label_dims)
    plt.xlabel("Sound Quality Output Categories")
    plt.ylabel("Percentage of Dataset")
    plt.xlim(-0.5, len(all_dims) - 0.5)
    plt.ylim(0, 1.0)
    plt.yticks([0.1*i for i in range(11)], [str(10*i) + "%" for i in range(11)], )
    
    plt.title(f"{config_file['dataset_name']}: Percentage of Segments\nUnder Threshold for Categories Over Dataset", fontsize=14)

    plt.savefig(os.path.join(output_dir, "perc_under_thresh.png"), dpi=100)
    plt.clf()
    plt.close()

    return 0

    
def plot_sys_chan_bar_chart(
        config_file, output_df, max_chans, output_dir
    ):
    '''
    Plots a bar chart of percentages of samples that are below the threshold

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    output_df (pandas.dataframe) : dataset containing the output scores from SQ_AST across all dimensions
    max_chans (int) : maximum number of wav channels in dataset
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''


    if max_chans > 1:

        plt.figure(figsize=(8, 8))

        hist_info = {str(chan) : 0 for chan in range(max_chans)}

        for chan in range(max_chans):
            hist_info[str(chan)] = len(output_df["wav_channel"][output_df["wav_channel"] == chan])/len(output_df)

        plt.bar(list(hist_info.keys()), list(hist_info.values()), color='skyblue')

        plt.xticks(range(max_chans), labels=range(max_chans))
        plt.xlabel("Wav Channel")
        plt.ylabel("Percentage of Dataset")
        plt.xlim(-0.5, max_chans - 0.5)
        plt.ylim(0, 1.0)
        plt.yticks([0.1*i for i in range(11)], [str(10*i) + "%" for i in range(11)], )
        
        plt.title(f"{config_file['dataset_name']}: Percentage of Wav Channel\nNumber in Thresholded Dataset", fontsize=14)

        plt.savefig(os.path.join(output_dir, "wav_chan_perc.png"), dpi=100)
        plt.clf()
        plt.close()

    return 0


def plot_sys_asr_conf_violin_plot(
        config_file, output_df, output_dir
    ):
    '''
    Plots a violin plot of the distribution of the mean and median asr confidence per segment

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    output_df (pandas.dataframe) : thresholded dataset containing the asr confidence scores
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''

    plt.figure(figsize=(6, 8))

    plot_data = [output_df[f"mean_asr_conf"], output_df[f"median_asr_conf"]]

    if len(output_df) == 1:
        plt.scatter(range(2), plot_data, s=25, c="blue", marker="x")
    else:
        parts = plt.violinplot(plot_data, positions=range(2), 
                            showmeans=True, showmedians=False, showextrema=True)

        for pc in parts['bodies']:
            pc.set_facecolor('blue')
            pc.set_edgecolor('black')
            pc.set_alpha(0.3)
    
    plt.xticks(range(2), labels=["Segment Mean", "Segment Median"])
    plt.xlabel("ASR Average Type")
    plt.ylabel("Confidence")
    plt.xlim(-0.5, 2 - 0.5)
    plt.ylim(0, 1.0)
    plt.yticks([0.1*i for i in range(11)], [str(10*i)+"%" for i in range(11)])
    
    if len(output_df) == 1:
        plt.title(f"{config_file['dataset_name']}: Scatter Plot for Average ASR\nConfidence per Segment for Data under Threshold", fontsize=14)
    else:
        plt.title(f"{config_file['dataset_name']}: Violin Plot for Average ASR\nConfidence per Segment For Data Under Threshold", fontsize=14)
        
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(output_dir, "asr_conf_violin.png"), dpi=100)
    plt.clf()
    plt.close()

    return 0


def plot_phoneme_hists(
        config_file, dim,
        single_hist, double_hist, 
        output_dir
    ):
    '''
    Plots histograms of the occurance of each phoneme (and phoneme pairs) deemed important on a system level

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    dim (str) : Dimension to show saliency for
    single_phoneme_hist (dict) : Histogram data for single phonemes (None if empty)
    double_phoneme_hist (dict) : Histogram data for phonemes-pairs (None if empty)
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''

    thresh_val = config_file["score_threshold"]
    output_phoneme_hist = config_file["plots"]["output_phoneme_hist"]
    output_double_phoneme_hist = config_file["plots"]["output_double_phoneme_hist"]
    dataset_name = config_file["dataset_name"]

    if (single_hist is not None) and (output_phoneme_hist):
        plt.figure(figsize=(15, 5))
        single_hist_len = len(list(single_hist.keys()))
        if single_hist_len > MAX_PHONEME_PLOT:
            plt.bar(list(single_hist.keys())[:MAX_PHONEME_PLOT], list(single_hist.values())[:MAX_PHONEME_PLOT], color='skyblue')
            plt.title(f'{dataset_name}: Proportions of {MAX_PHONEME_PLOT} most Important Phonemes at Threshold Value {thresh_val} for {sq_ast_dim_str[dim]}')
        else:
            plt.bar(list(single_hist.keys()), list(single_hist.values()), color='skyblue')
            plt.title(f'{dataset_name}: Proportions of {single_hist_len} most Important Phonemes at Threshold Value {thresh_val} for {sq_ast_dim_str[dim]}')
        plt.ylabel('Proportion of Important Phonemes')
        plt.savefig(os.path.join(output_dir, f"sys_single_hist_{dim}.png"), dpi=100)
        plt.clf()
        plt.close()

    if (double_hist is not None) and (output_double_phoneme_hist):
        plt.figure(figsize=(15, 5))
        double_hist_len = len(list(double_hist.keys()))
        if double_hist_len > MAX_PHONEME_PLOT:
            plt.bar(list(double_hist.keys())[:MAX_PHONEME_PLOT], list(double_hist.values())[:MAX_PHONEME_PLOT], color='skyblue')
            plt.title(f'{dataset_name}: Proportions of {MAX_PHONEME_PLOT} most Important Phoneme-Pairs at Threshold Value {thresh_val} for {sq_ast_dim_str[dim]}', fontsize=14)
        else:
            plt.bar(list(double_hist.keys()), list(double_hist.values()), color='skyblue')
            plt.title(f'{dataset_name}: Proportions of {double_hist_len} most Important Phoneme-Pairs at Threshold Value {thresh_val} for {sq_ast_dim_str[dim]}', fontsize=14)
        plt.ylabel('Proportion of Important Phonemes')

        plt.savefig(os.path.join(output_dir, f"sys_double_hist_{dim}.png"), dpi=100)
        plt.clf()
        plt.close()

    return 0


def plot_kde_for_freq_sys(
        config_file, kde_freq, all_dims, 
        output_dir
    ):
    '''
    Plots a figure of Frequency Kernel Density Estimation over the entire system under threshold

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    kde_freq (numpy.array) : Kernel Density Estimation for mel-frequency bins across all dimensions
    all_dims (list) : SQ_AST dimensions to run
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''
    
    if np.any(kde_freq):
        plt.figure(figsize=(10, 5))
        max_val = 0
        for dim_index in range(len(all_dims)):
            temp_agg = np.sum(kde_freq[:, dim_index, :], axis=0)
            if np.any(temp_agg):
                temp_agg = temp_agg/np.sum(temp_agg)
                if np.max(temp_agg) > max_val:
                    max_val = np.max(temp_agg)
                plt.plot(temp_agg, label=all_dims[dim_index])
        plt.legend(loc='upper right')
        plt.xlabel("Mel Frequency Bin")
        plt.xlim(0, 128)
        plt.ylim(0, max_val*1.05)
        plt.title(f"{config_file["dataset_name"]}: KDE Aggregate Frequency Importance For System", fontsize=14)
        plt.savefig(os.path.join(output_dir, f"frequency_KDE_for_dims.png"), dpi=100)
        plt.clf()
        plt.close()

    return 0

def plot_metric_corr_plot(
       config_file, output_df, all_dims, is_thresholded,
       output_dir
):
    
    '''
    Plots a figure of Frequency Kernel Density Estimation over the entire system under threshold

    Parameters
    ----------
    config_file (dict) : Config file read in by yaml, see ./configs/default.yaml for a more in-depth understanding
    output_df (pandas.dataframe) : dataset containing the output scores from SQ_AST across all dimensions
    all_dims (list) : SQ_AST dimensions to run
    is_thresholded (bool) : If the dataframe is thresholded, can see both full dataset and thresholded
    output_dir (os.path) : output directory for figure

    Returns
    ----------
    0 : 
    '''
    df_to_plot = output_df[[f"sq_{dim}" for dim in all_dims]]
    plot_columns = {}
    for dim in all_dims:
        plot_columns[f"sq_{dim}"] = sq_ast_dim_str[dim]
    df_to_plot = df_to_plot.rename(columns=plot_columns)

    plt.figure(figsize=(15, 12))
    g = sns.heatmap(df_to_plot.corr(), annot=True)

    if is_thresholded:
        plt.suptitle(f"{config_file["dataset_name"]}: Correlation Between\nthe Metrics over Thresholded Dataset", fontsize=14)
        plt.savefig(os.path.join(output_dir, f"sq_metric_corr_thresh.png"), dpi=100, bbox_inches='tight')
    else:
        plt.suptitle(f"{config_file["dataset_name"]}: Correlation Between Between\nthe Metrics over Whole Dataset", fontsize=14)
        plt.savefig(os.path.join(output_dir, f"sq_metric_corr.png"), dpi=100, bbox_inches='tight')
    plt.clf()
    plt.close()

    return 0
