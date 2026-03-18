import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scipy import io

def plot_phoneme_hists(single_hist, double_hist, dataset_name, output_sysfig_dir, dim, thresh_val):
    if single_hist is not None:
        plt.figure(figsize=(15, 5))
        plt.bar(single_hist.keys(), single_hist.values(), color='skyblue')
        plt.ylabel('Proportion of Important Phonemes')
        plt.title(f'{dataset_name}: Proportions of Phonemes Deemed Important at Threshold Value {thresh_val} for {dim}')
        plt.savefig(os.path.join(output_sysfig_dir, f"sys_single_hist_{dim}.png"))
        plt.clf()
        plt.close()

    if double_hist is not None:
        plt.figure(figsize=(15, 5))
        plt.bar(double_hist.keys(), double_hist.values(), color='skyblue')
        plt.ylabel('Proportion of Important Phonemes')
        plt.title(f'{dataset_name}: Proportions of Phoneme-Pairs Deemed Important at Threshold Value {thresh_val} for {dim}')
        plt.savefig(os.path.join(output_sysfig_dir, f"sys_double_hist_{dim}.png"))
        plt.clf()
        plt.close()

    return 0

def plot_saliency_with_pdsm(
        file_idx, mel_spec, dim_pdsm, fbank_length, 
        dim_phon, config_file, file_path, dim, sq_ast_pred_i, 
        output_saliency_dir, attn_rescaled
    ):

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(mel_spec.T, aspect='auto', origin='lower', cmap='gray')
    plt.imshow(dim_pdsm, alpha=0.6, aspect='auto', origin='lower', cmap='turbo')
    plt.xlim(0, fbank_length)
    plt.ylim(0, 128)

    y_offset_index = 0
    for phon in dim_phon:
        y_offset =  128*0.9 - 128*0.1*(y_offset_index%8)
        plt.text((phon[2]+phon[3])*0.5, y_offset, phon[1], fontdict={"fontsize":6, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"})
        y_offset_index += 1
    plt.xlabel("Time in 10ms Frames")
    plt.ylabel("Mel Frequency Bin")

    if config_file["pdsm"]["k_method"] == "threshold":
        plt.title(f"File: \'{file_path}\', SQ_AST ({dim} score): {np.round(sq_ast_pred_i, 1):.1f} (With the {config_file["pdsm"]["k"]:.0f} Most Important Phonemes Highlighted)")
    elif config_file["pdsm"]["k_method"] == "percent":
        plt.title(f"File: \'{file_path}\', SQ_AST ({dim} score): {np.round(sq_ast_pred_i, 1):.1f} (With the {100*config_file["pdsm"]["k"]:.0f}% Most Important Phonemes Highlighted)")

    plt.subplot(2, 1, 2)
    plt.imshow(attn_rescaled, alpha=0.6, aspect='auto', origin='lower', cmap='jet')
    plt.xlim(0, fbank_length)
    plt.ylim(0, 128)
    plt.title(f"Attention Rollout for {dim}")
    plt.xlabel("Time in 10ms Frames")
    plt.ylabel("Mel Frequency Bin")
    plt.savefig(os.path.join(output_saliency_dir, f"Phoneme_{file_idx}_{dim}.png"))
    plt.clf()
    plt.close()

    return 0


def plot_sys_violin_plot(config_file, dims, output_ind_df, output_sysfig_dir):

    plt.figure(figsize=(8, 8))

    plot_data = [output_ind_df[f"sq_{dim}"] for dim in dims]

    parts = plt.violinplot(plot_data, positions=range(len(dims)), 
                        showmeans=True, showmedians=False, showextrema=True)

    for pc in parts['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.3)

    plt.axhline(y=config_file["score_threshold"], color="red", 
                linestyle="--", alpha=0.5, label="Threshold")
    
    plt.xticks(range(len(dims)), labels=dims)
    plt.xlabel("Sound Quality Output Categories")
    plt.ylabel("Score (1-5)")
    plt.xlim(-0.5, len(dims) - 0.5)
    plt.ylim(0, 5.5)
    plt.title(f"Violin Plot for SQ_AST Outputs Categories Over {config_file['dataset_name']} Dataset")
    plt.legend()

    plt.savefig(os.path.join(output_sysfig_dir, "sq_ast_violin.png"))
    plt.clf()
    plt.close()

    return 0


def plot_saliency_jointgrid(
        config_file, saliency_map, spectrogram, kde_x_est, kde_y_est, result_word_alignment, 
        file_idx, file_path, dim, sq_ast_pred_i, output_saliency_dir
    ):

    h, w = saliency_map.shape
    x_flat = np.arange(w)
    y_flat = np.arange(h)

    g = sns.JointGrid(height=5, ratio=5, space=0.1)
    g.fig.set_size_inches(15, 5)
    g.ax_joint.imshow(spectrogram.T, aspect='auto', cmap='gray', origin='lower')
    g.ax_joint.imshow(saliency_map, aspect='auto', cmap='jet', origin='lower', alpha=0.1)
    g.ax_joint.set_xticks(
        [i/(2*0.01) for i in range(int(np.ceil(spectrogram.shape[1]*2*0.01)))], 
        [i/2 for i in range(int(np.ceil(spectrogram.shape[1]*2*0.01)))]
    )
    g.ax_joint.set_xlabel("Time in Seconds")
    g.ax_joint.set_ylabel("Mel Frequency Bins")
    
    last_end = 0
    for word_segment in result_word_alignment:
        if int(word_segment["start"]/0.01) != last_end:
            g.ax_joint.plot(
                [int(word_segment["start"]/0.01), int(word_segment["start"]/0.01)], 
                [0, 128], 
                c="black", 
                alpha=0.5
            )
        g.ax_joint.plot(
            [int(word_segment["end"]/0.01), int(word_segment["end"]/0.01)], 
            [0, 128], 
            c="black", 
            alpha=0.5
        )
        last_end = int(word_segment["end"]/0.01)
        g.ax_joint.text(
            (int(word_segment["start"]/0.01) + int(word_segment["end"]/0.01))*0.5, 
            128*0.95, 
            word_segment["word"], 
            fontdict={"fontsize":7, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"}
        )

    g.ax_marg_x.plot(x_flat, kde_x_est)
    g.ax_marg_y.plot(kde_y_est, y_flat)

    g.ax_joint.set_xlim(0, w)
    g.ax_joint.set_ylim(0, h)
    if config_file["pdsm"]["k_method"] == "threshold":
        g.fig.suptitle(
            f"File: \'{file_path}\', SQ_AST ({dim} score): {np.round(sq_ast_pred_i, 1):.1f} with KDE for Time/Freq", 
            fontsize=16, y=1.03
        )
    elif config_file["pdsm"]["k_method"] == "percent":
        g.fig.suptitle(
            f"File: \'{file_path}\', SQ_AST ({dim} score): {np.round(sq_ast_pred_i, 1):.1f} with KDE for Time/Freq", 
            fontsize=16, y=1.03
        )

    plt.savefig(os.path.join(output_saliency_dir, f"KDE_Word_{file_idx}_{dim}.png"), bbox_inches='tight')
    plt.close()

    return 0


def plot_kde_along_waveform(
        config_file, time_kdes, all_dims, wav_path, result_word_alignment, 
        sq_ast_scores, output_saliency_dir, file_idx
    ):

    fs_test, audio_test = io.wavfile.read(wav_path)
    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [0.3, 1]}, figsize=(15, 5))
    
    for dim_index in range(len(all_dims)):
        dim = all_dims[dim_index]
        if sq_ast_scores[dim_index] <= config_file["score_threshold"]:
            time_kde_rescaled = np.interp(
                np.linspace(0, len(time_kdes[dim_index]), len(audio_test)), 
                np.linspace(0, len(time_kdes[dim_index]), len(time_kdes[dim_index])), 
                time_kdes[dim_index]
            )
            time_kde_rescaled = time_kde_rescaled/np.max(time_kde_rescaled)
            axs[0].plot(time_kde_rescaled, label=dim)
    
    axs[0].set_xlim(0, len(time_kde_rescaled))
    axs[0].set_xticks([0], [None])
    axs[0].axis("off")
    axs[0].set_yticks([0], [None])
    axs[0].set_ylim(0, 1.05) # extra 5%

    last_end = 0
    for word_segment in result_word_alignment:
        if word_segment["start"]*fs_test != last_end:
            axs[0].plot(
                [word_segment["start"]*fs_test, word_segment["start"]*fs_test], 
                [0, np.max(time_kde_rescaled)*1.05], 
                c="black", 
                alpha=0.5
            )
        axs[0].plot(
            [word_segment["end"]*fs_test, word_segment["end"]*fs_test], 
            [0, np.max(time_kde_rescaled)*1.05], 
            c="black", 
            alpha=0.5
        )
        last_end = word_segment["end"]*fs_test

    axs[0].legend(
        bbox_to_anchor=[0.0, 0.0], loc='bottom left', 
        fontdict={"fontsize":7, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"}
    )

    axs[1].plot(audio_test)

    last_end = 0
    for word_segment in result_word_alignment:
        if word_segment["start"]*fs_test != last_end:
            axs[1].plot(
                [word_segment["start"]*fs_test, word_segment["start"]*fs_test], 
                [-1, +1], 
                c="black", 
                alpha=0.5
            )
        axs[1].plot(
            [word_segment["end"]*fs_test, word_segment["end"]*fs_test], 
            [-1, +1], 
            c="black", 
            alpha=0.5
        )
        last_end = word_segment["end"]*fs_test

        axs[1].text(
            (word_segment["start"]*fs_test+word_segment["end"]*fs_test)*0.5, 
            1.1, 
            word_segment["word"], 
            fontdict={"fontsize":7, "color":"white", "backgroundcolor":"black", "horizontalalignment":"center"}
        )

    axs[1].set_ylim(-1, 1)
    axs[1].set_xlim(0, len(audio_test))
    axs[1].set_xticks([i*fs_test/2 for i in range(int(len(audio_test)*2/fs_test))], [i/2 for i in range(int(len(audio_test)*2/fs_test))])
    axs[1].set_xlabel("Time in seconds")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_yticks([0], [None])

    plt.suptitle(f"File: {wav_path}, Importance over Time For Sound Quality Metrics")

    plt.savefig(os.path.join(output_saliency_dir, f"KDE_Time_Word_{file_idx}.png"), bbox_inches='tight')
    plt.close()

