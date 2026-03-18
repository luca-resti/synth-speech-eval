import matplotlib.pyplot as plt
import os
import numpy as np

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
    plt.savefig(os.path.join(output_saliency_dir, f"{file_idx}_{dim}.png"))
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
