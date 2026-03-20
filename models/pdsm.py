import numpy as np


PDSM_INFO = {
    "pdsm_sq_mos":[],
    "pdsm_sq_noi":[],
    "pdsm_sq_dis":[],
    "pdsm_sq_col":[],
    "pdsm_sq_loud":[],
}
THRESHOLD_VALUE = 0.4
COUNT_AS_DOUBLE_PAD = 5 # 50 ms


def thresh_abs(x):
    out_x = np.abs(x)
    out_x[out_x < THRESHOLD_VALUE] = 0.0
    return out_x


def l2_norm(x):
    return np.power(np.sum(np.power(x, 2)), 1/2)


def get_preprocess_and_pool(config_file):
    """
    Returns the preprocess and pooling functions wanted from the config file

    Parameters
    ----------
    config_file (dict) : input config file detailing the options to run the evaluation with

    Returns
    ----------
    preprocess_fn : Function determining the preprocess function as detailed in paper
    pool_fn : Function determining the pooling function as detailed in paper
    """
    
    if config_file["pdsm"]["preprocess"] == "abs":
        pdsm_preprocess = np.abs
    elif config_file["pdsm"]["preprocess"] == "thresh_abs":
        pdsm_preprocess = thresh_abs

    if config_file["pdsm"]["pool"] == "mean":
        pdsm_pool = np.mean
    elif config_file["pdsm"]["pool"] == "sum":
        pdsm_pool = np.sum
    elif config_file["pdsm"]["pool"] == "l2_norm":
        pdsm_pool = l2_norm
    
    return pdsm_preprocess, pdsm_pool


def PDSM(saliency_map, ppg, ppg_dict, preprocess_fn, pool_fn, k_method, k):
    """
    Runs PDSM algorithm as described in [1]S. Gupta, M. Ravanelli, P. Germain, and C. Subakan, “Phoneme Discretized Saliency Maps for Explainable Detection of AI-Generated Voice,” Sep. 2024, [Online]. Available: http://arxiv.org/abs/2406.10422

    Parameters
    ----------
    saliency_map (numpy.array) : Saliency map for a given input to the SQ_AST model
    ppg (numpy.array) : Phoneme PosteriorGram showing the Confidence of phonemes over time
    ppg_dict (dict) : Dictionary showing the translation of indeces to phonemes
    preprocess_fn : Function determining the preprocess function as detailed in paper
    pool_fn : Function determining the pooling function as detailed in paper
    k_method (str) : "threshold" or "percent" for k most important phonemes
    k (float) : if threshold, k is the number of important phonemes, 
        else it is a ratio of the number of phonemes in utterance

    Returns
    ----------
    m_out (numpy.array) : Output mask of most important phonemes
    phonemes_return (list) : List of dicts with information on each important phoneme
    """

    # preprocess step
    m_pp = preprocess_fn(saliency_map)

    # get max phoneme certainty at every time interval
    max_ppg = np.argmax(ppg, axis=0)

    # make list of phonemes id, starts and end times
    phonemes = []
    prev_phon_id = max_ppg[0]
    prev_phon_start = 0
    for t in range(len(max_ppg)):
        if (max_ppg[t] != prev_phon_id):
            phonemes.append((prev_phon_id, ppg_dict[str(prev_phon_id)], prev_phon_start, t-1))
            prev_phon_id = max_ppg[t]
            prev_phon_start = t

    phonemes.append((prev_phon_id, ppg_dict[str(prev_phon_id)], prev_phon_start, len(max_ppg)-1))

    # gather pooled energy for each phoneme
    phoneme_energy = np.zeros(shape=(len(phonemes)), dtype=np.float32)

    index = 0
    for _, _, start, end in phonemes:
        phoneme_energy[index] = pool_fn(m_pp[:, start:end])
        index += 1

    # get "k" max indices from pooled energy per phoneme
    if k_method == "threshold":
        max_indices = np.argsort(phoneme_energy)[-k:]
    elif k_method == "percent":
        k_percent = int(len(phoneme_energy) * k)
        max_indices = np.argsort(phoneme_energy)[-k_percent:]

    # initialise discretised output array
    m_out = np.zeros_like(saliency_map)

    # check whether the phonemes are in the max indeces array
    index = 0
    phonemes_return = []
    for _, _, start, end in phonemes:
        if index in max_indices:
            m_out[:, start:end] = 1
            phonemes_return.append(phonemes[index])
        index += 1

    return m_out, phonemes_return


def get_bulk_hists_for_system(df, dim):
    """
    Gets the system-wide histograms for most important phonemes

    Parameters
    ----------
    df (pandas.dataframe) : Input dataframe to gather hist data
    dim (str) : Dimension to run analysis on
    
    Returns
    ----------
    single_phoneme_hist (dict) : Histogram data for single phonemes (returns None if empty)
    double_phoneme_hist (dict) : Histogram data for phonemes-pairs (returns None if empty)
    """

    single_phoneme_hist = {}
    double_phoneme_hist = {}

    for index, row in df.iterrows():
        phoneme_info = row[f"pdsm_sq_{dim}"]

        for index in range(len(phoneme_info)):
            phoneme = phoneme_info[index]
            if phoneme[1] not in single_phoneme_hist.keys():
                single_phoneme_hist[phoneme[1]] = 1
            else:
                single_phoneme_hist[phoneme[1]] += 1

            if index < len(phoneme_info) - 1:
                next_phoneme = phoneme_info[index+1]
                if phoneme[3] == next_phoneme[2] - COUNT_AS_DOUBLE_PAD:
                    phoneme_transition = phoneme[1] + "-" + next_phoneme[1]
                    if phoneme_transition not in double_phoneme_hist.keys():
                        double_phoneme_hist[phoneme_transition] = 1
                    else:
                        double_phoneme_hist[phoneme_transition] += 1

    # get proportions rather than quantities
    single_phoneme_hist_total = sum(single_phoneme_hist.values())
    if single_phoneme_hist_total > 0:
        single_phoneme_hist = {k: v / single_phoneme_hist_total for k, v in single_phoneme_hist.items()}
        single_phoneme_hist = dict(sorted(single_phoneme_hist.items(), key=lambda item: item[1], reverse=True))
    else:
        single_phoneme_hist = None

    double_phoneme_hist_total = sum(double_phoneme_hist.values())
    if double_phoneme_hist_total > 0:
        double_phoneme_hist = {k: v / double_phoneme_hist_total for k, v in double_phoneme_hist.items()}
        double_phoneme_hist = dict(sorted(double_phoneme_hist.items(), key=lambda item: item[1], reverse=True))
    else:
        double_phoneme_hist = None

    return single_phoneme_hist, double_phoneme_hist

