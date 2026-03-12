import numpy as np

def PDSM(saliency_map, ppg, ppg_dict, preprocess_fn, pool_fn, k_method, k):

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
            phonemes.append((prev_phon_id, ppg_dict[str(prev_phon_id)], prev_phon_start, t))
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