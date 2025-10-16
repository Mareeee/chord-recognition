import numpy as np

def compute_framewise_metrics(ref_idx, est_idx, frame_sec):
    T = min(len(ref_idx), len(est_idx))
    if T == 0:
        return 0.0, 0.0, 0.0

    ref_idx = ref_idx[:T]
    est_idx = est_idx[:T]

    correct = (ref_idx == est_idx)
    csr = float(np.sum(correct)) / float(T)

    wcsr = float(np.sum(correct) * frame_sec) / float(T * frame_sec)

    overlap = wcsr

    return csr, wcsr, overlap
