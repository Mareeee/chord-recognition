import numpy as np

def compute_framewise_metrics(ref_idx, est_idx, frame_sec):
    """
    Frame-wise CSR/WCSR/Overlap (po definiciji: Overlap = vremenski udeo poklapanja).
    - CSR   (unweighted accuracy): broj tačnih frame-ova / ukupan broj frame-ova
    - WCSR  (weighted by duration): isto kao CSR jer su svi frame-ovi iste dužine,
             ali ostavljeno eksplicitno radi jasnoće.
    - Overlap: ovde isto = WCSR (udio vremena gde su labeli jednaki)

    ref_idx, est_idx: np.array [T] celobrojni indeksi klasa
    frame_sec: trajanje jednog frame-a u sekundama (hop/sr)
    """
    T = min(len(ref_idx), len(est_idx))
    if T == 0:
        return 0.0, 0.0, 0.0

    ref_idx = ref_idx[:T]
    est_idx = est_idx[:T]

    correct = (ref_idx == est_idx)
    csr = float(np.sum(correct)) / float(T)

    # WCSR: ponderisano trajanjem; frame_sec se skraćuje jer je konstantan
    wcsr = float(np.sum(correct) * frame_sec) / float(T * frame_sec)

    # Overlap ratio (vremenski udeo poklapanja)
    overlap = wcsr

    return csr, wcsr, overlap


def intervals_to_frame_labels(intervals, labels, to_index_fn, total_frames):
    """
    Konvertuje ref intervale (u sekundama) + chord string labels -> frame-wise indekse [T].
    to_index_fn: funkcija (str)->int (npr. dataset._chord_to_index)
    """
    ref = np.zeros(total_frames, dtype=np.int64)  # default 'N'=0
    for (start, end), lab in zip(intervals, labels):
        if end <= 0:
            continue
        # uokviri
        # Napomena: ovde ne znamo hop/sr -> prebacivanje vremenskog u frame
        # Radi generičnosti, caller treba da nam da već prekonvertovane frame indekse,
        # ali pošto koristimo ovaj helper u evaluatoru, tamo ćemo prevesti.
        pass  # ovaj helper neće se koristiti direktno (vidi evaluator)
    return ref  # placeholder, ne koristimo direktno
