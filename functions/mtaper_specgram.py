import numpy as np
import scipy.signal as sig
def mtaper_specgram(
    signal,
    nw=2.5,
    ntapers=None,
    win_len=0.1,
    win_overlap=0.09,
    fs=int(192e3),
    clip=None,
    **kwargs
):
    """Multi-taper specgram
    Args:
        signal (array type): Signal.
        nw (float): Time-bandwidth product.
        ntapers (int): Number of tapers (None to set to 2 * nw -1)
        win_len (float): Window length in seconds.
        win_overlap (float): Window overlap in seconds.
        fs (float): Sampling rate.
        clip (2-tuple of floats): Normalize amplitudes to 0-1 using clips (in dB)
        **kwargs: Additional arguments for scipy.signal.spectrogram
    Returns:
        array type: Frequency bin centers
        array type: Time incides
        array type: Specgram
    """
    if ntapers is None:
        ntapers = int(nw * 2)
    len_samples = np.round(win_len * fs).astype("int")
    overlap_samples = np.round(win_overlap * fs)
    sequences, r = sig.windows.dpss(
        len_samples, NW=nw, Kmax=ntapers, sym=False, norm=2, return_ratios=True
    )
    sxx_ls = None
    for sequence, weight in zip(sequences, r):
        f, t, sxx = sig.spectrogram(
            signal,
            fs=fs,
            window=sequence,
            nperseg=len_samples,
            noverlap=overlap_samples,
            nfft=None,
            **kwargs
        )
        if sxx_ls is None:
            sxx_ls = np.abs(sxx * weight)
        else:
            sxx_ls += np.abs(sxx * weight)
    sxx = sxx_ls / len(sequences)
    if clip is not None:
        sxx = 20 * np.log10(sxx)
        sxx = sxx - clip[0]
        sxx[sxx < 0] = 0
        sxx[sxx > (clip[1] - clip[0])] = clip[1] - clip[0]
        sxx /= clip[1] - clip[0]
    return f, t, sxx