import numpy as np
import librosa

DEFAULT_HOP_LENGTH = 160
DEFAULT_N_FFT = 2048
DEFAULT_N_MELS = 4
DEFAULT_FMIN = 0
DEFAULT_FMAX = 1000


def get_log_melspectrogram(audio, sr = 16000, hop_length = DEFAULT_HOP_LENGTH, win_length = 400, n_fft = DEFAULT_N_FFT, n_mels = DEFAULT_N_MELS, fmin = DEFAULT_FMIN, fmax = DEFAULT_FMAX, normalize = False):
    """Return the log-scaled Mel bands of an audio signal."""
    normalized_audio = librosa.util.normalize(audio) if normalize else audio
    bands = librosa.feature.melspectrogram(
        y=normalized_audio, sr=sr, hop_length=hop_length, win_length = win_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)
    return librosa.core.power_to_db(bands, amin=1e-7)