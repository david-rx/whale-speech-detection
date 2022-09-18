import numpy as np
import librosa

def get_log_melspectrogram(audio, sr = 16000, hop_length = 160, win_length = 400, n_fft = 512, n_mels = 64, fmin = 125, fmax = 7500):
    """Return the log-scaled Mel bands of an audio signal."""
    bands = librosa.feature.melspectrogram(
        y=audio, sr=sr, hop_length=hop_length, win_length = win_length, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, dtype=np.float32)
    return librosa.core.power_to_db(bands, amin=1e-7)