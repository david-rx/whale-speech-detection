import numpy as np
from torch.data import Dataset
import torch
from typing import List
from dataclasses import dataclass
import soundfile as sf
import os

from whale_speech_detection.data.yoho import Event, Yoho


BEGIN_FILE = 3
END_FILE = 4
BEGIN_TIME = 5
END_TIME = 6

WINDOW_LENGTH = 0.3 #seconds

class CallDataset(Dataset):

    def __init__(self, yolo_chunks: List[np.ndarray], yolo_labels) -> None:
        self.yolo_chunks = torch.from_numpy(np.stack(yolo_chunks))
        self.yolo_labels = torch.from_numpy(yolo_labels)
    
    def __len__(self):
        return len(self.wav_files) #what is a single example? Treat as NER (?)

    def __getitem__(self, idx):
        return self.yolo_chunks[idx], self.yolo_labels[idx]

@dataclass
class CallLabels:
    call_starts: List
    call_ends: List
    file_starts: List
    file_ends: List
    call_types: List


def load_file_to_labels(filename) -> CallLabels:
    """
    Processes a single file of a specific label type and returns the corresponding labels
    """
    label_type = filename.split('.')[1]
    call_starts = []
    call_ends = []
    file_starts = []
    file_ends = []
    call_types = []
    with open(filename) as file:
        labels_file = file.readlines()
    for line in labels_file:
        columns = line.split('  ')
        call_starts.append(columns[BEGIN_TIME])
        call_ends.append(columns[END_TIME])
        file_starts.append(columns[BEGIN_FILE])
        file_ends.append(columns[END_FILE])
        call_types.append(label_type)
    return CallLabels(call_starts=call_starts, call_ends=call_ends, file_starts=file_starts, file_ends=file_ends, call_types=call_types)

def load_dataset(site_dir) -> None:
    """
    Loads dataset from WAV files and labels
    """
    wav_file_names = os.listdir(os.path.join(site_dir, 'wav'))
    wav_arrays = [sf.read(wav_file_name)[0] for wav_file_name in wav_file_names]
    all_call_labels = []
    for file in os.listdir(site_dir):
        call_labels = load_file_to_labels(file)
        all_call_labels.append(call_labels)
    return _build_dataset(wav_arrays, all_call_labels)

def _call_labels_to_events(call_labels: List[CallLabels]) -> List[Event]:
    return

def _time_to_sample_number(times: List[float], file_numbers: List[int], wav_arrays: np.ndarray,
    sampling_rates: List[float]):
    """
    Convert all times to samples numbers
    """
    wav_file_lengths: List[float] = [len(wav_arrays[i]) * sampling_rates[i] for i in range(len(wav_arrays))]
    cum_wav_file_lengths: List[float] = []
    cum_wav_file_lengths[0] = wav_file_lengths[0]
    sample_numbers: List[int] = []
    for i in range(0, len(wav_file_lengths)):
        cum_wav_file_lengths[i + 1] = cum_wav_file_lengths[i] + wav_file_lengths[i + 1]
    for time, sr, file_number in zip(times, sampling_rates, file_numbers):
        sample_numbers.append(cum_wav_file_lengths[file_number] + int(time * sr))
    return sample_numbers    


def _build_dataset(wav_arrays: np.ndarray, sampling_rates: List[float], all_call_labels: List[CallLabels]) -> CallDataset:
    """
    Constructs the dataset
    """
    starts = [start for call_labels in all_call_labels for start in call_labels.call_starts]
    ends = [end for call_labels in all_call_labels for end in call_labels.call_ends]
    file_starts = [fs for call_labels in all_call_labels for fs in call_labels.file_starts]
    file_ends = [fe for call_labels in all_call_labels for fe in call_labels.file_starts]
    types = [t for call_labels in all_call_labels for t in call_labels]
    sample_starts = _time_to_sample_number(starts, file_starts, wav_arrays, sampling_rates)
    sample_ends = _time_to_sample_number(ends, file_ends, wav_arrays, sampling_rates)
    full_array = wav_arrays.flatten()
    events = [Event(event_type=types[i], event_start=starts[i], event_end=ends[i]) for i in range(len(starts))]
    yoho = Yoho(window_length=sampling_rates[0] * WINDOW_LENGTH, event_types=set(types), total_length=len(full_array))
    yoho_labels = yoho.get_yoho_labels(events)
    num_windows = yoho_labels.shape[0]
    assert num_windows == 1
    yoho_windows = [full_array]
    #yoohoo! got yoho labels and full audio array!s
    
    
    


    


