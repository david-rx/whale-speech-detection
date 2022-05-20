from typing import Tuple
import random
from random import sample
from turtle import end_fill
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import List
from dataclasses import dataclass
import soundfile as sf
import os
from itertools import chain
import pandas as pd
from datetime import datetime

from whale_speech_detection.data.yoho import Event, Yoho


BEGIN_FILE = 3
END_FILE = 4
BEGIN_TIME = 5
END_TIME = 6
BEGIN_SAMPLE = 7
END_SAMPLE = 8

WINDOW_LENGTH = 0.3 #seconds

WAVFILENAME = 'wavFileName'

class CallDataset(Dataset):

    def __init__(self, yoho_chunks: List[np.ndarray], yoho_labels) -> None:
        self.yoho_chunks = torch.from_numpy(np.stack(yoho_chunks))
        self.yoho_labels = torch.from_numpy(yoho_labels)
        assert len(yoho_chunks) == len(yoho_labels)
    
    def __len__(self):
        return len(self.yoho_chunks) #what is a single example? Treat as NER (?)

    def __getitem__(self, idx):
        return self.yoho_chunks[idx], self.yoho_labels[idx]

    def _chunks_to_mfcc(self):
        pass

@dataclass
class CallLabels:
    call_starts: List
    call_ends: List
    file_starts: List
    file_ends: List
    call_types: List


def load_file_to_labels(filename, selected_wav_files: List[str], offsets) -> CallLabels:
    """
    Processes a single file of a specific label type and returns the corresponding labels
    """
    label_type = filename.split('.')[1]
    print(filename)
    call_starts = []
    call_ends = []
    file_starts = []
    file_ends = []
    call_types = []
    with open(filename) as file:
        labels_file = file.readlines()
    for line in labels_file[1:]:
        columns = line.split('\t')
        begin_file = columns[BEGIN_FILE]
        end_file = columns[END_FILE]
        if begin_file not in selected_wav_files or end_file not in selected_wav_files:
            continue
        start_offset = offsets[selected_wav_files.index(begin_file)]
        end_offset = offsets[selected_wav_files.index(end_file)]
        call_starts.append(int(columns[BEGIN_SAMPLE]) - start_offset)
        call_ends.append(int(columns[END_SAMPLE]) - end_offset)
        file_starts.append(begin_file)
        file_ends.append(end_file)
        call_types.append(label_type)
    return CallLabels(call_starts=call_starts, call_ends=call_ends, file_starts=file_starts, file_ends=file_ends, call_types=call_types)

def _date_to_sample(wav_file_name: str, sampling_rate: float, sample_time: str):
    wav_file_start_date_str = f'{wav_file_name[4:6]}/{wav_file_name[6:8]}/{wav_file_name[0:4]} {wav_file_name[9:11]}:{wav_file_name[11:13]}:{wav_file_name[13:15]}'
    wav_file_start_date = datetime.strptime(wav_file_start_date_str, '%m/%d/%Y %H:%M:%S')
    print(sample_time)
    # sample_time_date = datetime.strptime(sample_time, '%Y/%m/%d  %H:%M:%S')
    return int((sample_time - wav_file_start_date).total_seconds() * sampling_rate)


def _filter_unlabeled(wav_file_names: List[str], wav_arrays: List[np.ndarray], metadata_path: str, sampling_rate: float,
    selected_wav_files: List[str]) -> Tuple[List[np.ndarray], List[int]]:
    """
    Exclude arrays which have no labels, and truncate their ends before / after labels.
    """
    wav_array_windows: List[np.ndarray] = []
    offsets: List[int] = []
    metadata_df = pd.read_excel(metadata_path, engine='openpyxl')

    for num, row in metadata_df.iterrows(): #very small df
        wav_file_name = row['wavFileName']
        if wav_file_name not in selected_wav_files:
            continue
        start_time = row['StartTime']
        stop_time = row['StopTime']
        arr_index = wav_file_names.index(wav_file_name)
        start_sample = _date_to_sample(wav_file_name, sampling_rate, start_time)
        end_sample = _date_to_sample(wav_file_name, sampling_rate, stop_time)
        wav_array_windows.append(wav_arrays[arr_index][start_sample:end_sample])
        offsets.append(start_sample)
    return wav_array_windows, offsets

def get_train_eval_splits(wav_file_names: List[str], split: str, metadata_path: str) -> List[str]:
    # shuffle()
    metadata_df = pd.read_excel(metadata_path, engine='openpyxl')
    relevant_files: List[str] = []
    for num, row in metadata_df.iterrows(): #very small df
        wav_file_name = row['wavFileName']
        relevant_files.append(wav_file_name)
    print(len(relevant_files))
    return relevant_files[0 : int(len(relevant_files) / 10)]
    # return random.choice(wav_file_names)


def load_dataset(site_dir: str, split: str) -> None:
    """
    Loads dataset from WAV files and labels
    """
    wav_dir = os.path.join(site_dir, 'wav')
    wav_file_names = os.listdir(wav_dir)
    metadata_path = [m for m in os.listdir(site_dir) if m.startswith('annotation_metadata')][0]
    wav_array_tuples = [sf.read(os.path.join(wav_dir, wav_file_name)) for wav_file_name in wav_file_names]
    wav_arrays = [wat[0] for wat in wav_array_tuples]
    sampling_rates = [wat[1] for wat in wav_array_tuples]
    selected_wav_files = get_train_eval_splits(wav_file_names, split, os.path.join(site_dir, metadata_path))
    wav_arrays, offsets = _filter_unlabeled(wav_file_names=wav_file_names, wav_arrays=wav_arrays,
        metadata_path=os.path.join(site_dir, metadata_path), sampling_rate=sampling_rates[0], selected_wav_files=selected_wav_files)
    print('files selected: ' , len(wav_arrays))
    all_call_labels = []
    for file in os.listdir(site_dir):
        if not file.endswith('.txt'):
            continue
        call_labels = load_file_to_labels(os.path.join(site_dir, file), selected_wav_files, offsets)
        all_call_labels.append(call_labels)
    return _build_dataset(wav_arrays, sampling_rates, all_call_labels, list(selected_wav_files))

def _time_to_sample_number(times: List[float], file_numbers: List[int], wav_arrays: np.ndarray,
    sampling_rates: List[float]):
    """
    Convert all times to samples numbers.
    TEMP
    """
    wav_file_lengths: List[float] = [len(wav_arrays[i]) for i in range(len(wav_arrays))]
    cum_wav_file_lengths: List[float] = [wav_file_lengths[0]]
    sample_numbers: List[int] = []
    for i in range(1, len(wav_file_lengths)):
        cum_wav_file_length_i = cum_wav_file_lengths[i - 1] + wav_file_lengths[i]
        cum_wav_file_lengths.append(cum_wav_file_length_i)
    for time, sr, file_number in zip(times, sampling_rates, file_numbers):
        sample_numbers.append(cum_wav_file_lengths[file_number] + int(time))
    return sample_numbers    


def _build_dataset(wav_arrays: np.ndarray, sampling_rates: List[float], all_call_labels: List[CallLabels], wav_file_names: List[str]) -> CallDataset:
    """
    Constructs the dataset
    """
    starts = [start for call_labels in all_call_labels for start in call_labels.call_starts]
    ends = [end for call_labels in all_call_labels for end in call_labels.call_ends]
    sampling_rates = [sampling_rates[0] for i in range(len(starts))] #all should be same sr
    file_starts = [wav_file_names.index(fs) for call_labels in all_call_labels for fs in call_labels.file_starts]
    file_ends = [wav_file_names.index(fe) for call_labels in all_call_labels for fe in call_labels.file_starts]
    types = [t for call_labels in all_call_labels for t in call_labels.call_types]
    print('running time to sample number')
    sample_starts = _time_to_sample_number(starts, file_starts, wav_arrays, sampling_rates)
    sample_ends = _time_to_sample_number(ends, file_ends, wav_arrays, sampling_rates)
    print('finished time to sample number')
    samples = list(chain.from_iterable(wav_arrays))
    print('samples len: ', len(samples))
    print(type(samples[0]))
    full_wav_array = np.asarray(samples)
    print('finished making wav array')
    events = [Event(event_type=types[i], event_start=sample_starts[i], event_end=sample_ends[i]) for i in range(len(starts))]
    print('making yoho')
    yoho = Yoho(window_length=sampling_rates[0] * WINDOW_LENGTH, event_types=set(types), total_length=full_wav_array.shape[0])
    print('made yoho')
    yoho_labels = yoho.get_yoho_labels(events)
    print('yoohoo!')
    #yoohoo! got yoho labels and full audio array!s
    yoho_windows = yoho.get_yoho_windows(full_wav_array)
    call_dataset = CallDataset(yoho_windows, yoho_labels)
    return call_dataset


    


