from torch.data import Dataset
from typing import List
from dataclasses import dataclass
import soundfile as sf
import os



BEGIN_FILE = 3
END_FILE = 4
BEGIN_TIME = 5
END_TIME = 6


# FILE_TO_LABEL = {'BmAnt-A': 'BmAnt-A'}

class CallDataset(Dataset):

    def __init__(self, wav_files) -> None:
        self.wav_files = wav_files
        pass
    
    def __len__(self):
        return len(self.wav_files) #what is a single example? Treat as NER (?)

    def __getitem__(self, idx):
        return self.wav_files[idx]

@dataclass
class CallLabels:
    call_starts: List
    call_ends: List
    file_starts: List
    file_ends: List


def load_file_to_labels(filename) -> CallLabels:
    """
    Processes a single file of a specific label type and returns the corresponding labels
    """
    label_type = filename.split('.')[1]
    call_starts = []
    call_ends = []
    label_file_starts = []
    label_file_ends = []
    with open(filename) as file:
        labels_file = file.readlines()
    for line in labels_file:
        columns = line.split('  ')
        call_starts.append(columns[BEGIN_TIME])
        call_ends.append(columns[END_TIME])
        file_starts.append(columns[BEGIN_FILE])
        file_ends.append(columns[END_FILE])
    return CallLabels(call_starts=call_starts, call_ends=call_ends, file_starts=file_starts, file_ends=file_ends)

def load_dataset(site_dir) -> None:
    wav_files = os.listdir(os.path.join(site_dir, 'wav'))
    wav_array = [sf.read(wav_file)[0] for wav_file in wav_files]
    all_call_labels = []
    for file in os.listdir(site_dir):
        call_labels = load_file_to_labels(file)
        all_call_labels.append(call_labels)
    return build_dataset(wav_files, all_call_labels)


def build_dataset(wav_files: List, all_call_labels: List[CallLabels]) -> CallDataset:
    return 

