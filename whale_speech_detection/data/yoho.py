from re import L
from dataclasses.dataclass import dataclass
from utils import get_log_melspectrogram
import numpy as np
from typing import Dict, Iterable, List
import math

MAX_TYPED_SILENCE = 0.2

@dataclass
class Event:
    event_type: str
    event_start: float
    event_end: float

@dataclass
class TypeParams:
    max_silence: float


class Yoho:

    def __init__(self, window_length: int, event_types: Iterable[str], total_length: float):
        # self.type_params = type_params
        self.window_length = window_length
        self.event_types = list(event_types)
        self.total_length = total_length

    def smooth_events(self, events: List[Event]) -> List[Event]:
        """
        Post-process events to smooth out anomalies
        """
        event_types = {e.type for e in events}
        smoothed_events: List[Event] = []

        for event_type in event_types:
            typed_events = [e for e in events if e.event_type == event_type]
            typed_events.sort(key=lambda x: x.event_start)
            # max_typed_silence = self.type_params[event_type].max_silence
            max_typed_silence = 0.1
            start_event = -1000 #what is this?
            count = 0

            while count < len(typed_events) - 1:
                if (typed_events[count][1] >= typed_events[count + 1][0]) or (typed_events[count + 1][0] - typed_events[count][1] <= max_typed_silence):
                    typed_events[count][1] = max(typed_events[count + 1][1], typed_events[count][1])
                    del typed_events[count + 1]
                else:
                    count += 1
            smoothed_events.extend(typed_events)

        for i in range(len(smoothed_events)):
            smoothed_events[i].event_start = round(smoothed_events[i].event_start, 3)
            smoothed_events[i].event_end = round(smoothed_events[i].event_end, 3)

        smoothed_events.sort(key=lambda x: x.event_start)

        return smoothed_events

    def get_yoho_labels(self, events: List[Event]) -> np.ndarray:
        """
        Given a list of Events, smooth them and window to generate labels for YOHO
        """
        events = self.smooth_events(events)
        num_divisions = int(self.total_length / self.window_length)
        labels = np.zeros(num_divisions, len(self.label_types) * 3)
        
        for event in events:

            start_bin = int(event.event_start / self.window_length)
            stop_bin = int(event.event_end / self.window_length)

            start_time_2 = event.event_start - start_bin * self.window_length
            stop_time_2 = event.event_end - stop_bin * self.window_length

            n_bins = stop_bin - start_bin
            if n_bins == 0:
                label_index = self.event_types.index(event.event_type)
                labels[start_bin, label_index : label_index + 3] = [1, start_time_2, stop_time_2]
            elif n_bins == 1:
                labels[start_bin, label_index : label_index + 3] = [1, start_time_2, self.window_length]
                if stop_time_2 > 0.0:
                    labels[stop_bin, label_index : label_index + 3] = [1, 0.0, stop_time_2]
            elif n_bins > 1:
                labels[start_bin, label_index : label_index + 3] = [1, start_time_2, self.window_length]
                for i in range(1, n_bins):
                    labels[start_bin + i, label_index : label_index + 3] = [1, 0.0, self.window_length]
                if stop_time_2 > 0.0:
                    labels[stop_bin, label_index : label_index + 3] = [1, 0.0, stop_time_2]
        labels[:, [i for i in range(len(labels) -1) if i % 3 != 0]] /= self.window_length
        return labels

    def get_yoho_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        num_windows = math.ceil(audio.shape[0] / self.window_length)
        return [audio[i*self.window_length : i + 1 * self.window_length] for i in range(num_windows)]


