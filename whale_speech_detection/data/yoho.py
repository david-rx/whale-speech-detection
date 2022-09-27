from re import L
from dataclasses import dataclass
import numpy as np
from typing import Dict, Iterable, List
import math
import logging

from whale_speech_detection.data.utils import get_log_melspectrogram

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
        smoothed_events: List[Event] = []

        for event_type in self.event_types:
            logging.info("starting event %s", event_type)
            typed_events = [e for e in events if e.event_type == event_type]
            typed_events.sort(key=lambda x: x.event_start)
            # max_typed_silence = self.type_params[event_type].max_silence
            max_typed_silence = 0.1
            start_event = -1000 #what is this?
            count = 0

            while count < len(typed_events) - 1:
                if (typed_events[count].event_end >= typed_events[count + 1].event_start) or (typed_events[count + 1].event_start - typed_events[count].event_end <= max_typed_silence):
                    typed_events[count].event_end = max(typed_events[count + 1].event_end, typed_events[count].event_end)
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
        print('smoothing events')
        events = self.smooth_events(events)
        print('smoothed!')
        num_divisions = math.ceil(self.total_length / self.window_length)
        labels = np.zeros((num_divisions, len(self.event_types) * 3),)

        for event in events:
            print(f"event start time is {event.event_start} and end time is {event.event_end}")

            start_bin = int(event.event_start / self.window_length)
            stop_bin = int(event.event_end / self.window_length)

            start_time_2 = event.event_start - start_bin * self.window_length
            stop_time_2 = event.event_end - stop_bin * self.window_length

            n_bins = stop_bin - start_bin
            label_index = self.event_types.index(event.event_type)
            if n_bins == 0:
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
        labels[:, [i for i in range(len(labels[0]) -1) if i % 3 != 0]] /= self.window_length
        return labels

    def get_yoho_windows(self, audio: np.ndarray) -> List[np.ndarray]:
        num_windows = math.ceil(audio.shape[0] / self.window_length)
        windowed_audio = [audio[int(i*self.window_length) : int((i + 1) * self.window_length)] for i in range(num_windows)]
        windowed_audio[-1] = windowed_audio[0]
        return windowed_audio


