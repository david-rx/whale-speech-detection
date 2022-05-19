from dataclasses.dataclass import dataclass
from utils import get_log_melspectrogram
import numpy as np

@dataclass
class Event:
    event_type: str
    event_start: float
    event_end: float

class Yoho:

    def __init__(self, type_params: Dict[str, TypeParams], window_length: int):
        self.type_params = type_params
        self.window_length = window_length

    def smoothe_events(self, events: List[Event]):
        """
        # Post-process events to smooth out anomalies
        """
        event_types = {e.type for e in event}
        smoothed_events: List[Event] = []

        for event_type in event_types:
            typed_events = [e for e in events if event.type == event_type]
            typed_events.sort(key=lambda x: x.event_start)
            max_typed_silence = self.type_params[event_type].max_silence
            start_event = -1000
            count = 0

            while count < len(speech_events) - 1:
                if (speech_events[count][1] >= typed_events[count + 1][0]) or (typed_events[count + 1][0] - typed_events[count][1] <= max_typed_silence):
                    typed_events[count][1] = max(typed_events[count + 1][1], typed_events[count][1])
                    del typed_events[count + 1]
                else:
                    count += 1
            smoothed_events.extend(typed_events)

        for i in range(len(smooth_events)):
            smooth_events[i][0] = round(smooth_events[i][0], 3)
            smooth_events[i][1] = round(smooth_events[i][1], 3)

        smooth_events.sort(key=lambda x: x.event_start)

        return smooth_events

    def get_yoho_labels(self, events: List[Event]) -> np.array:
        """
        Given a list of Events, smooth them and window to generate labels for YOHO
        """
        events = self.smooth_events(events)
        num_divisions = int(self.total_length / self.window_length)
        labels = np.zeroes(num_divisions, len(self.label_types) * 3)
        
        for e in events:


            start_bin = int(event.start_time / self.window_length)
            stop_bin = int(event.stop_time / self.window_length)

            start_time_2 = event.start_time - start_bin * self.window_length
            stop_time_2 = event.stop_time - stop_bin * self.window_length

            n_bins = stop_bin - start_bin

            if n_bins == 0: #what case is this?
                label_index = self.label_types.index(e.event_type)
                labels[start_bin, label_index : label_index + 3] = [1, start_time_2, stop_time_2]
        return labels


    