import torch
from typing import Any
from vap.utils.utils import get_vad_list_subset

VAD_LIST = list[list[list[float]]]


def get_vad_list_lims(vad_list: VAD_LIST) -> tuple[float, float]:
    start = max(vad_list[0][0][0], vad_list[1][0][0])
    end = max(vad_list[0][-1][-1], vad_list[1][-1][-1])
    return start, end


def get_sliding_windows(
    vad_list: VAD_LIST, duration: float = 20, overlap: float = 5
) -> list[float]:
    """
    Sliding windows of a session
    """
    # Get boundaries from vad
    start, end = get_vad_list_lims(vad_list)

    # get valid starting times for sliding window
    duration = end - start
    step = duration - overlap
    n_clips = int((duration - duration) / step + 1)
    starts = torch.arange(start, end, step)[:n_clips].tolist()
    return starts


def sliding_window(
    vad_list: VAD_LIST, duration: float = 20, overlap: float = 5, horizon: float = 2
) -> list[dict[str, Any]]:
    """
    Get overlapping samples from a vad_list of a conversation
    """
    samples = []
    starts = get_sliding_windows(vad_list, duration, overlap)
    for start in starts:
        end = start + duration
        vad_list = get_vad_list_subset(vad_list, start, end + horizon)
        samples.append(
            {
                "start": start,
                "end": end,
                "vad_list": vad_list,
            }
        )
    return samples
