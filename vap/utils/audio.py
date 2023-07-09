import torch
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT
from typing import Any, Dict, Optional, Tuple


SAMPLE_RATE: int = 16_000
N_MELS: int = 80
N_FFT: int = 400
HOP_LENGTH: int = 320
VAD_LIST = list[list[list[float]]]


def time_to_samples(t: float, sample_rate: int) -> int:
    return int(t * sample_rate)


def time_to_frames(t: float, hop_time: float) -> int:
    return int(t / hop_time)


def sample_to_time(n_samples: int, sample_rate: int) -> float:
    return n_samples / sample_rate


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    info = torchaudio.info(audio_path)
    return {
        "name": audio_path,
        "duration": sample_to_time(info.num_frames, info.sample_rate),
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.bits_per_sample,
        "encoding": info.encoding,
    }


def load_waveform(
    path: str,
    sample_rate: Optional[int] = 16000,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    mono: bool = False,
) -> Tuple[torch.Tensor, int]:
    if start_time is None and end_time is None:
        x, sr = torchaudio.load(path)
    else:
        info = get_audio_info(path)

        start_frame = 0
        if start_time is not None:
            start_frame = time_to_samples(start_time, info["sample_rate"])

        end_frame = info["num_frames"]
        if end_time is not None:
            end_frame = time_to_samples(end_time, info["sample_rate"])

        num_frames = end_frame - start_frame
        x, sr = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)

    if mono and x.shape[0] > 1:
        x = x.mean(dim=0).unsqueeze(0)

    if sample_rate is not None:
        if sr != sample_rate:
            x = AF.resample(x, orig_freq=sr, new_freq=sample_rate)
            sr = sample_rate
    return x, sr


def mono_to_stereo(
    audio: torch.Tensor, vad_list: VAD_LIST, sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    """
    audio: Tensor, (1, n_samples)
    vad_list: list[  list[list[float,float]],  list[list[float,float]]  ]
    sample_rate: int, sampling rate of the audio (default: 16_000)

    Returns
        stereo: Tensor, (2, n_samples)
    """
    assert (
        audio.ndim == 2
    ), f"audio must be mono (1, n_samples), got {tuple(audio.shape)}"
    assert (
        audio.shape[0] == 1
    ), f"audio must be mono (1, n_samples), got {tuple(audio.shape)}"
    stereo = torch.zeros_like(audio).repeat(2, 1)
    for ch, ch_vad in enumerate(vad_list):
        for s, e in ch_vad:
            s = time_to_samples(s, sample_rate)
            e = time_to_samples(e, sample_rate)
            stereo[ch, s:e] = audio[0, s:e]
    return stereo


def log_mel_spectrogram(
    waveform: torch.Tensor,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """
    Inspired by OpenAIs whisper repo
    """
    mel_spec = AT.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        normalized=True,
    )(waveform)
    log_mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_mel_spec = torch.maximum(log_mel_spec, log_mel_spec.max() - 8.0)
    log_mel_spec = (log_mel_spec + 4.0) / 4.0
    return log_mel_spec


if __name__ == "__main__":

    from vap.data.datamodule import VAPDataset

    dset = VAPDataset(path="example/data/sliding_dev.csv")

    ii = 0
    d = dset[ii]
    mono = d["waveform"].mean(0).unsqueeze(0)
    vad_list = dset.df.iloc[ii]["vad_list"]
    stereo = mono_to_stereo(mono, vad_list, sample_rate)
