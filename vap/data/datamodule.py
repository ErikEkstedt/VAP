from os.path import exists
import pandas as pd
import json
from typing import Optional, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import lightning as L

from vap.utils.audio import load_waveform, mono_to_stereo
from vap.utils.utils import vad_list_to_onehot


SAMPLE = Mapping[str, Tensor]


def load_df(path: str) -> pd.DataFrame:
    def _vl(x):
        return json.loads(x)

    def _session(x):
        return str(x)

    converters = {
        "vad_list": _vl,
        "session": _session,
    }
    return pd.read_csv(path, converters=converters)


def force_correct_nsamples(w: Tensor, n_samples: int) -> Tensor:
    if w.shape[-1] > n_samples:
        w = w[:, -n_samples:]
    elif w.shape[-1] < n_samples:
        w = torch.cat([w, torch.zeros_like(w)[:, : n_samples - w.shape[-1]]], dim=-1)
    return w


class VAPDataset(Dataset):
    def __init__(
        self,
        path: str,
        horizon: float = 2,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
    ) -> None:
        self.path = path
        self.df = load_df(path)

        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.horizon = horizon
        self.mono = mono

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> SAMPLE:
        d = self.df.iloc[idx]
        # Duration can be 19.99999999999997 for some clips and result in wrong vad-shape
        # so we round it to nearest second
        dur = round(d["end"] - d["start"])
        w, _ = load_waveform(
            d["audio_path"],
            start_time=d["start"],
            end_time=d["end"],
            sample_rate=self.sample_rate,
            mono=self.mono,
        )

        # Ensure correct duration
        # Some clips (20s) becomes
        # [2, 320002] insted of [2, 320000]
        # breaking the batching
        n_samples = int(dur * self.sample_rate)
        w = force_correct_nsamples(w, n_samples)

        # Stereo Audio
        # Use the vad-list information to convert mono to stereo
        if w.shape[0] == 1:
            w = mono_to_stereo(w, d["vad_list"], sample_rate=self.sample_rate)

        vad = vad_list_to_onehot(
            d["vad_list"], duration=dur + self.horizon, frame_hz=self.frame_hz
        )

        return {
            "session": d.get("session", ""),
            "waveform": w,
            "vad": vad,
            "dataset": d.get("dataset", ""),
        }


class VAPDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        horizon: float = 2,
        sample_rate: int = 16000,
        frame_hz: int = 50,
        mono: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        **kwargs,
    ):
        super().__init__()

        # Files
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # values
        self.mono = mono
        self.horizon = horizon
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz

        # DataLoder
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tTrain: {self.train_path}"
        s += f"\n\tVal: {self.val_path}"
        s += f"\n\tTest: {self.test_path}"
        s += f"\n\tHorizon: {self.horizon}"
        s += f"\n\tSample rate: {self.sample_rate}"
        s += f"\n\tFrame Hz: {self.frame_hz}"
        s += f"\nData"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"
        return s

    def prepare_data(self):
        if self.train_path is not None:
            assert self.path_exists("train"), f"No TRAIN file found: {self.train_path}"

        if self.val_path is not None:
            assert self.path_exists("val"), f"No TRAIN file found: {self.train_path}"

        if self.test_path is not None:
            assert exists(self.test_path), f"No TEST file found: {self.test_path}"

    def path_exists(self, split):
        path = getattr(self, f"{split}_path")
        if path is None:
            return False

        if not exists(path):
            return False
        return True

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""

        if stage in (None, "fit"):
            assert self.path_exists("train"), f"Train path not found: {self.train_path}"
            assert self.path_exists("val"), f"Val path not found: {self.val_path}"
            self.train_dset = VAPDataset(
                self.train_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
            )
            self.val_dset = VAPDataset(
                self.val_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
            )

        if stage in (None, "test"):
            assert self.path_exists("test"), f"Test path not found: {self.test_path}"
            self.test_dset = VAPDataset(
                self.test_path,
                horizon=self.horizon,
                sample_rate=self.sample_rate,
                frame_hz=self.frame_hz,
                mono=self.mono,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=False,
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from tqdm import tqdm
    from vap.utils.plot import plot_melspectrogram, plot_vad

    dset = VAPDataset(path="example/data/sliding_dev.csv")

    d = dset[0]

    # PLOT A DATASAMPLE
    fig, ax = plt.subplots(2, 1)
    plot_melspectrogram(d["waveform"], ax=ax[:2])
    # plot vad.
    # VAD is by default longer than the audio (for prediction)
    # So you will probably see zeros at the end where the VAD is defined but the audio not.
    x = torch.arange(d["vad"].shape[0]) / dset.frame_hz
    plot_vad(x, d["vad"][:, 0], ax[0])
    plot_vad(x, d["vad"][:, 1], ax[1])
    plt.show()

    dm = VAPDataModule(
        train_path="example/data/sliding_dev.csv",
        val_path="example/data/sliding_dev.csv",
        test_path="example/data/sliding_dev.csv",
        batch_size=4,
        num_workers=2,
    )
    dm.prepare_data()
    dm.setup("fit")
    print(dm)
    print("Train: ", len(dm.train_dset))
    print("Val: ", len(dm.val_dset))

    dloader = dm.train_dataloader()
    for batch in tqdm(dloader, total=len(dloader)):
        pass
