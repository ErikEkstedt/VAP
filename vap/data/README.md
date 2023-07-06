# Data


To train the model using the included `LightningDataModule` we assume that we have csv-files where each row defines a sample.


* audio_path
    - `PATH/TO/AUDIO.wav`
* start: float, definining the start time of the sample
    - `0.0`
* end: float, definining the end time of the sample
    - `20.0`
* session: str, the name of the sample session
    - `4637`
* dataset: str, the name of the dataset
    - `switchboard`
* vad_list: a list containing the voice-activity start/end-times inside of the `start`/`end` times of the row-sample

VAD-list example with relative start/end times grounded in the `start`/`end` time of the row sample audio
```json
[
    [
        [1.16, 1.43], 
        [1.73, 3.17], 
        [3.27, 3.74], 
        [3.94, 4.83], 
        [5.41, 6.8]
    ], 
    [
        [0.04, 0.28], 
        [5.35, 5.83], 
        [7.18, 9.3], 
        [10.17, 15.12], 
        [16.2, 17.17], 
        [18.08, 19.03], 
        [20.4, 20.75], 
        [21.2, 22.0]
    ]
]
```

The top of the csv-file should look like this (see `example/data/sliding_dev.csv`)
```csv
audio_path,start,end,vad_list,session,dataset
/PATH/AUDIO.wav,0.0,20.0,"[[[1.16, 1.43], [1.73, 3.17], [3.27, 3.74], [3.94, 4.83], [5.41, 6.8]], [[0.04, 0.28], [5.35, 5.83], [7.18, 9.3], [10.17, 15.12], [16.2, 17.17], [18.08, 19.03], [20.4, 20.75], [21.2, 22.0]]]",4637,switchboard
```

### Dataset
```python

dset = VAPDataset(
    path:str ="example/data/sliding_dev.csv"
    horizon: float = 2,
    sample_rate: int = 16_000,
    frame_hz: int = 50,
    mono: bool = False,
    )

d = dset[0]
```

### DataModule

* [LightningDataModule](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule)


```python
dm = VAPDataModule(
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
    )
dm.prepare_data()
dm.setup("fit")

print(dm)
print("Train: ", len(dm.train_dset))
print("Val: ", len(dm.val_dset))

dloader = dm.train_dataloader()
for batch in tqdm(dloader, total=len(dloader)):
    pass
```
