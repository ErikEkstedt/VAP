#!/bin/bash

python vap/main.py datamodule.train_path=data/splits_twilio/train_sliding.csv datamodule.val_path=data/splits_twilio/val_sliding.csv datamodule.batch_size=100 datamodule.num_workers=26 datamodule.prefetch_factor=4
