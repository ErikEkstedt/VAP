python vap/main.py \
  datamodule.train_path=/home/erik/projects/data/air/splits/train.csv \
  datamodule.val_path=/home/erik/projects/data/air/splits/val.csv \
  datamodule.batch_size=4 \
  datamodule.num_workers=4 \
  +pretrained_checkpoint_path=/home/erik/projects/CCConv/VAP/example/checkpoints/checkpoint.ckpt
