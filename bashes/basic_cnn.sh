#!/usr/bin/env bash

cd ..
python3 train.py\
    --model_class BasicCNNModel\
    --checkpoint_dir checkpoints/basic_cnn\
    --tensor_board_dir events/basic_cnn