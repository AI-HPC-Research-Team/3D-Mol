#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

### start pretrain
compound_encoder_config="model_configs/geognn_l8.json"
model_config="model_configs/pretrain_gem.json"
dataset="zinc"
data_path="./pretrain_dataset/pubchem.txt"
cached_data_path="./cached_data/pubchem"
python pretrain.py \
    --cached_data_path=$cached_data_path \
		--batch_size=256 \
		--num_workers=96 \
		--max_epoch=3 \
		--lr=1e-3 \
		--task=data \
		--dropout_rate=0.2 \
		--dataset=$dataset \
		--data_path=$data_path \
		--compound_encoder_config=$compound_encoder_config \
		--model_config=$model_config \
		--model_dir=./pretrain_models


