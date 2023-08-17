#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="../../../.."
export PYTHONPATH="$(pwd)/$root_path/":$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

### download demo data
### wget https://baidu-nlp.bj.bcebos.com/PaddleHelix/datasets/compound_datasets/demo_zinc_smiles.tgz
### tar xzf demo_zinc_smiles.tgz

### start pretrain
compound_encoder_config="model_configs/geognn_l8.json"
model_config="model_configs/pretrain_gem.json"
data_path="./demo_zinc_smiles"
cached_data_path="./cached_data/pubchem"
python pretrain_all.py \
    --cached_data_path=$cached_data_path \
		--batch_size=128 \
		--num_workers=96 \
		--max_epoch=50 \
		--lr=1e-3 \
		--dropout_rate=0.2 \
		--dataset=$dataset \
		--data_path=$data_path \
		--compound_encoder_config=$compound_encoder_config \
		--model_config=$model_config \
		--model_dir=./pretrain_model