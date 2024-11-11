#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../../.."
export PYTHONPATH="$root_path/":$PYTHONPATH
#datasets="esol freesolv lipophilicity"
datasets="esol"
compound_encoder_config="model_configs/geognn_l8.json"
init_model="./pretrain_models-chemrl_gem/regr.pdparams"
log_prefix="./logs/finetune_data"
#thread_num=4
count=0
for dataset in $datasets; do
	echo "==> $dataset"
	data_path="./downstream_datasets/$dataset"
	cached_data_path="./cached_data/$dataset"
	if [ ! -f "$cached_data_path.done" ]; then
		rm -r $cached_data_path
		python finetune_regr.py \
				--task=data \
				--num_workers=16 \
				--dataset_name=$dataset \
				--data_path=$data_path \
				--cached_data_path=$cached_data_path \
				--compound_encoder_config=$compound_encoder_config \
				--model_config="model_configs/down_mlp2.json"
		if [ $? -ne 0 ]; then
			echo "Generate data failed for $dataset"
			exit 1
		fi
	fi
done
wait

#!/bin/bash
cd $(dirname $0)
cd ..

source ~/.bashrc
source ./scripts/utils.sh

root_path="$(pwd)/../../../.."
export PYTHONPATH="$root_path/":$PYTHONPATH
#datasets="bace sider tox21 toxcast"
datasets="bace"
compound_encoder_config="model_configs/geognn_l8.json"
init_model="./pretrain_models-chemrl_gem/class.pdparams"
log_prefix="./logs/finetune_data"
#thread_num=4
count=0
for dataset in $datasets; do
	echo "==> $dataset"
	data_path="./downstream_datasets/$dataset"
	cached_data_path="./cached_data/$dataset"
     if [ ! -f "$cached_data_path.done" ]; then
		rm -r $cached_data_path
		python finetune_class.py \
				--task=data \
				--num_workers=16 \
				--dataset_name=$dataset \
				--data_path=$data_path \
				--cached_data_path=$cached_data_path \
				--compound_encoder_config=$compound_encoder_config \
				--model_config="model_configs/down_mlp2.json"
		if [ $? -ne 0 ]; then
			echo "Generate data failed for $dataset"
			exit 1
		fi
	fi
done
wait
