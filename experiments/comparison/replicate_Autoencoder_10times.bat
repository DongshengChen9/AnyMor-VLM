@echo off

set cuda_id=0
set Model_exp_name=CompareAutoencoder

for /f "tokens=1-4 delims=:. " %%a in ("%date% %time%") do (
    set exp_id=%Model_exp_name%_%%a%%b%%c_%%d
)


	
python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run00.json ^
	--seed 42
	
python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run01.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run02.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run03.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run04.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run05.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run06.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run07.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run08.json ^
	--seed 42

python SOTAbaseline_AE_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 128 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --weight_decay 1e-06 ^
    --transform imagenet ^
    --lr 0.001 ^
    --eval_funcs v4 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run09.json ^
	--seed 42

