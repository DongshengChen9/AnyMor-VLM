@echo off

set cuda_id=0
set Model_exp_name=ComparerandomPermute

for /f "tokens=1-4 delims=:. " %%a in ("%date% %time%") do (
    set exp_id=%Model_exp_name%_%%a%%b%%c_%%d
)


	
python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run00.json ^
	--TES_model_dir saved_models/tes_222_run00.pt ^
    --test_model saved_models/anymor_222_run00.pt ^
	--seed 42
	
python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run01.json ^
	--TES_model_dir saved_models/tes_222_run01.pt ^
    --test_model saved_models/anymor_222_run01.pt ^
	--seed 42

python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run02.json ^
	--TES_model_dir saved_models/tes_222_run02.pt ^
    --test_model saved_models/anymor_222_run02.pt ^
	--seed 42

python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run03.json ^
	--TES_model_dir saved_models/tes_222_run03.pt ^
    --test_model saved_models/anymor_222_run03.pt ^
	--seed 42
	
python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run04.json ^
	--TES_model_dir saved_models/tes_222_run04.pt ^
    --test_model saved_models/anymor_222_run04.pt ^
	--seed 42

python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run05.json ^
	--TES_model_dir saved_models/tes_222_run05.pt ^
    --test_model saved_models/anymor_222_run05.pt ^
	--seed 42

python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run06.json ^
	--TES_model_dir saved_models/tes_222_run06.pt ^
    --test_model saved_models/anymor_222_run06.pt ^
	--seed 42
	
python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run07.json ^
	--TES_model_dir saved_models/tes_222_run07.pt ^
    --test_model saved_models/anymor_222_run07.pt ^
	--seed 42

python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run08.json ^
	--TES_model_dir saved_models/tes_222_run08.pt ^
    --test_model saved_models/anymor_222_run08.pt ^
	--seed 42

python SOTAbaseline_randomPermute.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --grad_from_block 11 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run09.json ^
	--TES_model_dir saved_models/tes_222_run09.pt ^
    --test_model saved_models/anymor_222_run09.pt ^
	--seed 42