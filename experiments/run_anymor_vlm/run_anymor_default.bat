@echo off

cd /d "%~dp0\..\.."

set cuda_id=0
set Step1_exp_name=TES
set Step2_exp_name=AnyMor
set "CLASS_SPLIT_FILE=data\class_split_info_run00.json"

for /f "tokens=1-4 delims=:. " %%a in ("%date% %time%") do (
    set exp_id=%Step1_exp_name%_%Step2_exp_name%_%%a%%b%%c_%%d
)

	
python Anymor_Step1.py ^
    --dataset_name urbanform ^
    --class_split_file "%CLASS_SPLIT_FILE%" ^
    --batch_size 256 ^
    --epochs 100 ^
    --num_workers 8 ^
    --lr 0.01 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Step1_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
	--seed 42

python Anymor_Step2.py ^
    --dataset_name urbanform ^
    --class_split_file "%CLASS_SPLIT_FILE%" ^
    --captions_path captions_6cls.npz ^
    --batch_size 256 ^
    --epochs 70 ^
    --num_workers 8 ^
    --sup_weight 0.15 ^
    --lr 0.01 ^
    --eval_funcs v4 ^
    --warmup_teacher_temp_epochs 10 ^
    --cuda_dev %cuda_id% ^
    --exp_name %Step2_exp_name% ^
    --exp_id %exp_id% ^
	--num_words 30 ^
	--prop_train_labels 0.4 ^
	--seed 42
