@echo off

set Model_exp_name=CompareMorphometrics

for /f "tokens=1-4 delims=:. " %%a in ("%date% %time%") do (
    set exp_id=%Model_exp_name%_%%a%%b%%c_%%d
)


	
python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run00.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42
	
python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run01.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run02.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run03.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run04.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run05.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run06.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run07.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run08.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

python SOTAbaseline_Morphometrics_kmeans.py ^
    --dataset_name urbanform ^
    --batch_size 256 ^
    --num_workers 8 ^
    --eval_funcs v4 ^
    --exp_name %Model_exp_name% ^
    --exp_id %exp_id% ^
    --class_split_file datasets/urbanForm/class_split_info_10times_run09.json ^
	--morphology_npz datasets/batch_morphology_59_features_standardized.npz ^
	--seed 42

