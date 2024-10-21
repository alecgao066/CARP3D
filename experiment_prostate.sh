python create_splits_seq.py --task task_1_tumor_vs_normal --seed 3 --label_frac 1.0 --k 8  --leave_one_out 

CUDA_VISIBLE_DEVICES=1 python main.py --drop_out --lr 2e-4 --k 8 --label_frac 1.0 --no_inst_cluster --leave_one_out --agg_range 3 --agg_gap 1 --exp_code exp_prostate_range80gap40_cat --weighted_sample --max_epochs 5 --bag_loss ce --task task_1_tumor_vs_normal --model_type abmil_attn --log_data --data_root_dir test_data/cpath_prostate2-5/ --data_aug_dir test_data/resnet_aug_prostate2-5/
