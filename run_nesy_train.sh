python train_files/train_nesy_pnns.py \
--method pnns \
--results_dir ./results \
--dataset_type case_disjoint_main \
--epochs 50 \
--batch_size 16 \
--model_name Slot_Attention \
--slot_attention_weights ./train_files/slot-attention-clevr-state-145 \
\
--train_path_task0 ./dataset/case_disjoint_main/train/task0 \
--train_path_task1 ./dataset/case_disjoint_main/train/task1 \
--train_path_task2 ./dataset/case_disjoint_main/train/task2 \
\
--val_path_task0 ./dataset/case_disjoint_main/val/task0 \
--val_path_task1 ./dataset/case_disjoint_main/val/task1 \
--val_path_task2 ./dataset/case_disjoint_main/val/task2 \
--val_flag \
\
--test_path_task0 ./dataset/case_disjoint_main/test/task0 \
--test_path_task1 ./dataset/case_disjoint_main/test/task1 \
--test_path_task2 ./dataset/case_disjoint_main/test/task2 \
--test_path_global ./dataset/unconfounded \
--rtpt "RK"
