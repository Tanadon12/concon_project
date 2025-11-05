#!/bin/bash

METHODS=('baseline')
DATASET_TYPE='global_pix_200'

ROOT_DIR=$3
SEED=$1
MODEL=$2

TEST_PATH_GLOBAL=$ROOT_DIR'/'$DATASET_TYPE'/test/images/t0'
TRAIN_PATH_GLOBAL=$ROOT_DIR'/'$DATASET_TYPE'/train/images/t0'
VAL_PATH_GLOBAL=$ROOT_DIR'/'$DATASET_TYPE'/val/images/t0'


for method in "${METHODS[@]}"
do

    if [ "$MODEL" = 'Slot' ]; then

        EPOCHS=150
        BATCHSIZE=64

        python train_files/train_global_slot.py --dataset_type $DATASET_TYPE --method $method --test_path_global $TEST_PATH_GLOBAL --train_path_global $TRAIN_PATH_GLOBAL --val_path_global $VAL_PATH_GLOBAL --seed $SEED \
        --epochs $EPOCHS --batch_size $BATCHSIZE --model_name $MODEL --lr 0.001 --l2_grads 1000 --n-slots 10 --n-iters-slot-att 3 --n-attr 18 --results_dir $ROOT_DIR --rtpt 'RK'
        
    else 

        EPOCHS=50
        BATCHSIZE=128

        python train_files/train_global.py --dataset_type $DATASET_TYPE --method $method --test_path_global $TEST_PATH_GLOBAL --train_path_global $TRAIN_PATH_GLOBAL --val_path_global $VAL_PATH_GLOBAL --seed $SEED \
        --epochs $EPOCHS --batch_size $BATCHSIZE --model_name $MODEL --results_dir $ROOT_DIR --rtpt 'RK'
        
    fi
    
done