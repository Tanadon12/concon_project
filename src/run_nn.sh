#!/bin/bash

# ✅ Set general training parameters
MODEL="Slot"
EPOCHS=10
LR=0.001

# ✅ Parse input arguments
SEED=$1
DATASET_TYPE=$2
ROOT_DIR=$3
SWA=$4   # Slot Attention Weights path

# ✅ Print config summary
echo "🔧 Running pipeline with:"
echo "SEED: $SEED"
echo "DATASET_TYPE: $DATASET_TYPE"
echo "ROOT_DIR: $ROOT_DIR"
echo "SLOT_ATTENTION_WEIGHTS: $SWA"
echo "=============================="

# ✅ Train and collect attrs_trans for task t0
echo "🔷 Task t0: Train + Collect attrs_trans"
python train_collect_attr_trans.py \
  --task t0 \
  --dataset_type $DATASET_TYPE \
  --root_dir $ROOT_DIR \
  --slot_attention_weights $SWA \
  --epochs $EPOCHS \
  --lr $LR

# ✅ Train and collect attrs_trans for task t1
echo "🔷 Task t1: Train + Collect attrs_trans"
python train_collect_attr_trans.py \
  --task t1 \
  --dataset_type $DATASET_TYPE \
  --root_dir $ROOT_DIR \
  --slot_attention_weights $SWA \
  --epochs $EPOCHS \
  --lr $LR

# ✅ Intervention analysis
echo "🔬 Running intervention analysis on task t1 using confounder from task t0"
python intervene.py \
  --attrs_t0 ${ROOT_DIR}/attrs_trans_t0.npy \
  --attrs_t1 ${ROOT_DIR}/attrs_trans_t1.npy \
  --slot_attention_weights $SWA \
  --dataset_path_t1 ${ROOT_DIR}/${DATASET_TYPE}/train/images/t1 \
  --device cuda

echo "✅ Pipeline finished successfully."
    