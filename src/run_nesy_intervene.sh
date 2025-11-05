#!/bin/bash

# ✅ Set general training parameters
MODEL="Slot"
EPOCHS=10
BATCHSIZE=64
LR=0.001

# ✅ Parse input arguments
SEED=$1
DATASET_TYPE=$2

# ✅ Fixed ROOT_DIR to point to your dataset path
ROOT_DIR="/workspaces/dataset"

# ✅ Slot Attention Weights path
SWA="slot-attention-clevr-state-145"

# ✅ Print config summary
echo "🔧 Running pipeline with:"
echo "SEED: $SEED"
echo "DATASET_TYPE: $DATASET_TYPE"
echo "ROOT_DIR: $ROOT_DIR"
echo "SLOT_ATTENTION_WEIGHTS: $SWA"
echo "=============================="


# ✅ Train and collect attrs_trans for task t0
echo "🔷 Task t0: Train + Collect attrs_trans"
python /workspaces/src/train_files/collect_attrs_trans.py \
  --tasks t0,t1 \
  --dataset_type $DATASET_TYPE \
  --root_dir $ROOT_DIR \
  --slot_attention_weights $SWA \
  --epochs $EPOCHS \
  --lr $LR \
  --seed $SEED

# ✅ Intervention analysis
echo "🔬 Running intervention analysis on task t1 using confounder from task t0"
python /workspaces/src/train_files/intervene.py \
  --attrs_t0 ${ROOT_DIR}/attrs_trans_t0.npy \
  --attrs_t1 ${ROOT_DIR}/attrs_trans_t1.npy \
  --slot_attention_weights $SWA \
  --dataset_path_t1 ${ROOT_DIR}/${DATASET_TYPE}/train/images/t1 \
  --device cuda

echo "✅ Pipeline finished successfully."
