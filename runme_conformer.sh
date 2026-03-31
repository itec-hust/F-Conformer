#!/bin/bash

# ============================================================================
# Piano Transcription System - Training & Inference Script
# ============================================================================

# --- 路径与环境配置 ---
# 建议用户通过命令行参数或修改此处来指向其本地路径
WORKSPACE="workspaces"
DATASET="maestro"           # 选项: maestro, omap, ST500_cqt, 等
MODEL_TYPE="Conformer"      # 选项: Regress, Conformer, Regress_HAT, 等
GPU_ID=0                    # 指定使用的 GPU 编号

# --- 训练超参数 ---
BATCH_SIZE=4
LEARNING_RATE=1e-3
LOSS_TYPE="regress_onset_offset_frame_velocity_bce"
EARLY_STOP=1500000

# --- 断点续传配置 ---
# 开源版本中，将此路径设为空，或者指向项目内的相对路径
# 用户可以根据需要自行指定之前训练好的 .pth 文件
RESUME_CHECKPOINT="" 

# ============================================================================
# 1. 训练模型 (Train)
# ============================================================================
echo "Starting Training: Model=$MODEL_TYPE, Dataset=$DATASET"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 pytorch/main.py train \
    --workspace=$WORKSPACE \
    --dataset=$DATASET \
    --model_type=$MODEL_TYPE \
    --loss_type=$LOSS_TYPE \
    --augmentation='none' \
    --max_note_shift=0 \
    --batch_size=$BATCH_SIZE \
    --learning_rate=$LEARNING_RATE \
    --reduce_iteration=20000 \
    --resume_iteration=0 \
    --early_stop=$EARLY_STOP \
    --cuda \
    ${RESUME_CHECKPOINT:+--resume_checkpoint=$RESUME_CHECKPOINT}

# ============================================================================
# 2. 推理与评估 (Inference & Evaluation) - 可选
# ============================================================================
# 如果你想在训练结束后直接进行评估，可以取消下面代码的注释

# SPLIT="test"
# CHECKPOINT_PATH="$WORKSPACE/checkpoints/main/your_best_model.pth"

# echo "Starting Inference for evaluation..."
# CUDA_VISIBLE_DEVICES=$GPU_ID python3 pytorch/calculate_score_for_paper.py infer_prob \
#     --workspace=$WORKSPACE \
#     --model_type=$MODEL_TYPE \
#     --checkpoint_path=$CHECKPOINT_PATH \
#     --augmentation='none' \
#     --dataset=$DATASET \
#     --split=$SPLIT \
#     --cuda \
#     --save_pkl

# echo "Calculating Metrics..."
# python3 pytorch/calculate_score_for_paper.py calculate_metrics \
#     --workspace=$WORKSPACE \
#     --model_type=$MODEL_TYPE \
#     --augmentation='none' \
#     --dataset=$DATASET \
#     --split=$SPLIT \
#     --checkpoint_path=$CHECKPOINT_PATH