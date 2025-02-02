# #!bin/bash
# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$(pwd):$PYTHONPATH
# NUM_FIRST_CLS=90
# PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))
# # PER_FORGET_CLS=10
# # lr=1e-3 # 1e-4?
# # for lr in 1e-2 5e-2 1e-3
# EPOCH=100
# RATIO=0.1
# RANK=8

# TIME=$(date "+%Y%m%d%H%M%S")

# # for lr in 1e-2; do
# #     for shot in 4; do
# #         for beta in 0.05; do
# #             for alpha in 0.005; do
# #                 for weight in 0.005; do
# #                     python3 -u train/train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e $EPOCH \
# #                         -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 20 \
# #                         --outdir ./exps/forget-CL-pos/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}epoch${EPOCH}-${TIME} \
# #                         --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank $RANK --decay-epochs $EPOCH --wandb_group data \
# #                         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# #                         -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
# #                         --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5 --warmup_alpha --big_alpha 0.01  \
# #                           --prototype --pro_f_weight $weight --pro_r_weight 0  --average_weight --ema_epoch 50 --ema_decay 0.9 
# #                 done
# #             done
# #         done
# #     done
# # done

# # # few shot
# # for lr in 1e-2; do
# #     for shot in 4; do
# #         for beta in 0.15; do
# #             for alpha in 0.01; do
# #                 for weight in 0.02; do
# #                     python3 -u train/train_own_forget.py -b 4 -w 0 -d casia100 -n VIT -e $EPOCH \
# #                         -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 20 \
# #                         --outdir ./exps/forget-CL-pos/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}epoch${EPOCH}-${TIME} \
# #                         --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs $EPOCH --wandb_group fewshot \
# #                         --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# #                         -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
# #                         --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5 --warmup_alpha --big_alpha $alpha  \
# #                           --prototype --pro_f_weight $weight --pro_r_weight 0  --average_weight --ema_epoch 50 --ema_decay 0.9 \
# #                          --few_shot --few_shot_num $shot
# #                 done
# #             done
# #         done
# #     done
# # done

# # # 12 layers
# # for lr in 1e-2; do
# #     for shot in 4; do
# #         for beta in 0.03; do
# #             for alpha in 0.005; do
# #                 for weight in 0.001; do
# #                     python3 -u train/train_own_forget.py -b 48 -w 0 -d casia100 -n VIT -e $EPOCH \
# #                         -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 20 \
# #                         --outdir ./exps/forget-CL-pos/ratio${RATIO}$r8start${NUM_FIRST_CLS}forget${PER_FORGET_CLS}lr${lr}beta${beta}alpha${alpha}epoch${EPOCH}-${TIME} \
# #                         --warmup-epochs 0 --lr $lr --num_workers 8 --lora_rank 8 --decay-epochs $EPOCH --wandb_group rebuttal_beta \
# #                         --vit_depth 12 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
# #                         -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth12new-bs480/Backbone_VIT_Epoch_1111_Batch_83320_Time_2024-12-13-17-21_checkpoint.pth \
# #                         --BND 110 --beta $beta --alpha $alpha --min-lr 1e-5 --warmup_alpha --big_alpha 0.01  \
# #                           --prototype --pro_f_weight $weight --pro_r_weight 0.001   --average_weight --ema_epoch 50 --ema_decay 0.9
# #                     # --beta_decay --small_beta 1e-4 --few_shot --few_shot_num $shot
# #                 done
# #             done
# #         done
# #     done
# # done




#!/bin/bash

# Set GPU to use (change to 0 since you only have one GPU)
export CUDA_VISIBLE_DEVICES=0  
export PYTHONPATH=$(pwd):$PYTHONPATH

# Define number of classes to keep and forget
NUM_FIRST_CLS=90
PER_FORGET_CLS=$((100 - $NUM_FIRST_CLS))  # Forgetting 10 classes

# Training hyperparameters
EPOCH=100
RATIO=0.1  # Data ratio for forgetting
RANK=8  # LoRA rank

# Set timestamp for logging
TIME=$(date "+%Y%m%d%H%M%S")

# Run Single-Step Forgetting with GS-LoRA
python3 -u train/train_own_forget.py -b 32 -w 0 -d casia100 -n VIT -e $EPOCH \
    -head CosFace --grouping block --data_ratio $RATIO --alpha_epoch 20 \
    --outdir ./exps/forget-CL-pos/ratio${RATIO}_start${NUM_FIRST_CLS}_forget${PER_FORGET_CLS}_epoch${EPOCH}_${TIME} \
    --warmup-epochs 0 --lr 1e-3 --num_workers 4 --lora_rank $RANK --decay-epochs $EPOCH \
    --vit_depth 6 --num_of_first_cls $NUM_FIRST_CLS --per_forget_cls $PER_FORGET_CLS \
    -r results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6new/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth \
    --BND 110 --beta 0.05 --alpha 0.005 --min-lr 1e-5 --warmup_alpha --big_alpha 0.01 \
    --prototype --pro_f_weight 0.005 --pro_r_weight 0 --average_weight --ema_epoch 50 --ema_decay 0.9

