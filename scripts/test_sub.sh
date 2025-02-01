export CUDA_VISIBLE_DEVICES=0
pretrain=./results/ViT-P8S8_casia100_cosface_s1-1200-150de-depth6/Backbone_VIT_Epoch_1185_Batch_45020_Time_2024-09-26-03-26_checkpoint.pth

python test/test_own.py -w 0 \
 --batch_size 2 --lora_rank 0 \
 --model ${pretrain}
