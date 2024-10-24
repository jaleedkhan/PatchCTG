#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/CTG" ]; then
    mkdir ./logs/CTG
fi

seq_len=960
model_name=PatchTST
root_path_name=./dataset/
data_path_name=X.npy
model_id_name=ctg
data_name=CTG
random_seed=2021

# Ensure the checkpoint directory exists
if [ ! -d "./checkpoints/$model_id_name" ]; then
    mkdir -p ./checkpoints/$model_id_name
fi

# # SCRIPT 1: TRAIN
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --num_workers 0 \
  --root_path $root_path_name \
  --dataset_path "../ctg_dataset/Old Dataset (Cases Diff 0-2)" \
  --model_id $model_id_name'_'$seq_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --enc_in 2 \
  --num_classes 2 \
  --e_layers 6 \
  --n_heads 4 \
  --d_model 512 \
  --d_ff 128 \
  --dropout 0.1 \
  --fc_dropout 0.4 \
  --head_dropout 0.2 \
  --patch_len 16 \
  --stride 16 \
  --kernel_size 15 \
  --activation 'relu' \
  --des 'Exp' \
  --train_epochs 1 \
  --patience 25 \
  --itr 1 \
  --batch_size 48 \
  --learning_rate 0.0001 \
  --checkpoints ./checkpoints/$model_id_name >logs/CTG/$model_name'_'$model_id_name'_'$seq_len.log 

# # SCRIPT 2: FINETUNE
# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --num_workers 0 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Old Dataset (Cases Diff 0-2)" \
#   --pre_train_model_path "../ctg_dataset/Old Dataset (Cases Diff 3-7)/trained 20241019 0209" \
#   --model_id $model_id_name'_'$seq_len \
#   --model $model_name \
#   --data $data_name \
#   --features M \
#   --seq_len $seq_len \
#   --enc_in 2 \
#   --num_classes 2 \
#   --e_layers 6 \
#   --n_heads 4 \
#   --d_model 512 \
#   --d_ff 128 \
#   --dropout 0.1 \
#   --fc_dropout 0.4 \
#   --head_dropout 0.2 \
#   --patch_len 16 \
#   --stride 16 \
#   --kernel_size 15 \
#   --activation 'relu' \
#   --des 'Exp' \
#   --train_epochs 1 \
#   --patience 25 \
#   --itr 1 \
#   --batch_size 48 \
#   --learning_rate 0.0001 \
#   --checkpoints ./checkpoints/$model_id_name >logs/CTG/$model_name'_'$model_id_name'_'$seq_len.log 

# # SCRIPT 3: TEST
# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 0 \
#   --num_workers 0 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Old Dataset (Cases Diff 0-2)" \
#   --model_to_test "../ctg_dataset/Old Dataset (Cases Diff 3-7)/trained 20241019 0209" \
#   --model_id $model_id_name'_'$seq_len \
#   --model $model_name \
#   --data $data_name \
#   --features M \
#   --seq_len $seq_len \
#   --enc_in 2 \
#   --num_classes 2 \
#   --e_layers 6 \
#   --n_heads 4 \
#   --d_model 512 \
#   --d_ff 128 \
#   --dropout 0.1 \
#   --fc_dropout 0.4 \
#   --head_dropout 0.2 \
#   --patch_len 16 \
#   --stride 16 \
#   --kernel_size 15 \
#   --activation 'relu' \
#   --des 'Exp' \
#   --train_epochs 100 \
#   --patience 25 \
#   --itr 1 \
#   --batch_size 48 \
#   --learning_rate 0.0001 \
#   --checkpoints ./checkpoints/$model_id_name >logs/CTG/$model_name'_'$model_id_name'_'$seq_len.log 