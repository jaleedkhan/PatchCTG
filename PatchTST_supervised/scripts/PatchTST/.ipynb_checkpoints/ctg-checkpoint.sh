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

# train-test 
# ../ctg_dataset/Old Dataset
# ../ctg_dataset/Dataset with Folds/model_datasets/fold_0

python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --dataset_path "../ctg_dataset/Old Dataset" \
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
  --train_epochs 100 \
  --patience 25 \
  --itr 1 \
  --batch_size 48 \
  --learning_rate 0.0001 \
  --checkpoints ./checkpoints/$model_id_name >logs/CTG/$model_name'_'$model_id_name'_'$seq_len.log 

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_0" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_1" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_2" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_3" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_4" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_5" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_6" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_7" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_8" \
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

# python -u run_longExp.py \
#   --random_seed $random_seed \
#   --is_training 1 \
#   --root_path $root_path_name \
#   --dataset_path "../ctg_dataset/Dataset with Folds/model_datasets/fold_9" \
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