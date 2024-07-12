#!/bin/bash

# Create directories if they don't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Define parameters
seq_len=48  # Further reduced sequence length
model_name=PatchTST

root_path_name=./dataset/weather/
data_path_name=weather.csv
model_id_name=weather_quick_test
data_name=custom

random_seed=2021
pred_len=24  # Further reduced prediction length

# Run the model with specified parameters
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --e_layers 1 \
  --n_heads 4 \
  --d_model 32 \
  --d_ff 64 \
  --dropout 0.1 \
  --fc_dropout 0.1 \
  --head_dropout 0 \
  --patch_len 4 \
  --stride 2 \
  --des 'Exp' \
  --train_epochs 1 \
  --patience 1 \
  --itr 1 \
  --batch_size 8 \
  --learning_rate 0.0001 | tee logs/LongForecasting/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log
