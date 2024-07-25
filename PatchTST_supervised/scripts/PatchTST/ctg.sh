if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/CTG" ]; then
    mkdir ./logs/CTG
fi

seq_len=960
model_name=PatchTST
root_path_name=./dataset/
data_path_name=ctg_data.npz
model_id_name=ctg
data_name=ctg
random_seed=2021

python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --enc_in 2 \
  --num_classes 2 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp' \
  --train_epochs 100 \
  --patience 20 \
  --itr 1 \
  --batch_size 128 \
  --learning_rate 0.0001 >logs/CTG/$model_name'_'$model_id_name'_'$seq_len.log
