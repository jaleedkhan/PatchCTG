import optuna
import argparse
import os
import torch
import torch.multiprocessing as mp
from exp.exp_main import Exp_Main
import random
import numpy as np
from datetime import datetime

# Use 'spawn' for multiprocessing (compatible with CUDA)
mp.set_start_method('spawn', force=True)

mp.set_sharing_strategy('file_system')

# Set the default tensor type to CUDA
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Override the `to()` method, ensuring it doesn't recursively call itself
original_to = torch.nn.Module.to

def to_cuda(self, *args, **kwargs):
    return original_to(self, *args, **kwargs)

torch.nn.Module.to = to_cuda

# Patch the `__init__` method to move models to the GPU without recursion
original_init = torch.nn.Module.__init__

def new_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)  # Call the original __init__
    original_to(self, device)  # Move the model to the device using the original `to` method

torch.nn.Module.__init__ = new_init

def objective(trial):
    try:
        # Suggest hyperparameters for optimization
        e_layers = trial.suggest_int('e_layers', 2, 4)
        #n_heads = trial.suggest_categorical('n_heads', [8, 16, 32])
        #d_model = trial.suggest_int('d_model', 64, 256, step=64) # should be divisible by n_heads
        #d_ff = trial.suggest_int('d_ff', 128, 512, step=128)
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8, 16]) 
        d_model = trial.suggest_categorical('d_model', [16, 64, 128]) 
        d_ff = trial.suggest_categorical('d_ff', [64, 128, 256]) 
        dropout = trial.suggest_float('dropout', 0.3, 0.5, step=0.1)
        fc_dropout = trial.suggest_float('fc_dropout', 0.1, 0.3, step=0.1)
        head_dropout = trial.suggest_float('head_dropout', 0.0, 0.2, step=0.1)
        lr = trial.suggest_loguniform('learning_rate', 1e-7, 1e-4)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
        # Set up experiment with the suggested hyperparameters
        args = argparse.Namespace(
            random_seed=2021,
            is_training=1,
            model_id='ctg',
            model='PatchTST',
            data='CTG',
            root_path='./dataset/',
            data_path='X.npy',
            features='M',
            seq_len=960,
            enc_in=2,
            num_classes=2,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            patch_len=16,
            stride=8,
            padding_patch='end',
            revin=1,
            affine=0,
            subtract_last=0,
            decomposition=0,
            kernel_size=25,
            individual=0,
            embed_type=0,
            c_out=1,
            distil=True,
            activation='gelu',
            embed='timeF',
            output_attention=False,
            train_epochs=50,  
            patience=10,
            itr=1,
            batch_size=batch_size,
            learning_rate=lr,
            checkpoints='./checkpoints/ctg',
            use_gpu=True,
            gpu=0,
            use_multi_gpu=True,
            devices='0,1',
            num_workers=5,
            lradj='type3',
            pct_start=0.3,
            use_amp=True,
            is_optuna=True
        )
    
        # Set the random seed
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
    
        # Run the experiment
        exp = Exp_Main(args)
        setting = f"{args.model_id}_optuna_trial"
    
        # Train the model
        exp.train(setting)
    
        # Validate the model and return the validation AUC
        _, val_auc = exp.vali(exp._get_data('val')[0], exp._get_data('val')[1], exp._select_criterion())
        
        return val_auc
    
    finally:
        torch.cuda.empty_cache()

def save_csv_callback(study, optuna_dir):
    def callback(study, trial):
        study.trials_dataframe().to_csv(f"{optuna_dir}/optuna_study_results.csv", index=False)
    return callback


if __name__ == "__main__":
    # Adjust max_split_size_mb to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' 

    # Ensure the directory for the SQLite database exists
    optuna_dir = f"/home/jaleedkhan/patchctg/PatchTST_supervised/jResults/optuna/{datetime.now().strftime('%Y%m%d %H%M')}/"
    os.makedirs(optuna_dir, exist_ok=True)
    
    # Create an Optuna study to maximize the validation AUC
    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{optuna_dir}/optuna_study_results.db", study_name="patchtst_ctg_study", load_if_exists=True)

    # Run the optimization process
    study.optimize(objective, n_trials=100, callbacks=[save_csv_callback(study, optuna_dir)])

    # Output the best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    # Save the study results at the end
    study.trials_dataframe().to_csv(f"{optuna_dir}/optuna_study_results_final.csv")
