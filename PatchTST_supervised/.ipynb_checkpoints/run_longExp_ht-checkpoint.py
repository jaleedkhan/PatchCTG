# To resume an on-going tuning experiment: python run_longExp_ht.py --resume /home/jaleedkhan/patchctg/PatchTST_supervised/jResults/optuna/20240909\ 1452/
# To run, python run_longExp_ht.py --dataset "../ctg_dataset/Old Dataset"

import optuna
import argparse
import os
import torch
import torch.multiprocessing as mp
from exp.exp_main import Exp_Main
import random
import numpy as np
from datetime import datetime
import sys
sys.path.append('..')
from jDataResultsAnalysis import explore_optuna_results

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

        e_layers = trial.suggest_int('e_layers', 3, 6)
        n_heads = trial.suggest_categorical('n_heads', [4, 8, 16, 32]) 
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512, 640, 768]) #* # should be divisible by n_heads
        d_ff = trial.suggest_categorical('d_ff', [128, 256, 384, 512, 640, 768]) 
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1) 
        fc_dropout = trial.suggest_float('fc_dropout', 0.1, 0.5, step=0.1)
        head_dropout = trial.suggest_float('head_dropout', 0.1, 0.5, step=0.1)
        lr = trial.suggest_categorical('learning_rate', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64])
        
        patch_len = trial.suggest_categorical('patch_len', [4, 8, 16, 32])
        stride = trial.suggest_categorical('stride', [4, 8, 16])
        kernel_size = trial.suggest_categorical('kernel_size', [15, 25, 31])
        embed_type = 0
        decomposition = 0
        distil = True
        revin = 1
        activation = trial.suggest_categorical('activation', ['relu', 'gelu','elu'])
        
        # e_layers = trial.suggest_int('e_layers', 3, 5)
        # n_heads = trial.suggest_categorical('n_heads', [8, 16, 32]) 
        # d_model = trial.suggest_categorical('d_model', [64, 128, 192, 256, 384, 512]) #* # should be divisible by n_heads
        # d_ff = trial.suggest_categorical('d_ff', [128, 256, 320, 384, 512, 640]) #*
        # dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1) 
        # fc_dropout = trial.suggest_float('fc_dropout', 0.1, 0.5, step=0.1)
        # head_dropout = trial.suggest_float('head_dropout', 0.0, 0.2, step=0.1)
        # lr = trial.suggest_categorical('learning_rate', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        # batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64])
        
        # patch_len = trial.suggest_categorical('patch_len', [4, 8, 16])
        # stride = trial.suggest_categorical('stride', [4, 8, 16])
        # kernel_size = trial.suggest_categorical('kernel_size', [15, 25, 31])
        # embed_type = trial.suggest_categorical('embed_type', [0, 1, 2, 3, 4])
        # decomposition = 0
        # distil = trial.suggest_categorical('distil', [True, False])
        # activation = trial.suggest_categorical('activation', ['relu', 'gelu', 'elu'])
    
        # Set up experiment with the suggested hyperparameters
        args = argparse.Namespace(
            random_seed=2021,
            is_training=1,
            model_id='ctg',
            model='PatchTST',
            data='CTG',
            root_path='./dataset/',
            data_path='X.npy',
            dataset_path=dataset_path,  
            features='M',
            seq_len=960,
            enc_in=2,
            num_classes=2,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            revin=revin,
            dropout=dropout,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            patch_len=patch_len,
            stride=stride,
            padding_patch='end',
            affine=0,
            subtract_last=0,
            decomposition=decomposition,
            kernel_size=kernel_size,
            individual=0,
            embed_type=embed_type,
            c_out=1,
            distil=distil,
            activation=activation,
            embed='timeF',
            output_attention=False,
            train_epochs=60,  
            patience=10,
            itr=1,
            batch_size=batch_size,
            learning_rate=lr,
            checkpoints='./checkpoints/ctg',
            use_gpu=True,
            gpu=0,
            use_multi_gpu=True,
            devices='0,1',
            num_workers=2,
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
        torch.cuda.empty_cache() # Clear CUDA cache after every trial

def save_csv_callback(study, optuna_dir):
    def callback(study, trial):
        study.trials_dataframe().to_csv(f"{optuna_dir}/optuna_study_results.csv", index=False)
    return callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument('--resume', type=str, help="Path to existing optuna directory to resume from, else leave empty to start new.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset directory.")
    args = parser.parse_args()
    dataset_path = args.dataset

    # Adjust max_split_size_mb to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    # Check if resuming or starting a new study
    if args.resume:
        optuna_dir = args.resume
        print(f"Resuming from {optuna_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        optuna_dir = os.path.join(args.dataset, "optuna", timestamp)
        os.makedirs(optuna_dir, exist_ok=True)

    # Create or load the Optuna study
    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{optuna_dir}/optuna_study_results.db", study_name="patchtst_ctg_study", load_if_exists=True)

    # Run the optimization process
    study.optimize(objective, n_trials=100, callbacks=[save_csv_callback(study, optuna_dir)])

    # Output the best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    # Save the study results at the end
    study.trials_dataframe().to_csv(f"{optuna_dir}/optuna_study_results_final.csv")

    # Generate and save plots using explore_optuna_results
    explore_optuna_results(
        study_name="patchtst_ctg_study",
        sqlite_path=f"{optuna_dir}/optuna_study_results.db"
    )

