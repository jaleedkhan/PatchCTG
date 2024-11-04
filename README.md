# PatchCTG: Patch Transformer for CTG Classification

This is the codebase for Patch Transformer for CTG classification (PatchCTG). The time series forecasting model, Patch Transformer ([PatchTST](https://github.com/yuqinie98/PatchTST/tree/main/PatchTST_supervised)), is adapted for binary classification of CTGs.

## Related resources
- PatchCTG Manuscript Draft: https://www.overleaf.com/read/ytymztpqtgpf#144035
- PatchCTG Presentation Slides: [Patch Transformer for Antepartum CTG Classification.pptx](https://gitlab.com/oxmat_project/patchctg/-/blob/classification/Slides/Patch%20Transformer%20for%20Antepartum%20CTG%20Classification.pptx?ref_type=heads)
- Presentation slides of the talk on Deep Learning in Antepartum Fetal Monitoring at [SPaM in Labour 2024](https://sites.unica.it/spam2024/program/): [SPaM24 - Jaleed - Deep Learning in Antepartum Fetal Monitoring.pptx](https://gitlab.com/oxmat_project/patchctg/-/blob/classification/Slides/SPaM24%20-%20Jaleed%20-%20Deep%20Learning%20in%20Antepartum%20Fetal%20Monitoring.pptx?ref_type=heads)

## Important Files and Directories
1. `patchctg/ctg_dataset`: Contains the CTG datasets used for model training, validation, testing and finetuning. Each dataset subdirectory includes `.npy` files for Fetal Heart Rate (FHR) signals, Uterine Contraction (TOCO) signals and their corresponding labels and `.csv` files containing the associated clinical data:
    - `X_train_fhr.npy` contains Fetal Heart Rate (FHR) training data (unstandardized).
    - `X_val_fhr.npy` contains FHR validation data (unstandardized)
    - `X_train_toco.npy` conains Uterine Contraction (TOCO) training data (unstandardized)
    - `X_val_toco.npy` contains TOCO validation data (unstandardized)
    - `y_train.npy` contains labels for the training data
    - `y_val.npy` containing labels for the validation data
    - `clinical_train.csv` contains clinical data for the training CTGs
    - `clinical_val.csv` contains clinical data for the validation CTGs
    - After training, finetuning, or testing, model results, logs, and checkpoints are saved in a timestamped subdirectory within the dataset directory.
2. `patchctg/dataset.ipynb`: A Jupyter notebook for analyzing and preprocessing CTG datasets.
3. `patchctg/PatchTST_supervised`: Contains the implementation of the Patch Transformer model adapted for supervised binary classification of CTGs.
4. `patchctg/PatchTST_supervised/models/PatchTST.py`: Implements the core model architecture, including the definition of transformer layers and the classification head for binary classification.
5. `patchctg/PatchTST_supervised/layers/PatchTST_backbone.py`: Defines the backbone architecture of the model that handles the feature extraction part of the model for sequential CTG data.
6. `patchctg/PatchTST_supervised/layers/PatchTST_layers.py`: Provides support functions and modules for transformer layers, including positional encoding and attention mechanisms used by the backbone.
7. `patchctg/PatchTST_supervised/data_provider/data_factory.py`: Manages the data loading pipeline, including preprocessing the CTG data.
8. `patchctg/PatchTST_supervised/data_provider/data_loader.py`: Defines custom data loading classes to handle specific dataset requirements, such as CTG features and labels, making sure data is accessible to the model during training.
9. `patchctg/PatchTST_supervised/utils/metrics.py`: Contains implementations of the metrics used to evaluate model performance, such as accuracy, AUC, sensitivity, specificity, PPV, NPV and F1 score, for binary classification.
10. `patchctg/PatchTST_supervised/exp/exp_main.py`: The main experiment script that orchestrates model training, validation and testing. It integrates the PatchTST model with the data and manages the overall training pipeline.
11. `patchctg/PatchTST_supervised/run_longExp.py`: The main script for executing long experiments, including model training and evaluation for multiple iterations. 
12. `patchctg/PatchTST_supervised/run_longExp_ht.py`: Similar to run_longExp.py, but used for hyperparameter tuning experiments, to find the best set of hyperparameters for CTG classification.
13. `patchctg/PatchTST_supervised/scripts/PatchTST/ctg.sh`: A bash script for automating the training process, including setting up paths, defining hyperparameters and running training and testing commands. The arguments in `ctg.sh` script can be updated as per your dataset, pre-trained model or hyperparameters requirements before running each experiment. The script has separate commands for training, finetuning and testing. To run a specific mode, uncomment the relevant command (marked as `# SCRIPT 1: TRAIN`, `# SCRIPT 2: FINETUNE`, or `# SCRIPT 3: TEST`) and comment out the others.
14. `patchctg/PatchTST_supervised/logs/CTG`: Stores log files generated during model training and testing, which help in tracking the training progress, hyperparameters and any issues that arise.
15. `patchctg/check_results.ipynb`: A Jupyter notebook to visualize and analyze the results of a trained/finetuned PatchCTG model and interrogate the results across confounders.
16. `patchctg/check_results_ht.ipynb`: Similar to `check_results.ipynb`, this notebook is used for evaluating hyperparameter tuning experiments.
17. `patchctg/Slides`: Powerpoint slides on this work 

## Running the PatchCTG model

### Setup
1. Clone this repository (classification branch): `git clone -b classification https://gitlab.com/oxmat_project/patchctg.git`
2. Datasets, along with results obtained using them, are located in `~/patchctg/ctg_dataset`. A dataset with ~20k CTGs and associated clinical data is available in `~/patchctg/ctg_dataset/Old Dataset`. To experiment with a new dataset, you can create a new directory in `~/patchctg/ctg_dataset` for your dataset, e.g. `~/patchctg/ctg_dataset/my_dataset`, and copy your dataset (or dataset fold) files, including X_train_fhr.npy, X_train_toco.npy, X_val_fhr.npy, X_val_toco.npy, y_train.npy, y_val.npy and clinical_data.csv, to the new directory. When you run an experiment following the following steps, the results will be saved to the dataset directory in a subdirectory with timestamp as name.
3. Install PyTorch 1.11 if not aleady installed: `conda install pytorch=1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch`. This repository requires PyTorch 1.11, and has been successfully tested with Ubuntu 22.04, GCC 10.5, NVIDIA driver version 560.35, CUDA 11.3, Python 3.8 and torch 1.11.05cu113).
4. Navigate to ~/patchctg/PatchTST_supervised: `cd ~/patchctg/PatchTST_supervised`
5. Install the required packages: `pip install -r requirements.txt`.

### Training
1. Navigate to ~/patchctg/PatchTST_supervised: `cd ~/patchctg/PatchTST_supervised`
2. Uncomment the python command with comment `# SCRIPT 1: TRAIN`. Comment out the other two python commands. 
3. The path to the dataset (including train and val/test sets) is set to `~/patchctg/ctg_dataset/Old Dataset` and the hyperparameters tuned for this dataset are set in the script `scripts/PatchTST/ctg.sh`. You can update the dataset path (`--dataset_path` argument) and hyperparameters for training in this script if needed.
4. Run `sh scripts/PatchTST/ctg.sh` to train and test the model. Training progress is logged in `~/patchctg/PatchTST_supervised/logs/CTG`. Upon completion, results and checkpoints are saved in a timestamped subdirectory within the dataset directory.
5. Run the cells in `~/patchctg/check_results.ipynb` notebook to see the dataset statistics, hyperparameters and results after the experiment has completed. The dataset_path variable is set to `ctg_dataset/Old Dataset/` in this notebook, which you can update to your dataset path if needed. 

### Finetuning
1. Navigate to ~/patchctg/PatchTST_supervised: `cd ~/patchctg/PatchTST_supervised`
2. Uncomment the python command with comment `# SCRIPT 2: FINETUNE`. Comment out the other two python commands.
3. The pretrained model path is set to `~/patchctg/ctg_dataset/Old Dataset (Cases Diff 3-7)/trained 20241019 0209` in the script `scripts/PatchTST/ctg.sh`. You can update it using the `--pre_train_model_path` argument if needed.
4. The path to the dataset (including train and val/test sets) is set to `~/patchctg/ctg_dataset/Old Dataset (Cases Diff 0-2)` and the hyperparameters are set in the script `scripts/PatchTST/ctg.sh`. You can update the dataset path (`--dataset_path` argument) and hyperparameters for training in this script if needed.
5. Run `sh scripts/PatchTST/ctg.sh` to finetune and test the model. Training progress is logged in `~/patchctg/PatchTST_supervised/logs/CTG`. Upon completion, results and checkpoints are saved in a timestamped subdirectory within the dataset directory.
6. Run the cells in `~/patchctg/check_results.ipynb` notebook to see the dataset statistics, hyperparameters and results after the experiment has completed. The dataset_path variable is set to `ctg_dataset/Old Dataset/` in this notebook, which you can update to `~/patchctg/ctg_dataset/Old Dataset (Cases Diff 0-2)` or your dataset path. 

### Testing 
1. Navigate to ~/patchctg/PatchTST_supervised: `cd ~/patchctg/PatchTST_supervised`
2. Uncomment the python command with comment `# SCRIPT 3: TEST`. Comment out the other two python commands.
3. The trained model path is set to `~/patchctg/ctg_dataset/Old Dataset (Cases Diff 3-7)/trained 20241019 0209` in the script `scripts/PatchTST/ctg.sh`. You can update it using the `--model_to_test` argument if needed.
4. The path to the dataset (including val/test set) is set to `~/patchctg/ctg_dataset/Old Dataset (Cases Diff 0-2)` in the script `scripts/PatchTST/ctg.sh`. You can update the dataset path (`--dataset_path` argument) in this script if needed.
5. Run `sh scripts/PatchTST/ctg.sh` to test the model. Upon completion, results are saved in a timestamped subdirectory within the dataset directory.
6. Run the cells in `~/patchctg/check_results.ipynb` notebook to see the dataset statistics, hyperparameters and results after the experiment has completed. The dataset_path variable is set to `ctg_dataset/Old Dataset/` in this notebook, which you can update to `~/patchctg/ctg_dataset/Old Dataset (Cases Diff 0-2)` or your dataset path. 

### Hyperparameter Tuning
1. Navigate to ~/patchctg/PatchTST_supervised: `cd ~/patchctg/PatchTST_supervised`
2. Set the hyperparameter tuning search space in `run_longExp_ht.py` (or use the already set search space).
3. Run `python run_longExp_ht.py --dataset "../ctg_dataset/Old Dataset"` to run the hyperparameter tuning experiment on the specified dataset. Update the dataset path if required. A new timestamped subdirectory will be created in a subdirectory named "optuna" within the dataset directory, where the hyperparameter tuning results will be saved. To check the completed trials during execution of this script, check `optuna_study_results.csv` in the timestamped subdirectory.
4. (Optional) To resume an existing or partially completed hyperparameter tuning experiment, run `python run_longExp_ht.py --resume ../ctg_dataset/Old Dataset/optuna/20241008_1152`. Update the path to an existing or partially completed hyperparameter tuning experiment if required.
5. Run the cells in `~/patchctg/check_results_ht.ipynb` notebook to see the hyperparameter tuning results. For more details on the hyperparameters used and performance achieved in each trial, check `optuna_study_results_final.csv` in the timestamped subdirectory after the experiment has completed. 

<!-- ## Updates made in the original repository 

1. **PatchTST.py** (done)
   - Set `target_window=1` during the initialization to ensure proper handling of output dimensions for binary classification
   - Modified the final layer to output a single value with a sigmoid activation for binary classification 

2. **data_factory.py** (done)
   - Added handling for our dataset 
 
3. **data_loader.py** (done)
   - Implemented `Dataset_CTG` class for loading our dataset 

4. **ctg.sh** (done)
   - Created a script to run the binary classification task using our dataset 

5. **exp_main.py** (done)
   - Modified the main experiment script to handle binary classification, including changing the loss function to `nn.BCEWithLogitsLoss()`, adjusting the output processing to handle binary labels, and including validation AUC

6. **metrics.py** (done)
   - Implemented metrics for binary classification, including functions for accuracy, precision, recall, F1-score and AUC 

7. **Update run_longExp.py** (done)
   - Adapt the long experiment script to execute the binary classification task. Change the dataset and model handling for binary classification. Adjust logging to include binary classification metrics.

8. **Verify and Update ctg.sh Script** (done)
   - Ensure the script correctly references all updated files and settings. Confirm paths, model parameters and logging are correctly set for binary classification.

9. **Test and Debug** (done)
   - Test the complete repository for binary classification. -->
  
<!-- # PatchTST (ICLR 2023)

### This is an offical implementation of PatchTST: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730). 

:triangular_flag_on_post: Our model has been included in [GluonTS](https://github.com/awslabs/gluonts). Special thanks to the contributor @[kashif](https://github.com/kashif)!

:triangular_flag_on_post: Our model has been included in [NeuralForecast](https://github.com/Nixtla/neuralforecast). Special thanks to the contributor @[kdgutier](https://github.com/kdgutier) and @[cchallu](https://github.com/cchallu)!

:triangular_flag_on_post: Our model has been included in [timeseriesAI(tsai)](https://github.com/timeseriesAI/tsai/blob/main/tutorial_nbs/15_PatchTST_a_new_transformer_for_LTSF.ipynb). Special thanks to the contributor @[oguiza](https://github.com/oguiza)!

We offer a video that provides a concise overview of our paper for individuals seeking a rapid comprehension of its contents: https://www.youtube.com/watch?v=Z3-NrohddJw



## Key Designs

:star2: **Patching**: segmentation of time series into subseries-level patches which are served as input tokens to Transformer.

:star2: **Channel-independence**: each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/model.png)

## Results

### Supervised Learning

Compared with the best results that Transformer-based models can offer, PatchTST/64 achieves an overall **21.0%** reduction on MSE and **16.7%** reduction
on MAE, while PatchTST/42 attains a overall **20.2%** reduction on MSE and **16.4%** reduction on MAE. It also outperforms other non-Transformer-based models like DLinear.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table3.png)

### Self-supervised Learning

We do comparison with other supervised and self-supervised models, and self-supervised PatchTST is able to outperform all the baselines. 

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table4.png)

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table6.png)

We also test the capability of transfering the pre-trained model to downstream tasks.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table5.png)

## Efficiency on Long Look-back Windows

Our PatchTST consistently <ins>reduces the MSE scores as the look-back window increases</ins>, which confirms our model’s capability to learn from longer receptive field.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/varying_L.png)

## Getting Started

We seperate our codes for supervised learning and self-supervised learning into 2 folders: ```PatchTST_supervised``` and ```PatchTST_self_supervised```. Please choose the one that you want to work with.

### Supervised Learning

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/PatchTST```. The default model is PatchTST/42. For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/PatchTST/weather.sh
```

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). We also provide codes for the baseline models.

### Self-supervised Learning

1. Follow the first 2 steps above

2. Pre-training: The scirpt patchtst_pretrain.py is to train the PatchTST/64. To run the code with a single GPU on ettm1, just run the following command
```
python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
```
The model will be saved to the saved_model folder for the downstream tasks. There are several other parameters can be set in the patchtst_pretrain.py script.
 
 3. Fine-tuning: The script patchtst_finetune.py is for fine-tuning step. Either linear_probing or fine-tune the entire network can be applied.
```
python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAZiqing/FEDformer

https://github.com/alipay/Pyraformer

https://github.com/ts-kim/RevIN

https://github.com/timeseriesAI/tsai

## Contact

If you have any questions or concerns, please contact us: ynie@princeton.edu or nnguyen@us.ibm.com or submit an issue

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

```
@inproceedings{Yuqietal-2023-PatchTST,
  title     = {A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
  author    = {Nie, Yuqi and
               H. Nguyen, Nam and
               Sinthong, Phanwadee and 
               Kalagnanam, Jayant},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
``` -->

