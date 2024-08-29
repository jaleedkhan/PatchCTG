from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_CTG
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import os
from datetime import datetime, timedelta
import pandas as pd

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'CTG': Dataset_CTG,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    
    if args.data == 'CTG':
        # Load CTG dataset
        try:
            X = np.load('../../../ctg_dataset/X.npy')
            y = np.load('../../../ctg_dataset/y.npy')
            clinical_data = pd.read_csv('../../../ctg_dataset/clinical_data.csv')
        except:
            X = np.load('../../ctg_dataset/X.npy')
            y = np.load('../../ctg_dataset/y.npy')
            clinical_data = pd.read_csv('../../ctg_dataset/clinical_data.csv')

        # # Subset for debugging
        # selected_indices = np.concatenate([np.random.choice(np.where(y == c)[0], 500, replace=False) for c in [0, 1]])
        # X, y = X[selected_indices], y[selected_indices]
        
        # Split the data into training (80%) and testing/validation (20%) sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_seed)
        X_train, X_test, y_train, y_test, clinical_train, clinical_test = train_test_split(X, y, clinical_data, test_size=0.2, random_state=args.random_seed)

        # Save split dataset for analysis later
        X_train_list = [x for x in X_train]
        y_train_list = y_train.tolist()
        X_test_list = [x for x in X_test]
        y_test_list = y_test.tolist()
        clinical_train['input_signals'] = X_train_list
        clinical_train['label'] = y_train_list
        clinical_test['input_signals'] = X_test_list
        clinical_test['label'] = y_test_list
        timestamp = datetime.now().strftime('%Y%m%d %H%M')
        results_dir = './jResults/' + timestamp
        existing_dir = None # check for any existing directory with a timestamp within 5 minutes of the current timestamp
        for subdir in os.listdir('./jResults/'):
            subdir_path = os.path.join('./jResults/', subdir)
            if os.path.isdir(subdir_path):
                try:
                    subdir_time = datetime.strptime(subdir, '%Y%m%d %H%M')
                    if abs((subdir_time - datetime.now()).total_seconds()) <= 300:
                        existing_dir = subdir_path
                        break
                except ValueError:
                    continue
        if existing_dir:
            results_dir = existing_dir
        else:
            os.makedirs(results_dir, exist_ok=True)
        clinical_train.to_csv(os.path.join(results_dir, 'dataset_train.csv'), index=False)
        clinical_test.to_csv(os.path.join(results_dir, 'dataset_test.csv'), index=False)
        
        if flag == 'train':
            data_set = Data(X_train, y_train)
            shuffle_flag = True
            #drop_last = True
            drop_last = False
            batch_size = args.batch_size
        elif flag in ['val', 'test']:
            data_set = Data(X_test, y_test)
            shuffle_flag = False
            #drop_last = True
            drop_last = False
            batch_size = args.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            raise ValueError("Invalid flag passed to data_provider. Should be 'train', 'val', 'test', or 'pred'.")
        
        print(f"{flag} set size: {len(data_set)}")

        # Set up a generator for the DataLoader that uses CUDA if available
        if torch.cuda.is_available():
            generator = torch.Generator(device='cuda')
        else:
            generator = torch.Generator()

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            generator=generator  # Pass the generator to the DataLoader
        )
        return data_set, data_loader

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        #drop_last = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        #drop_last = True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader

# def data_provider(args, flag):
#     Data = data_dict[args.data]
#     timeenc = 0 if args.embed != 'timeF' else 1

#     if flag == 'test':
#         shuffle_flag = False
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq
#     elif flag == 'pred':
#         shuffle_flag = False
#         drop_last = False
#         batch_size = 1
#         freq = args.freq
#         Data = Dataset_Pred
#     else:
#         shuffle_flag = True
#         drop_last = True
#         batch_size = args.batch_size
#         freq = args.freq

#     data_set = Data(
#         root_path=args.root_path,
#         data_path=args.data_path,
#         flag=flag,
#         size=[args.seq_len, args.label_len, args.pred_len],
#         features=args.features,
#         target=args.target,
#         timeenc=timeenc,
#         freq=freq
#     )
#     print(flag, len(data_set))
#     data_loader = DataLoader(
#         data_set,
#         batch_size=batch_size,
#         shuffle=shuffle_flag,
#         num_workers=args.num_workers,
#         drop_last=drop_last)
#     return data_set, data_loader
