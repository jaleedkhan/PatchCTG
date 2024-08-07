from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_CTG
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

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
        X = np.load('../../gabriel_data/X.npy')
        y = np.load('../../gabriel_data/y.npy')
        
        # Split the data into training (80%) and testing/validation (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_seed)
        
        if flag == 'train':
            data_set = Data(X_train, y_train)
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        elif flag in ['val', 'test']:
            data_set = Data(X_test, y_test)
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            raise ValueError("Invalid flag passed to data_provider. Should be 'train', 'val', 'test', or 'pred'.")
        
        print(f"{flag} set size: {len(data_set)}")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader

    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
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
        drop_last = True
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
