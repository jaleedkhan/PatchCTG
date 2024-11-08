from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric, classification_metrics
from sklearn.metrics import roc_auc_score 

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import shutil
import time
from datetime import datetime

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # model = model_dict[self.args.model].Model(self.args, num_classes=self.args.num_classes).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        #model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # Only optimize the parameters that require gradients
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        model_optim = optim.Adam(trainable_params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_preds = []  
        all_trues = []  
        self.model.eval()
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y) in enumerate(vali_loader):

                # Move data to GPU here
                batch_x = batch_x.to(self.device)
                #batch_x = batch_x.float().to(self.device)
                #batch_y = batch_y.float()

                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                #dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                outputs = self.model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                all_preds.extend(outputs.detach().cpu().numpy()) 
                all_trues.extend(batch_y.detach().cpu().numpy()) 
                
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         if 'Linear' in self.args.model or 'TST' in self.args.model:
                #             outputs = self.model(batch_x)
                #         else:
                #             if self.args.output_attention:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #             else:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # else:
                #     if 'Linear' in self.args.model or 'TST' in self.args.model:
                #         outputs = self.model(batch_x)
                #     else:
                #         if self.args.output_attention:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #         else:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()

                # loss = criterion(pred, true)

                # total_loss.append(loss)
        total_loss = np.average(total_loss)
        # Check for NaN in predictions or ground truth
        if np.isnan(all_trues).any():
            print("Ground truth contains NaN values. Stopping trial.")
            return float('nan'), float('nan')
        if np.isnan(all_preds).any():
            print("Predictions contain NaN values. Stopping trial.")
            return float('nan'), float('nan')
        auc = roc_auc_score(all_trues, all_preds)
        self.model.train()
        torch.cuda.empty_cache()  # Clear cache after validation
        return total_loss, auc

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')  
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Load pre-trained checkpoint if specified
        if self.args.pre_train_model_path:
            checkpoint_path = os.path.join(self.args.pre_train_model_path, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                print(f"Loading pre-trained model from {checkpoint_path} for fine-tuning...")
                try:
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    for name, param in self.model.named_parameters():
                        # Option 1 - Freeze all layers except the classification head (linear probing)
                        if 'classification_head' not in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
        
                        # Option 2 - Don't freeze any layers (train entire model during finetuning)
                        # param.requires_grad = True
        
                    print("Testing the pretrained model on the new dataset before fine-tuning...")
                    self.test(setting, test=1)  
                except FileNotFoundError:
                    print(f"Checkpoint not found at {checkpoint_path}, training from scratch.")
                except Exception as e:
                    print(f"Error loading checkpoint: {e}, training from scratch.")
            else:
                print(f"No checkpoint.pth found in {self.args.pre_train_model_path}, training from scratch.")

        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        print("Trainable parameters:", trainable_params)

        torch.cuda.empty_cache()
        
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for i, (batch_x, batch_y) in enumerate(train_loader):
                
                iter_count += 1
                model_optim.zero_grad()

                # Move data to GPU here
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                #batch_x = batch_x.float().to(self.device)
                #batch_y = batch_y.float().to(self.device)
                
                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                #dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x).squeeze()
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if torch.isnan(loss).any():
                    print("NaN detected in loss. Stopping trial.")
                    return float('nan')

                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         if 'Linear' in self.args.model or 'TST' in self.args.model:
                #             outputs = self.model(batch_x)
                #         else:
                #             if self.args.output_attention:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #             else:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                #         f_dim = -1 if self.args.features == 'MS' else 0
                #         outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #         batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #         loss = criterion(outputs, batch_y)
                #         train_loss.append(loss.item())
                # else:
                #     if 'Linear' in self.args.model or 'TST' in self.args.model:
                #             outputs = self.model(batch_x)
                #     else:
                #         if self.args.output_attention:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                #         else:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                #     # print(outputs.shape,batch_y.shape)
                #     f_dim = -1 if self.args.features == 'MS' else 0
                #     outputs = outputs[:, -self.args.pred_len:, f_dim:]
                #     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                #     loss = criterion(outputs, batch_y)
                #     train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient Clipping
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient Clipping
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_auc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_auc = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali AUC: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_auc))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Empty the CUDA cache after training
        torch.cuda.empty_cache()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            if self.args.model_to_test:
                checkpoint_path = os.path.join(self.args.model_to_test, 'checkpoint.pth')
            else:
                checkpoint_path = os.path.join('./checkpoints/ctg/' + setting, 'checkpoint.pth')
            print(f'Loading model from: {checkpoint_path} for testing')
            # Load the model state
            self.model.load_state_dict(torch.load(checkpoint_path))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                #batch_x_mark = batch_x_mark.float().to(self.device)
                #batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                #dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                #dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                outputs = self.model(batch_x).squeeze()
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                
                # if self.args.use_amp:
                #     with torch.cuda.amp.autocast():
                #         if 'Linear' in self.args.model or 'TST' in self.args.model:
                #             outputs = self.model(batch_x)
                #         else:
                #             if self.args.output_attention:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #             else:
                #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # else:
                #     if 'Linear' in self.args.model or 'TST' in self.args.model:
                #             outputs = self.model(batch_x)
                #     else:
                #         if self.args.output_attention:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                #         else:
                #             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # f_dim = -1 if self.args.features == 'MS' else 0
                # # print(outputs.shape,batch_y.shape)
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs.detach().cpu().numpy()
                # batch_y = batch_y.detach().cpu().numpy()

                # pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                # true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                # preds.append(pred)
                # trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    input_array = inputx[0]  
                    trues_array = trues[0]  
                    preds_array = preds[0] 
                    gt = np.concatenate((input_array[0, :, -1], [trues_array[0]]), axis=0)
                    pd = np.concatenate((input_array[0, :, -1], [preds_array[0]]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # input = batch_x.detach().cpu().numpy()
                    # gt = np.concatenate((input[0, :, -1], trues[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], preds[0, :, -1]), axis=0)
                    # visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            # # Debugging steps
            # print("Type of preds:", type(preds))
            # print("Type of trues:", type(trues))
            
            # if isinstance(preds, list):
            #     print("Length of preds list:", len(preds))
            #     print("Shape of first preds element:", preds[0].shape)
            
            # if isinstance(trues, list):
            #     print("Length of trues list:", len(trues))
            #     print("Shape of first trues element:", trues[0].shape)
            
            # print("Sample content of preds[0]:", preds[0])
            # print("Sample content of trues[0]:", trues[0])

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        
        # preds = np.array(preds)
        # trues = np.array(trues)
        # inputx = np.array(inputx)

        # Concatenate all batches
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        torch.cuda.empty_cache()

        #preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        #trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        accuracy, auc, sensitivity, specificity, ppv, npv, f1 = classification_metrics(preds, trues)
        print('Accuracy: {}, AUC: {}, Sensitivity: {}, Specificity: {}, PPV: {}, NPV: {}, F1: {}'.format(accuracy, auc, sensitivity, specificity, ppv, npv, f1))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('Accuracy: {}, AUC: {}, Sensitivity: {}, Specificity: {}, PPV: {}, NPV: {}, F1: {}'.format(accuracy, auc, sensitivity, specificity, ppv, npv, f1))
            f.write('\n\n')

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'preds.npy', preds)
        np.save(folder_path + 'trues.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        # Save model, settings and results to jResults
        timestamp = datetime.now().strftime('%Y%m%d %H%M')
        
        if self.args.pre_train_model_path:
            results_dir = os.path.join(self.args.dataset_path, f"finetuned {timestamp}")
        elif self.args.model_to_test:
            results_dir = os.path.join(self.args.dataset_path, f"tested {timestamp}")
        else:
            results_dir = os.path.join(self.args.dataset_path, f"trained {timestamp}")

        os.makedirs(results_dir, exist_ok=True)
        np.save(os.path.join(results_dir, 'preds.npy'), preds)
        np.save(os.path.join(results_dir, 'trues.npy'), trues)
        shutil.copyfile(os.path.join('./checkpoints/ctg/' + setting, 'checkpoint.pth'), os.path.join(results_dir, 'checkpoint.pth'))
        shutil.copyfile(os.path.join('./logs/CTG/PatchTST_ctg_960.log'), os.path.join(results_dir, 'PatchTST_ctg_960.log'))
                
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x).squeeze()
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x).squeeze()
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
