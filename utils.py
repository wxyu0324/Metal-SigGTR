# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os, time, random
import datetime
from tqdm import tqdm

from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn import metrics 
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from noam_opt import *


id_name = 'ID' 
label_name = ['label'] 
metal_list = ["CA", "MG", "MN","FE","CO","ZN","Cu"]
sequence_name = "sequence"
gpus = list(range(torch.cuda.device_count()))
print("Available GPUs", gpus)


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def calculate_auto_pos_weight(df, label_col):

    print(f"--- 正在为任务 {label_col} 计算动态 pos_weight ---")

    all_labels_str = "".join(df[label_col].astype(str).tolist())
    
    num_ones = all_labels_str.count('1')
    num_zeros = all_labels_str.count('0')
    
    if num_ones == 0:
        print("Warning: 训练集中未发现正样本，使用默认权重 1.0")
        return 1.0

    raw_ratio = num_zeros / num_ones

    final_weight = min(raw_ratio, 50.0) 
    
    print(f"统计结果: 正样本(1)={num_ones}, 负样本(0)={num_zeros}")
    print(f"原始比例: {raw_ratio:.2f}, 最终采用 pos_weight: {final_weight:.2f}")
    
    return final_weight


def Metric(preds, labels):
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)

    if len(np.unique(labels)) < 2:
        return 0.0, 0.0, 0.0

    try:
        AUC = roc_auc_score(labels, preds)
    except:
        AUC = 0.0
        
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
  
    AUPRC = average_precision_score(labels, preds)
    

    if len(thresholds) > 0:
        f1_scores = np.divide(2 * (precisions[:-1] * recalls[:-1]), (precisions[:-1] + recalls[:-1] + 1e-10))
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5

    binary_preds = (preds >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds).ravel()
    
    
    tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
    
  
    mcc_num = (tp * tn) - (fn * fp)
    mcc_den = np.sqrt((tp + fn + 1e-6) * (tp + fp + 1e-6) * (tn + fp + 1e-6) * (tn + fn + 1e-6))
    mcc = mcc_num / mcc_den

    return AUC, AUPRC, mcc


def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None


class TaskDataset:
    def __init__(self, df, protein_data, label_name):
        self.df = df
        self.protein_data = protein_data
        self.label_name = label_name

    def __len__(self):
        return (self.df.shape[0])

    def __getitem__(self, idx):
        pdb_id = self.df.loc[idx,'ID']
        protein_X, protein_node_features, protein_masks = self.protein_data[pdb_id]

       
        protein_node_features = torch.nn.functional.normalize(protein_node_features, p=2, dim=-1)

      
        if isinstance(self.label_name, list):
            labels = self.df.loc[idx, self.label_name[0]]
        else:
            labels = self.df.loc[idx, self.label_name]

        if isinstance(labels, str):
            labels = np.array([float(x) for x in labels])
        labels = torch.tensor(labels, dtype=torch.float32)

      
        if torch.isnan(labels).any():
            labels[torch.isnan(labels)] = 0.0
        if torch.isinf(labels).any():
            labels[torch.isinf(labels)] = 0.0

      
        max_len = protein_masks.shape[0]
        if len(labels) < max_len:
            padded_labels = torch.zeros(max_len, dtype=torch.float32)
            padded_labels[:len(labels)] = labels
            labels = padded_labels
        elif len(labels) > max_len:
            labels = labels[:max_len]

        return {
            'PDB_ID': pdb_id,
            'PROTEIN_X': protein_X,
            'PROTEIN_NODE_FEAT': protein_node_features,
            'PROTEIN_MASK': protein_masks,
            'LABEL': labels,
        }

    def collate_fn(self, batch):
        pdb_ids = [item['PDB_ID'] for item in batch]
        protein_X = torch.stack([item['PROTEIN_X'] for item in batch], dim=0)
        protein_node_features = torch.stack([item['PROTEIN_NODE_FEAT'] for item in batch], dim=0)
        protein_masks = torch.stack([item['PROTEIN_MASK'] for item in batch], dim=0)
        labels = torch.stack([item['LABEL'] for item in batch], dim=0)

        return pdb_ids, protein_X, protein_node_features, protein_masks, labels


# main function
def NN_train_and_predict(train, test, protein_data, model_class, config, logit=False, output_root='.../output/', run_id=None, args=None):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if not run_id:
        run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    output_path = output_root + run_id + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    node_features = config['node_features']
    edge_features = config['edge_features']
    hidden_dim = config['hidden_dim']
    num_encoder_layers = config['num_encoder_layers']
    k_neighbors = config['k_neighbors']
    augment_eps = config['augment_eps']
    dropout = config['dropout']

    id_name = config['id_name']
    obj_max = config['obj_max']
    epochs = config['epochs']
    patience = config['patience']
    batch_size = config['batch_size']
    num_samples = config['num_samples']
    folds = config['folds']
    seed = config['seed']

    task = args.task

    if train is not None:
        os.system(f'cp ./*.py {output_path}')

        oof = train[[id_name, sequence_name]]
        oof['fold'] = -1
        if task == "Metal":
            for m in metal_list:
                oof[m] = 0.0
                oof[m] = oof[m].astype(np.float32)
        elif isinstance(label_name, list):
            for l in label_name:
                oof[l] = 0.0
                oof[l] = oof[l].astype(np.float32)
        else:
            oof[label_name] = 0.0
            oof[label_name] = oof[label_name].astype(np.float32)
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w', buffering=1)
        log.write(str(config) + '\n')
        
        current_label_col = label_name[0] if isinstance(label_name, list) else label_name
        dynamic_pos_weight = calculate_auto_pos_weight(train, current_label_col)
        log.write(f'=== 使用动态 BCEWithLogitsLoss (自动权重: {dynamic_pos_weight:.2f}) ===\n')

        all_valid_metric =[]

        kf = KFold(n_splits = folds, shuffle=True, random_state=seed)

        train_folds =[]
        for fold, (train_index, val_index) in enumerate(kf.split(train)):
            print("\n========== Fold " + str(fold + 1) + " ==========")

            train_dataset = TaskDataset(train.loc[train_index].reset_index(drop=True), protein_data, label_name)
            
            sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_samples)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, sampler=sampler, shuffle=False, drop_last=True, num_workers=args.num_workers, prefetch_factor=2)

            valid_dataset = TaskDataset(train.loc[val_index].reset_index(drop=True), protein_data, label_name)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)
            
            if model_class.__name__ in ['GraphTrans', 'MetalSite']: 
                model = model_class(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors, augment_eps, dropout)

            model.cuda()

            optimizer = get_std_opt(task, model.parameters(), hidden_dim)
            for param_group in optimizer.optimizer.param_groups:
                param_group['weight_decay'] = 1e-5

            if len(gpus) > 1:
                model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])

          
            pos_weight_tensor = torch.tensor([dynamic_pos_weight]).cuda()
            loss_tr = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor)

            best_valid_auprc = 0.0
            best_valid_auc = 0.0
            auc_improve_count = 0  
            not_improve_epochs = 0
            
            if args.train:
                for epoch in range(epochs):
                    train_loss = 0.0
                    train_num = 0
                    model.train()

                    train_preds =[]
                    train_Y =[]
                    bar = tqdm(train_dataloader)
                    for i, data in enumerate(bar):
                        optimizer.zero_grad()
                        protein_X, protein_node_features, protein_masks, y = [d.cuda() for d in data[1:]]

                        if task == "Metal":
                            protein_label_masks = protein_masks.repeat(1, 4) if y.shape[-1] == protein_masks.shape[-1] * 4 else protein_masks
                                
                        outputs = model(protein_X, protein_node_features, protein_masks)
                        
                        if task == "Metal":
                            loss = loss_tr(outputs, y) * protein_label_masks
                            loss_sum = loss.sum() / (protein_label_masks.sum() + 1e-6)
                        else:
                            loss = loss_tr(outputs, y) * protein_masks
                            loss_sum = loss.sum() / (protein_masks.sum() + 1e-6)
                        
                        if torch.isnan(loss_sum).any() or torch.isinf(loss_sum).any():
                            optimizer.zero_grad()
                            continue
                        
                        loss_sum.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                        if logit:
                            outputs = outputs.sigmoid()

                        if task == "Metal":
                            train_seq_preds = torch.masked_select(outputs, protein_label_masks.bool()) 
                            train_seq_y = torch.masked_select(y, protein_label_masks.bool())
                        else:
                            train_seq_preds = torch.masked_select(outputs, protein_masks.bool()) 
                            train_seq_y = torch.masked_select(y, protein_masks.bool())

                        train_preds.append(train_seq_preds.detach().cpu().numpy())
                        train_Y.append(train_seq_y.clone().detach().cpu().numpy())

                        train_num += len(train_seq_y)
                        train_loss += len(train_seq_y) * loss_sum.item()

                        bar.set_description('loss: %.4f' % (loss_sum.item()))

                    if train_num > 0:
                        train_loss /= train_num
                        train_preds = np.concatenate(train_preds)
                        train_Y = np.concatenate(train_Y)
                        train_metric = Metric(train_preds, train_Y) 
                    else:
                        continue

                    # Validation
                    model.eval()
                    valid_preds, valid_Y = [], []
                    valid_preds4, valid_Y4 = [[],[],[],[]], [[],[],[],[]]
                    for data in tqdm(valid_dataloader):
                        protein_X, protein_node_features, protein_masks, y =[d.cuda() for d in data[1:]]
                        if task == "Metal":
                            protein_label_masks = protein_masks.repeat(1, 4) if y.shape[-1] == protein_masks.shape[-1] * 4 else protein_masks
                                
                        with torch.no_grad():
                            outputs = model(protein_X, protein_node_features, protein_masks)
                            if logit: outputs = outputs.sigmoid()

                        if task == "Metal":
                            for i in range(4):
                                valid_seq_y = torch.masked_select(y[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool())
                                valid_seq_preds = torch.masked_select(outputs[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool())
                                valid_Y4[i].append(valid_seq_y.detach().cpu().numpy())
                                valid_preds4[i].append(valid_seq_preds.detach().cpu().numpy())
                        else:
                            valid_seq_y = torch.masked_select(y, protein_masks.bool())
                            valid_seq_preds = torch.masked_select(outputs, protein_masks.bool())
                            valid_Y.append(valid_seq_y.detach().cpu().numpy())
                            valid_preds.append(valid_seq_preds.detach().cpu().numpy())

                    if task == "Metal":
                        valid_auprcs, valid_aucs = [], []
                        for i in range(4):
                            vp, vy = np.concatenate(valid_preds4[i]), np.concatenate(valid_Y4[i])
                            auc_i, auprc_i, _ = Metric(vp, vy)
                            valid_auprcs.append(auprc_i); valid_aucs.append(auc_i)
                        current_valid_auprc, current_valid_auc = np.mean(valid_auprcs), np.mean(valid_aucs)
                    else:
                        current_valid_auc, current_valid_auprc, _ = Metric(np.concatenate(valid_preds), np.concatenate(valid_Y))

                    # Monitor
                    if current_valid_auprc > best_valid_auprc:
                        best_valid_auprc, best_valid_auc = current_valid_auprc, current_valid_auc
                        auc_improve_count, not_improve_epochs = 0, 0
                        torch.save(model.module.state_dict() if len(gpus)>1 else model.state_dict(), output_path + f'fold{fold}.ckpt')
                        Write_log(log,f'[epoch {epoch}] lr: {optimizer._rate:.6f}, train_loss: {train_loss:.6f}, valid_auc: {current_valid_auc:.6f}, valid_auprc: {current_valid_auprc:.6f} (Best)')
                    else:
                        not_improve_epochs += 1
                        if current_valid_auc > best_valid_auc:
                            best_valid_auc, auc_improve_count = current_valid_auc, 0
                        else:
                            auc_improve_count += 1
                        Write_log(log,f'[epoch {epoch}] lr: {optimizer._rate:.6f}, train_loss: {train_loss:.6f}, valid_auc: {current_valid_auc:.6f}, valid_auprc: {current_valid_auprc:.6f}, NIE:{not_improve_epochs}')
                        if not_improve_epochs >= patience and auc_improve_count >= 2: break

            # Load Best
            state_dict = torch.load(output_path + f'fold{fold}.ckpt', torch.device('cuda'))
            model.load_state_dict(state_dict)
            model.eval()

            valid_outputs, valid_preds, valid_Y = [], [], []
            valid_preds4, valid_Y4 = [[],[],[],[]], [[],[],[],[]]
            for data in tqdm(valid_dataloader):
                protein_X, protein_node_features, protein_masks, y =[d.cuda() for d in data[1:]]
                if task == "Metal":
                    protein_label_masks = protein_masks.repeat(1, 4) if y.shape[-1] == protein_masks.shape[-1] * 4 else protein_masks
                with torch.no_grad():
                    outputs = model(protein_X, protein_node_features, protein_masks)
                    if logit: outputs = outputs.sigmoid()
                valid_outputs.append(outputs.detach().cpu().numpy())
                if task == "Metal":
                    for i in range(4):
                        valid_Y4[i].append(torch.masked_select(y[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool()).detach().cpu().numpy())
                        valid_preds4[i].append(torch.masked_select(outputs[:,i*1000:(i+1)*1000], protein_label_masks[:,i*1000:(i+1)*1000].bool()).detach().cpu().numpy())
                else:
                    valid_Y.append(torch.masked_select(y, protein_masks.bool()).detach().cpu().numpy())
                    valid_preds.append(torch.masked_select(outputs, protein_masks.bool()).detach().cpu().numpy())

            if task == "Metal":
                va_cs, va_ps = [], []
                for i in range(4):
                    a, p, _ = Metric(np.concatenate(valid_preds4[i]), np.concatenate(valid_Y4[i]))
                    va_cs.append(a); va_ps.append(p)
                final_auc, final_auprc = np.mean(va_cs), np.mean(va_ps)
                all_valid_metric.append([final_auc, final_auprc])
            else:
                final_auc, final_auprc, _ = Metric(np.concatenate(valid_preds), np.concatenate(valid_Y))
                all_valid_metric.append([final_auc, final_auprc])

            Write_log(log,f'[fold {fold}] Final Best Valid AUC: {final_auc:.6f}, AUPRC: {final_auprc:.6f}')
            
            valid_outputs = np.concatenate(valid_outputs)
            if task == "Metal":
                for i in range(4):
                    oof.loc[val_index,metal_list[i]] =[valid_outputs[:,i*1000:(i+1)*1000][j, :len(oof.loc[val_idx,sequence_name])].tolist() for j,val_idx in enumerate(val_index)]
            else:
                oof.loc[val_index,label_name] = [valid_outputs[i, :len(oof.loc[val_idx,sequence_name])].tolist() for i,val_idx in enumerate(val_index)]
            oof.loc[val_index,'fold'] = fold
            train_folds.append(fold)

        mean_valid_auc = np.mean([m[0] for m in all_valid_metric])
        mean_valid_auprc = np.mean([m[1] for m in all_valid_metric])
        Write_log(log,f'All Valid Mean AUC: {mean_valid_auc:.6f}, Mean AUPRC: {mean_valid_auprc:.6f}')
        oof.loc[oof['fold'].isin(train_folds)].to_csv(output_path + 'oof.csv',index=False)

        log_df = pd.DataFrame({'run_id':[run_id],'folds':folds,'mean_auc':[mean_valid_auc],'mean_auprc':[mean_valid_auprc],'remark':f'Auto pos_weight: {dynamic_pos_weight:.2f}'})
        log_df.to_csv(output_root + 'experiment_log.csv', index=False, mode='a', header=not os.path.exists(output_root + 'experiment_log.csv'))

    if test is not None:
        if train is None:
            log = open(output_path + 'test.log','w', buffering=1)
            Write_log(log,str(config)+'\n')
        sub = test[[id_name, sequence_name]]
        if task == "Metal":
            for m in metal_list: sub[m] = 0.0
        elif isinstance(label_name,list):
            for l in label_name: sub[l] = 0.0
        else: sub[label_name] = 0.0
        
        current_label_name = label_name[0] if isinstance(label_name, list) else label_name
        test_dataset = TaskDataset(test, protein_data, label_name)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn, shuffle=False, drop_last=False, num_workers=args.num_workers)
        
        models =[]
        for fold in range(folds):
            if not os.path.exists(output_path + f'fold{fold}.ckpt'): continue
            model = model_class(node_features, edge_features, hidden_dim, num_encoder_layers, k_neighbors, augment_eps, dropout)
            model.cuda()
            model.load_state_dict(torch.load(output_path + f'fold{fold}.ckpt'))
            if len(gpus) > 1: model = nn.DataParallel(model, device_ids=gpus)
            model.eval(); models.append(model)
        
        test_outputs, test_pdb_ids = [], []
        with torch.no_grad():
            for data in tqdm(test_dataloader):
                test_pdb_ids.extend(data[0])
                protein_X, protein_node_features, protein_masks, _ = [d.cuda() for d in data[1:]]
                outputs = torch.stack([m(protein_X, protein_node_features, protein_masks) for m in models], 0).mean(0)
                if logit: outputs = outputs.sigmoid()
                test_outputs.append(outputs.detach().cpu().numpy())

        test_outputs = np.concatenate(test_outputs)
        if task == "Metal":
            for i in range(4):
                sub[metal_list[i]] = [test_outputs[:,i*1000:(i+1)*1000][j, :len(sub.loc[sub[id_name]==test_pdb_ids[j], sequence_name].values[0])].tolist() for j in range(len(test_pdb_ids))]
        else:
            pdb_id_to_pred = {pid: test_outputs[idx, :len(sub.loc[sub[id_name]==pid, sequence_name].values[0])].tolist() for idx, pid in enumerate(test_pdb_ids)}
            sub[current_label_name] = sub[id_name].map(pdb_id_to_pred)
        sub.to_csv(output_path + 'submission.csv',index=False)
        log.close()