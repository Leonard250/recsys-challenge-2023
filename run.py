import os
import sys
sys.path.append('./')
sys.path.append('./data/')
sys.path.append('./models/')

import warnings
warnings.filterwarnings('ignore')

import shutil
import argparse
import importlib
from tqdm import tqdm
from pytz import timezone
from datetime import datetime
import time

import wandb
import json 
from torch.cuda import amp

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import gc
from tqdm import tqdm_notebook
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, GroupKFold

from data.common import *
from sklearn.metrics import log_loss
import random 
import lightgbm as lgb
import sklearn.metrics as metrics

import os
import wandb
import torch

# Define the function to save the model checkpoint
def save_checkpoint(model, checkpoint_path):
    # Save the LightGBM model using the save_model method
    model.save_model(checkpoint_path)
# In the predict_each_fold function, after training the model, add the save_checkpoint call

def init_everything(cfg):
    """ Init everything
    - init environment 
    - init project
    - init logging
    Args:
        cfg: config file
    Returns:
        None
    """   
    # init project
    version = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M")
    cfg.project = f'{cfg.user_name}.{cfg.project}'
    cfg.name = f'{cfg.config}.seed{cfg.seed}'
    
    print(cfg.api_key)
    # init logging
    if not cfg.no_wandb:
        wandb.login(key=cfg.api_key)
        wandb.init(
            project=cfg.project,
            name=cfg.name,
            config=cfg.__dict__,
            job_type='Train',
            )
    os.makedirs(f'{cfg.save_dir}/{cfg.name}', exist_ok=True)
    shutil.copy(f'configs/{cfg.config}.py', f'{cfg.save_dir}/{cfg.name}/config.py')

def seed_everything(seed=42):
    """Seed everything
    Args:
        seed: seed number
    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def normalized_cross_entropy(y_true, y_pred): 
    N = len(y_true)
    p = sum(y_true) / N
     
    result = 0
    '''
    for true, pred in zip(y_true, y_pred):
        result += (true * np.log(pred) + (1-true) * np.log(1-pred))
    '''
    result = log_loss(y_true, y_pred, normalize=False)
    return -(1/N)*result*(1/(p*np.log(p) + (1-p)*(np.log(1-p))))
    
def plot_learning_curve(train_loss, valid_loss):
    """
    Plot the learning curve for training and validation loss.
    
    Args:
        train_loss: List of training loss values.
        valid_loss: List of validation loss values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(valid_loss, label='Validation Loss', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Logloss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()



def predict_each_fold(cfg, train_df, valid_df, test_df, is_feat_eng=True, params=None):
    print('predict_each_fold', cfg.label_col)
    if is_feat_eng:
        features, train_df, valid_df, test_df = get_features(
            cfg, 
            train_df,
            valid_df,
            test_df
        )
    else:
        features = [
            c for c in train_df.columns if c not in [
                'is_clicked', 'is_installed', 'file_name', 'f_0', 'f_1', 'f_7', 'f_7_count_full', 'f_9', 'f_11', 'f_43', 'f_51', 'f_58', 'f_59', 'f_64', 'f_65', 'f_66', 'f_67', 'f_68', 'f_69', 'f_70'
                ] + cfg.delete_features
            ]
            
    # get dataloader
    trn_lgb_data, val_lgb_data = get_dataloader(
        cfg, 
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        features=features,
        label_col=cfg.label_col
        )
    
    # get model 
    evals = {}
    params["random_state"] = cfg.seed
    clf = lgb.train(
        params, trn_lgb_data, cfg.num_iterations, valid_sets = [trn_lgb_data, val_lgb_data], 
        categorical_feature = cfg.categorical_features, 
        callbacks = [
            lgb.early_stopping(stopping_rounds=300), 
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(evals)
            ]
    )
    
    # Log best validation score
    best_valid_score = clf.best_score['valid_1']['binary_logloss']
    print(f"Best validation score (log loss): {best_valid_score}")
    
    # Save checkpoint to Kaggle and WandB
    checkpoint_path = save_checkpoint(
        clf, epoch=cfg.num_iterations, fold=0, score=best_valid_score, 
        save_dir=cfg.save_dir, is_kaggle=True
    )

    train_loss = evals_result['training']['binary_logloss']
    valid_loss = evals_result['valid_1']['binary_logloss']

    checkpoint_path = 'model_checkpoint.txt'  # Define your checkpoint path
    save_checkpoint(clf, checkpoint_path)

    # Plot learning curves
    plot_learning_curve(train_loss, valid_loss)
    
    start_time = time.time()
    preds = clf.predict(valid_df[features])
    end_time = time.time()

    inference_time = end_time - start_time

    print("Inference time is", inference_time)   
    logs = {
        'Train Loss': clf.best_score['training']['binary_logloss'],
        'Valid Loss': clf.best_score['valid_1']['binary_logloss'],
        'Valid Metric': normalized_cross_entropy(valid_df[cfg.label_col], preds),
        'Inference Time': inference_time
    }

    return clf, preds, logs, evals, train_df, valid_df, test_df, features

# During model training (in the main function), add a call to save checkpoints as well
def parse_arguments():
    parser = argparse.ArgumentParser(description="recsys challenge 2023")
    parser.add_argument(
        '--config', required=True,
        help="config filename")
    parser.add_argument(
        '--seed', type=int, default=47, 
        help="seed (default: 47)")
    parser.add_argument(
        '--no_wandb', action='store_true', default=False,
        help="no wandb (default: False)")
    parser.add_argument(
        '--verbose', action='store_true', default=False,
        help="verbose or not (default: False)")
    parser.add_argument(
        '--device', default='0', 
        help='cuda device, i.e. 0 or 0,1,2,3 or cpu') 
    parser.add_argument(
        '--smoothing', type=float, default=0.1
    )
    args = parser.parse_args()
    return args
    
def main():
    # parse arguments and load configurations
    args = parse_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.device}'
    cfg = importlib.import_module(f'configs.{args.config}').cfg
    cfg.update(args)
    
    # Initialize everything
    init_everything(cfg)
    print('verbose', cfg.verbose)

    # Print config details for debugging
    if cfg.verbose:
        msg = "Arguments\n"
        for k, v in vars(cfg).items():
            msg += f"  - {k}: {v}\n"
        print(msg)

    # Set random seeds
    seed_everything(cfg.seed)

    # Load raw data
    tr_path = "./train/train.parquet"
    test_path = "./test/test.parquet"
    dfs = load_data(tr_path, test_path)

    # Preprocess raw data
    test_df, train_df = preprocess(cfg, *dfs)

    # Train/Validation Split
    if cfg.split == "time":
        train_df, valid_df = train_valid_split(cfg, train_df)
        clf, _, logs, evals, train_df, valid_df, test_df, features = predict_each_fold(cfg, train_df, valid_df, test_df, is_feat_eng=True, params=cfg.params)
        
        # Plot feature importance
        ax = lgb.plot_importance(clf, max_num_features=60, figsize=(20, 20))    
        ax.figure.savefig(f'{cfg.save_dir}/{cfg.name}/feature_importance.png', dpi=300)
        
        # Evaluate and create the submission
        preds = clf.predict(valid_df[features])
        logs['Valid Loss'] = clf.best_score['valid_1']['binary_logloss']
        score = logs['Valid Loss']
        submission = pd.DataFrame()
        submission["row_id"] = test_df["f_0"]
        submission["is_clicked"] = preds
        submission["is_installed"] = preds
        save_file = f'submission_cv{score}_split{cfg.split}.csv'
        submission.to_csv(f'{cfg.save_dir}/{cfg.name}/{save_file}', sep ='\t', index=False)

    if not cfg.no_wandb:
        wandb.log(logs)

if __name__ == '__main__':
    main()
