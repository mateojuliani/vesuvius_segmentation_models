from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import pickle
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
import sys
import pandas as pd
import os
import gc
import sys
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import cv2

import scipy as sp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial

import argparse
import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW

import datetime
import segmentation_models_pytorch as smp

import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import psutil

import torch.nn as nn
import torch
import math
import time
import numpy as np
import torch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler

#| export
def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

#| export
def detect_env():
    """A helper function that detects where you are running code"""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", False):
        run_env = "kaggle"
    elif os.path.isdir("/content"):
        run_env = "colab"
    elif os.path.isdir("../nbs") or os.path.isdir("../../nbs"):
        run_env = "local_nb"
    else:
        run_env = "script"

    return run_env        

# | export
def get_paths(run_env = "local_nb"):
    """Returns data, models, and log folder paths based on your where you are running the code"""
    if run_env == "kaggle":
        path_main = path_data = Path(f"/kaggle/input/vesuvius-challenge-ink-detection")
        path_working = Path("/kaggle/working/outputs/")

    elif run_env == "colab":
        path_main = path_working = path_data = Path("./")

    elif run_env == "local_nb":
        path_main = path_working = Path("..")
        path_data = path_main / "data"

    elif run_env == "script":
        path_main = path_working = Path(".")
        path_data = path_main / "data"
        
    path_models = path_working / "models"
    path_logs = path_working / "logs"

    try:
        path_models.mkdir(parents=True, exist_ok=True)
        path_logs.mkdir(parents=True, exist_ok=True)
    except:
        print("Unable to create models and logs folders")

    path_train = path_data / "train"
    path_test = path_data / "test/"

    return path_train, path_test, path_models, path_logs, path_working


#| export
class CFG:
    def __init__(self):
        self.random_seed = 4321
        self.subset = 1.0
        self.n_fold = 5
        self.val_fold = 0
        self.img_size = 224
        self.bs = 8
        self.frag_min = 31 # 15
        self.frag_len = 4
        self.merge_img = "1d"  # "1d", "3d", "none"
        self.norm_img = False
        self.frag_sel = ["1", "2", "3"]
        self.fold_split = "stratify"  # "stratify", "fragment"
        self.augment = "baseline" #"bright_dropout_blur"
        self.preproc = "basic"
        self.postproc = "none"
        self.train_folds = "train_folds.csv"
        self.checkpoint = 'tu-eca_nfnet_l1'
        self.loss = "ce_weighted"
        self.metric = "fbeta"
        self.use_fp16 = True
        self.n_epochs = 15  #we're doing early stopping
        self.lr = 5e-5  # number or 'find' to use lr_find
        self.framework = "fastai"
        self.run_id = "null"
        self.grid_id = -1
        self.save_oof = False
        
        self.tile_size = 224
        self.stride = self.tile_size // 2
        self.valid_batch_size = self.bs * 2 #This needs to be added
        self.use_amp = True #need to add this to cfg
        self.scheduler = 'GradualWarmupSchedulerV2'
        self.warmup_factor = 10 #Need to add this 
        self.smp_lr = 1e-4 / self.warmup_factor #Need to add this
        self.max_grad_norm = 1000 #To be added
        self.num_workers = 4 #needs to be added


#| export
#from the util package, but putting here to make it easier
def get_train_aug_resize(img_size, in_chans): return albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.Normalize(mean = [0] * in_chans, std = [1] * in_chans),
    ToTensorV2(transpose_mask=True)
])

def get_valid_aug_resize(img_size, in_chans): return albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.Normalize(mean = [0] * in_chans, std = [1] * in_chans),
    ToTensorV2(transpose_mask=True)
], p=1.)


def get_train_aug_brightness(img_size, in_chans): return albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.RandomBrightnessContrast(p = 0.5),
    albumentations.HueSaturationValue(p = 0.5),
    albumentations.Normalize(mean = [0] * in_chans, std = [1] * in_chans),
    ToTensorV2(transpose_mask=True)
])

def get_train_aug_bright_dropout_geom_blur(img_size, in_chans): return albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.RandomBrightnessContrast(p = 0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Blur(p = 0.5),
    albumentations.CoarseDropout(p = 0.5),
    albumentations.ShiftScaleRotate(p=0.5, rotate_limit=15),
    albumentations.Normalize(mean = [0] * in_chans, std = [1] * in_chans),
    ToTensorV2(transpose_mask=True)
])

def get_train_aug_public_baseline(img_size, in_chans):  return albumentations.Compose([
    albumentations.Resize(img_size, img_size),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.RandomBrightnessContrast(p=0.75),
    albumentations.ShiftScaleRotate(p=0.75),
    albumentations.OneOf([
            albumentations.GaussNoise(var_limit=[10, 50]),
            albumentations.GaussianBlur(),
            albumentations.MotionBlur(),
            ], p=0.4),
    albumentations.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    albumentations.CoarseDropout(max_holes=1, max_width=int(img_size * 0.3), max_height=int(img_size * 0.3), 
                    mask_fill_value=0, p=0.5),
    albumentations.Normalize(mean = [0] * in_chans, std = [1] * in_chans),
    ToTensorV2(transpose_mask=True)
])

def get_train_transforms(cfg):
    
    # augmentations
    #---------------------
    if cfg.augment == "basic":
        get_train_aug = get_train_aug_resize
    elif cfg.augment == "brightness":
        get_train_aug = get_train_aug_brightness
#     elif cfg.augment == "geometry":
#         get_train_aug = get_train_aug_geometry
    elif cfg.augment == "baseline":
        get_train_aug = get_train_aug_public_baseline
    elif cfg.augment == "bright_dropout_geom_blur":
        get_train_aug = get_train_aug_bright_dropout_geom_blur
    else:
        get_train_aug = get_train_aug_resize
        
        
    return get_train_aug(cfg.img_size, cfg.frag_len)

def get_valid_transforms(cfg):
    get_valid_aug = get_valid_aug_resize
    return get_valid_aug(cfg.img_size, cfg.frag_len)


#| export
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
        

def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    
    for dir in [CFG.model_dir, CFG.figures_dir, CFG.submission_dir, CFG.log_dir]:
        os.makedirs(dir, exist_ok=True)
        
def cfg_init(cfg, mode='train'):
    set_seed(cfg.random_seed)

#| export

def read_image_mask(CFG, fragment_id, path_train): ##Reads an individual mask framgment

    images = []

    start = CFG.frag_min
    end = CFG.frag_min + CFG.frag_len
    idxs = range(start, end)

    for i in tqdm(idxs):

        image_path = str(path_train / f"{fragment_id}/surface_volume/{i:02}.tif")
        image = cv2.imread(image_path, 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    mask_path = str(path_train / f"{fragment_id}/inklabels.png")
    mask = cv2.imread(mask_path, 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0
    
    label_path = str(path_train / f"{fragment_id}/mask.png")
    label = cv2.imread(label_path, 0)
    label = np.pad(label, [(0, pad0), (0, pad1)], constant_values=0)

    label = label.astype('float32')
    label /= 255.0
    
    return images, mask, label.astype('int8')

def read_all_fragments(CFG, path_train): #Gets training/val data 
    
    full_images = []
    full_masks = []
    full_xyxys = []

    for fragment_id in range(1, 4): 
        
        images = []
        masks = []
        xyxys = []
        
        image, mask, label = read_image_mask(CFG, fragment_id, path_train)

        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                # xyxys.append((x1, y1, x2, y2))
                
                if np.max(label[y1:y2, x1:x2]) == 1: #we have ink here
                    images.append(image[y1:y2, x1:x2])
                    masks.append(mask[y1:y2, x1:x2, None])
                    xyxys.append([x1, y1, x2, y2])
        
        
        full_images.append(images)
        full_masks.append(masks)
        full_xyxys.append(xyxys)
        
        del image, mask, label
        gc.collect()

    return full_images, full_masks, full_xyxys

#| export

def get_transforms(data, cfg): 
    if data == 'train':
        aug = get_train_transforms(cfg)
    elif data == 'valid':
        aug = get_valid_transforms(cfg)

    # print(aug)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return image, label


#| export
  
def create_dataloaders_new(CFG, full_images, full_masks, full_xyxys, valid_id):
    valid_id_int = int(valid_id) - 1
    
    valid_images = full_images[valid_id_int]
    valid_masks = full_masks[valid_id_int]
    valid_xyxys  = full_xyxys[valid_id_int]
    
    if valid_id_int == 0:
        train_images = full_images[1] + full_images[2]
        train_masks = full_masks[1] + full_masks[2]
    elif valid_id_int == 1:
        train_images = full_images[0] + full_images[2]
        train_masks = full_masks[0] + full_masks[2] 
    elif valid_id_int == 2:
        train_images = full_images[0] + full_images[1]
        train_masks = full_masks[0] + full_masks[1]  
    
    #train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset(valid_id)
    #valid_xyxys = np.stack(valid_xyxys)
    
    train_dataset = CustomDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDataset(
        valid_images, CFG, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))


    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.bs,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    #print(len(train_loader))
    #print(len(valid_loader))
    return train_loader, valid_loader, valid_xyxys


#| export
def valid_mask_gt_func(fragment_id, path_train, CFG):
    valid_mask_path = str(path_train / f"{fragment_id}/inklabels.png")
    valid_mask_gt = cv2.imread(valid_mask_path, 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    return valid_mask_gt.astype('float16')



#| export
class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        self.encoder = smp.Unet(
            encoder_name=cfg.checkpoint, 
            encoder_weights=weight,
            in_channels=cfg.frag_len,
            classes=1, #hard coding for now 
            activation=None,
        )

    def forward(self, image):
        output = self.encoder(image)
        # output = output.squeeze(-1)
        return output


def build_model(cfg, weight="imagenet"):
    print('model_name', "Unet") #used to be cfg.model_name
    print('checkpoint', cfg.checkpoint)

    model = CustomModel(cfg, weight)

    return model

#| export
def initalize_model(device, CFG):
    model = build_model(CFG)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.smp_lr)
    scheduler = get_scheduler(CFG, optimizer)
    return model, optimizer, scheduler

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.n_epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    if cfg.scheduler == 'GradualWarmupSchedulerV2':
        ret_scheduler = scheduler
    elif cfg.scheduler == 'cosine':
        ret_scheduler = scheduler_cosine
    else:
        ret_scheduler = scheduler
    
    return ret_scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)


#| export

def criterion(y_pred, y_true):
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    return BCELoss(y_pred, y_true)

#| export
def train_fn(train_loader, model, criterion, optimizer, device, CFG):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with autocast(CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.avg

def valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt, CFG):
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        start_idx = step*CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

    print(f'mask_count_min: {mask_count.min()}')
    mask_pred /= mask_count
    return losses.avg, mask_pred

#| export
from sklearn.metrics import fbeta_score

def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    dice = fbeta_numpy(mask, (mask_pred >= 0.5).astype(int), beta=0.5)
    
    best_dice = dice
    best_th = 0.5 #assumed this is 0.5 vs iterating over different thresholds
    
# Commented this out to save memory 
#     for th in np.array(range(10, 50+1, 5)) / 100:
        
#         # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
#         dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
#         print(f'th: {th}, fbeta: {dice}')

#         if dice > best_dice:
#             best_dice = dice
#             best_th = th
    
#     Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th

#| export

def train_one_fold(CFG, full_images, full_masks, full_xyxys, fold, device, path_train, path_test, path_models, path_logs, path_working): #Inputs are config file, underlying data, and fold num 
    
    
    best_score = -1
    # timing
    start_all = time.time()
    
    best_epoch_num = 0

#     run_id = CFG.run_id
#     grid_id = CFG.grid_id

    #full_images, full_masks, full_xyxys = read_all_fragments() #reads and puts all fragments in list
    #Note the above code ^^^^ doesnt need to be re-initalized every fold.
    ###Maybe we create it in pre-proc and just pass it thru

    train_loader, valid_loader, valid_xyxys = create_dataloaders_new(CFG, full_images ,full_masks ,full_xyxys, fold) 
    valid_mask_gt = valid_mask_gt_func(fold, path_train, CFG) #Get validation mask
    model, optimizer, scheduler = initalize_model(device, CFG)
    
    epoc_count_l = []
    train_loss_l = []
    val_loss_l = []
    val_dice_l = []
    best_dice_score_l = []
    best_epoch_l = []
    epoch_time_l = []
    fold_list_l = []
    
    for epoch in range(CFG.n_epochs):

            start_time = time.time()

            # train
            avg_loss = train_fn(train_loader, model, criterion, optimizer, device, CFG)

            print(f'post training {fold}: {cpu_stats()}')
            # eval
            avg_val_loss, mask_pred = valid_fn(
                valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt, CFG)

            scheduler_step(scheduler, avg_val_loss, epoch)

            print(f'model end training cv: {cpu_stats()}')
            best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

            print(f'finished calc cv: {cpu_stats()}')
            # score = avg_val_loss
            score = best_dice

            elapsed = time.time() - start_time

#             Logger.info(
#                 f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
#             # Logger.info(f'Epoch {epoch+1} - avgScore: {avg_score:.4f}')
#             Logger.info(
#                 f'Epoch {epoch+1} - avgScore: {score:.4f}')


            update_best = score > best_score

            if update_best:
                best_loss = avg_val_loss
                best_score = score
                best_epoch_num = epoch
#                 Logger.info(
#                     f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
#                 Logger.info(
#                     f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')

                print(f'saving model: {cpu_stats()}')
                torch.save({'model': model.state_dict(),
                            'preds': mask_pred},
                            path_models / f'Unet_fold{fold}_best.pth')
                print(f'model saved: {cpu_stats()}')

            epoc_count_l.append(epoch)
            train_loss_l.append(avg_loss)
            val_loss_l.append(avg_val_loss)
            val_dice_l.append(best_dice)
            best_dice_score_l.append(best_score)
            best_epoch_l.append(best_epoch_num)
            epoch_time_l.append(elapsed)
            fold_list_l.append(fold)
            
            
    base_df = pd.DataFrame( #Create DF based on training data 
        list(zip(epoc_count_l ,train_loss_l ,val_loss_l ,val_dice_l ,best_dice_score_l ,best_epoch_l ,epoch_time_l, fold_list_l )),
                      
        columns = ['epoc_count','train_loss','val_loss','val_dice','best_dice','best_epoch','epoch_time', 'fold_number'])
                     
    cfg_df = pd.DataFrame([CFG.__dict__])

    log_df = pd.concat([base_df, cfg_df], axis = 1)
    log_df = log_df.fillna(method="ffill")
    
    return log_df
             

#| export
def run_grid(cfg):
    #cfg = CFG()
    cfg_init(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    path_train, path_test, path_models, path_logs, path_working = get_paths(detect_env())  #Gets our paths
    
    full_images, full_masks, full_xyxys = read_all_fragments(cfg, path_train) #Preprocessing, read in the fragments
    
    log_df_all = None
    for fold in cfg.frag_sel: #valid_set is for the framgments we want to use in validation
        log_df = train_one_fold(cfg, full_images, full_masks, full_xyxys, fold, device,
                               path_train, path_test, path_models, path_logs, path_working) #Does one round of training
        
        if log_df_all is None:
            log_df_all = log_df.copy()
        else:
            log_df_all = pd.concat([log_df_all, log_df])
        
    return log_df_all
        
    