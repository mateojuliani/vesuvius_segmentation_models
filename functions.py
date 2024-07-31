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
from sklearn.metrics import fbeta_score

#| export
#utils
def cpu_stats():
    """Returns CPU memory usage"""
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))

#| export
#utils
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
#utils
def get_paths(run_env = "local_nb"):
    """Returns data, models, and log folder paths based on where you are running the code"""
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
#cfg
class CFG:
    """Configs class to set parameters used in preprocessing and training the model"""
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

#preprc
def get_train_transforms(cfg):
    """
    Define the transformations performed on training data
    Currently used to implement image augmentations
    """
    # augmentations
    #---------------------
    if cfg.augment == "baseline":
        get_train_aug = get_train_aug_public_baseline
    else:
        get_train_aug = get_train_aug_resize
        
        
    return get_train_aug(cfg.img_size, cfg.frag_len)

def get_valid_transforms(cfg):
    """Define transformations needed for the validation set. Currently only resizes images"""
    get_valid_aug = get_valid_aug_resize
    return get_valid_aug(cfg.img_size, cfg.frag_len)


#| export
#utils
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

        
#utils
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

#utls
def make_dirs(cfg):
    """Creates folder for where to save model based on """
    for dir in [CFG.model_dir, CFG.figures_dir, CFG.submission_dir, CFG.log_dir]:
        os.makedirs(dir, exist_ok=True)
#utils       
def cfg_init(cfg, mode='train'):
    set_seed(cfg.random_seed)

#| export
#preproc
def read_image_mask(CFG, fragment_id, path_train): 
    """"
    For an indiviudal fragment id, reads in the individual image slices based on config parameters
    """

    images = []

    #Define the start and end image indices based on config file
    start = CFG.frag_min
    end = CFG.frag_min + CFG.frag_len
    idxs = range(start, end)


    #read in each image and pad depending tile size so we can get even strides
    for i in tqdm(idxs):

        image_path = str(path_train / f"{fragment_id}/surface_volume/{i:02}.tif")
        image = cv2.imread(image_path, 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    
    #stack all the images together 
    images = np.stack(images, axis=2)

    #read in the  the labeled fragment of where there is ink and where there is not ink
    mask_path = str(path_train / f"{fragment_id}/inklabels.png")
    mask = cv2.imread(mask_path, 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    #normalize between 0 - 1
    mask = mask.astype('float32')
    mask /= 255.0
    
    #Reads in the mask which outlines what part of the pictures is a fragment vs not a fragment
    # In other words, part of the picture where there could be ink  
    label_path = str(path_train / f"{fragment_id}/mask.png")
    label = cv2.imread(label_path, 0)
    label = np.pad(label, [(0, pad0), (0, pad1)], constant_values=0)

    #normalize between 0 - 1
    label = label.astype('float32')
    label /= 255.0
    
    return images, mask, label.astype('int8')

#preproc
def read_all_fragments(CFG, path_train): #Gets training/val data 
    """ 
    For each of the 3 fragments, does the following:
        1. Calls the read_image_mask function which reads in the predifined number of slices for each fragment, its mask, and its labels
        2. Breaks each fragment id into smaller tiles. This will increase the amount of training data and reduce memory requirements
    
     """

    full_images = []
    full_masks = []
    full_xyxys = []

    for fragment_id in range(1, 4): 
        
        images = []
        masks = []
        xyxys = []
        
        #Read in the fragement based on id
        image, mask, label = read_image_mask(CFG, fragment_id, path_train)

        #Calculate the starting x and y cordinates for each tile 
        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

        #For a given fragment, iterate through the tiles
        for y1 in y1_list:
            for x1 in x1_list:
                #Calculate the ending x and y cordinate for each tie
                y2 = y1 + CFG.tile_size 
                x2 = x1 + CFG.tile_size
                
                #for a given tile (x1, x2, y1, y2), check to see if the tile has a fragment piece in it
                if np.max(label[y1:y2, x1:x2]) == 1: 
                    images.append(image[y1:y2, x1:x2]) #Save the 3d slices of the tile down - this will be our input data
                    masks.append(mask[y1:y2, x1:x2, None]) #Save the labeled image down
                    xyxys.append([x1, y1, x2, y2]) #Save the cordinates down
        
        #append all the tiles of each fragment id together 
        full_images.append(images)
        full_masks.append(masks)
        full_xyxys.append(xyxys)
        
        del image, mask, label
        gc.collect()

    return full_images, full_masks, full_xyxys

#| export
#utils
def get_transforms(data, cfg): 
    """Define the transformation functions we want to apply to each image"""

    if data == 'train':
        aug = get_train_transforms(cfg)
    elif data == 'valid':
        aug = get_valid_transforms(cfg)

    # print(aug)
    return aug

#dataloaders
class CustomDataset(Dataset):
    """
    Class to define dataloader
    """
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return image, label


#| export dataloaders
  
def create_dataloaders_new(CFG, full_images, full_masks, full_xyxys, valid_id):
    """
    Function to define train / validation dataloaders based on which fragment id is imputed as the valid_id

    """

    #Get validation images
    valid_id_int = int(valid_id) - 1
    
    valid_images = full_images[valid_id_int]
    valid_masks = full_masks[valid_id_int]
    valid_xyxys  = full_xyxys[valid_id_int]
    
    #Get training images 
    if valid_id_int == 0:
        train_images = full_images[1] + full_images[2]
        train_masks = full_masks[1] + full_masks[2]
    elif valid_id_int == 1:
        train_images = full_images[0] + full_images[2]
        train_masks = full_masks[0] + full_masks[2] 
    elif valid_id_int == 2:
        train_images = full_images[0] + full_images[1]
        train_masks = full_masks[0] + full_masks[1]  
    
    #Create dataset based on train / validation data
    train_dataset = CustomDataset(
    train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDataset(
        valid_images, CFG, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

    #Create dataloaders based on training data and validation data
    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.bs,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    return train_loader, valid_loader, valid_xyxys


#| export utils
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
    """
    Custom model class. Currently only supports Unet architecture
    """

    def __init__(self, cfg, weight=None):
        """
        Initialized the model based on configs parameters / inputted weights
        """
        super().__init__()
        self.cfg = cfg

        self.encoder = smp.Unet(
            encoder_name=cfg.checkpoint, 
            encoder_weights=weight,
            in_channels=cfg.frag_len,
            classes=1,
            activation=None,
        )

    def forward(self, image):
        """
        Defines forward pass 
        """
        output = self.encoder(image)
        # output = output.squeeze(-1)
        return output


def build_model(cfg, weight="imagenet"):
    """
    Initializes model based on cfg file and imagenet pretrained weights
    """

    # print('model_name', "Unet") 
    # print('checkpoint', cfg.checkpoint)

    model = CustomModel(cfg, weight)

    return model

#| export
def initalize_model(device, CFG):
    """
    Initializes model, optimizer (AdamW), and learning rate scheduler based on configs
    """

    model = build_model(CFG)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.smp_lr)
    scheduler = get_scheduler(CFG, optimizer)
    return model, optimizer, scheduler

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    Defines learning rate scheduler. Pulled from this kaggle notebook: https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
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
    """
    Defines scheduler based on cfg file.
    Currently supports two options: Cosine scheduler or the scheduler defined in GradualWarmupSchedulerV2
    """

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
    """
    Adjusts the learning rate at the end of each epoch
    """
    scheduler.step()
    #scheduler.step(epoch) #apparently being deprecated


#| export

def criterion(y_pred, y_true):
    """Use Binary Cross Entroypy as loss function"""
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    return BCELoss(y_pred, y_true)

#| export
def train_fn(train_loader, model, criterion, optimizer, device, CFG):
    """
    Performs one pass through of the training data, calculates loss, and calculates gradients
    """

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
    """
    Calculates validation loss and reconstructs entire fragment to see how we would do in leaderboard
    """
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
def dice_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    Dice coefficient calculations based on: https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def calc_dice(mask, mask_pred):
    """
    For a given mask and predicted mask, calculate the dice coefficient.
    This is used in validation.
    """
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    dice = dice_numpy(mask, (mask_pred >= 0.5).astype(int), beta=0.5)
    
    best_dice = dice
    best_th = 0.5 #assumed this is 0.5 vs iterating over different thresholds
    
# Commented this out to save memory 
# Previously this was used to determine what was the best threshold to determine whether a pixel had ink or not
# Tests showed it was typically around .5, so commenting out to save memory / time
#     for th in np.array(range(10, 50+1, 5)) / 100:
        
#         # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
#         dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
#         print(f'th: {th}, fbeta: {dice}')

#         if dice > best_dice:
#             best_dice = dice
#             best_th = th
    
#     Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    
    return best_dice, best_th


def train_one_fold(CFG, full_images, full_masks, full_xyxys, fold, device, path_train, path_test, path_models, path_logs, path_working): #Inputs are config file, underlying data, and fold num 
    """
    Perform training on one fold of cross validation.
    """
    
    #initialize variables to keep track of performance
    best_score = -1
    best_epoch_num = 0
   
    epoc_count_l = []
    train_loss_l = []
    val_loss_l = []
    val_dice_l = []
    best_dice_score_l = []
    best_epoch_l = []
    epoch_time_l = []
    fold_list_l = []

    # timing
    start_all = time.time()
    
    

#     run_id = CFG.run_id
#     grid_id = CFG.grid_id


    #create training and validation dataloaders based on preprocessed data
    train_loader, valid_loader, valid_xyxys = create_dataloaders_new(CFG, full_images ,full_masks ,full_xyxys, fold) 

    #Get validation mask
    valid_mask_gt = valid_mask_gt_func(fold, path_train, CFG) 

    #initialize model, optimizer, scheduler
    model, optimizer, scheduler = initalize_model(device, CFG)
    
    
    
    #Perform n_epochs of training 
    for epoch in range(CFG.n_epochs):

            start_time = time.time()

            # train model
            avg_loss = train_fn(train_loader, model, criterion, optimizer, device, CFG)

            # calculate the validation set loss
            avg_val_loss, mask_pred = valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt, CFG)

            #change learning rate based on scheduler
            scheduler_step(scheduler, avg_val_loss, epoch)

            #Calculate dice for entire mask, not just singular tile and then average
            best_dice, best_th = calc_dice(valid_mask_gt, mask_pred)

            # score = avg_val_loss
            score = best_dice

            elapsed = time.time() - start_time

            #update best variables if we improved dice coeff
            update_best = score > best_score
            if update_best:
                best_loss = avg_val_loss
                best_score = score
                best_epoch_num = epoch

                #if the model improved, save it down
                torch.save({'model': model.state_dict(),
                            'preds': mask_pred},
                            path_models / f'Unet_fold{fold}_best.pth')

            #save down model performance stats
            epoc_count_l.append(epoch)
            train_loss_l.append(avg_loss)
            val_loss_l.append(avg_val_loss)
            val_dice_l.append(best_dice)
            best_dice_score_l.append(best_score)
            best_epoch_l.append(best_epoch_num)
            epoch_time_l.append(elapsed)
            fold_list_l.append(fold)
            
    #Create DF based on training data 
    base_df = pd.DataFrame( 
        list(zip(epoc_count_l ,train_loss_l ,val_loss_l ,val_dice_l ,best_dice_score_l ,best_epoch_l ,epoch_time_l, fold_list_l )),
                      
        columns = ['epoc_count','train_loss','val_loss','val_dice','best_dice','best_epoch','epoch_time', 'fold_number'])
                     
    cfg_df = pd.DataFrame([CFG.__dict__])

    log_df = pd.concat([base_df, cfg_df], axis = 1)
    log_df = log_df.fillna(method="ffill")
    
    return log_df
             

#| export
def run_training(cfg):
    """
    Takes in cfg, performs preprocessing, and then performs training based o
    """

    cfg_init(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Gets our paths
    path_train, path_test, path_models, path_logs, path_working = get_paths(detect_env())  
    
    #Preprocessing, read in the fragments
    full_images, full_masks, full_xyxys = read_all_fragments(cfg, path_train)
    
    log_df_all = None

    #valid_set is for the framgments we want to use in validation
    for fold in cfg.frag_sel: 
        log_df = train_one_fold(cfg, full_images, full_masks, full_xyxys, fold, device,
                               path_train, path_test, path_models, path_logs, path_working) #Does one round of training
        
        if log_df_all is None:
            log_df_all = log_df.copy()
        else:
            log_df_all = pd.concat([log_df_all, log_df])
        
    return log_df_all
        
    