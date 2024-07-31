
import os
import numpy as np
import torch
import albumentations
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import random
import psutil
from pathlib import Path
import cv2






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

def cfg_init(cfg, mode='train'):
    set_seed(cfg.random_seed)


def cpu_stats():
    """Returns CPU memory usage"""
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'memory GB:' + str(np.round(memory_use, 2))


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


def get_train_aug_resize(img_size, in_chans): 
    """Resizes and normalizes inputed data"""
    
    return albumentations.Compose([
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

#TODO: See if this is actually used 
def make_dirs(cfg):
    """Creates folder for where to save model based on """
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)

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

def valid_mask_gt_func(fragment_id, path_train, CFG):
    valid_mask_path = str(path_train / f"{fragment_id}/inklabels.png")
    valid_mask_gt = cv2.imread(valid_mask_path, 0)
    valid_mask_gt = valid_mask_gt / 255
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    return valid_mask_gt.astype('float16')