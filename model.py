import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.optim import Adam, SGD, AdamW
from warmup_scheduler import GradualWarmupScheduler
import torch


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
    """Defines Binary Cross Entroypy as loss function"""
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    return BCELoss(y_pred, y_true)


