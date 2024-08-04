from torch.utils.data import DataLoader, Dataset
import preproc

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
    train_images, CFG, labels=train_masks, transform=preproc.get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDataset(
        valid_images, CFG, labels=valid_masks, transform=preproc.get_transforms(data='valid', cfg=CFG))

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