import cv2
import utils
from tqdm.auto import tqdm
import numpy as np
import gc 


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

def get_train_transforms(cfg):
    """
    Define the transformations performed on training data
    Currently used to implement image augmentations
    """
    # augmentations
    #---------------------
    if cfg.augment == "baseline":
        get_train_aug = utils.get_train_aug_public_baseline
    else:
        get_train_aug = utils.get_train_aug_resize
        
        
    return get_train_aug(cfg.img_size, cfg.frag_len)

def get_valid_transforms(cfg):
    """Define transformations needed for the validation set. Currently only resizes images"""
    get_valid_aug = utils.get_valid_aug_resize
    return get_valid_aug(cfg.img_size, cfg.frag_len)

def get_transforms(data, cfg): 
    """Define the transformation functions we want to apply to each image"""

    if data == 'train':
        aug = get_train_transforms(cfg)
    elif data == 'valid':
        aug = get_valid_transforms(cfg)

    # print(aug)
    return aug