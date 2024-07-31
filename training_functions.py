
import utils
import preproc 
import dataloaders 
import model 

from torch.cuda.amp import autocast, GradScaler


def train_fn(train_loader, model, criterion, optimizer, device, CFG):
    """
    Performs one pass through of the training data, calculates loss, and calculates gradients
    """

    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = utils.AverageMeter()

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
    losses = utils.AverageMeter()

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
    For the competition, this is used as the scoring metric
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
    

    #create training and validation dataloaders based on preprocessed data
    train_loader, valid_loader, valid_xyxys = dataloaders.create_dataloaders_new(CFG, full_images ,full_masks ,full_xyxys, fold) 

    #Get validation mask
    valid_mask_gt = utils.valid_mask_gt_func(fold, path_train, CFG) 

    #initialize model, optimizer, scheduler
    model, optimizer, scheduler = model.initalize_model(device, CFG)
    
    #Perform n_epochs of training 
    for epoch in range(CFG.n_epochs):

            start_time = time.time()

            # train model
            avg_loss = train_fn(train_loader, model, model.criterion, optimizer, device, CFG)

            # calculate the validation set loss
            avg_val_loss, mask_pred = valid_fn(valid_loader, model, model.criterion, device, valid_xyxys, valid_mask_gt, CFG)

            #change learning rate based on scheduler
            model.scheduler_step(scheduler, avg_val_loss, epoch)

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
             

def run_training(cfg):
    """
    Takes in cfg, performs preprocessing, and then performs training
    """

    utils.cfg_init(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Gets our paths
    path_train, path_test, path_models, path_logs, path_working = get_paths(detect_env())  
    
    #Preprocessing, read in the fragments
    full_images, full_masks, full_xyxys = preproc.read_all_fragments(cfg, path_train)
    
    log_df_all = None

    #valid_set is for the framgments we want to use in validation
    for fold in cfg.frag_sel: 

        #Train the model. Note the model will be saved down based on 
        log_df = train_one_fold(cfg, full_images, full_masks, full_xyxys, fold, device,
                               path_train, path_test, path_models, path_logs, path_working) #Does one round of training
        
        if log_df_all is None:
            log_df_all = log_df.copy()
        else:
            log_df_all = pd.concat([log_df_all, log_df])
        
    return log_df_all