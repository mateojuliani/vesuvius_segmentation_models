

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
        # self.framework = "fastai"
        # self.run_id = "null"
        # self.grid_id = -1
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


