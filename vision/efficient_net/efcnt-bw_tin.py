# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import copy
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import models, datasets
from torchvision import transforms as T
# from torchsummary import summary

import os
import argparse
import pickle
from random import randint
import urllib
import zipfile

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from efficientnet_pytorch import EfficientNetBW,BatchWhiteningBlock,comp_avg_corr,get_rank,comp_cov_cond
import wandb
import optuna



# to download the dataset : wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# set directories
DATA_DIR = '/datasets/vision/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
# VALID_DIR = os.path.join(DATA_DIR, 'val')
VALID_DIR = os.path.join(DATA_DIR, 'val')
CHECKPOINT_PATH = "saved_models"
# HPARAM_OPT='TRAIN'
HPARAM_OPT='INFER'
# HPARAM_OPT='OFF'
 

############################################
# TinyImageNet dataset module functions
# Unlike training folder where images are already arranged in sub folders based 
# on their labels, images in validation folder are all inside a single folder. 
# Validation folder comes with images folder and val_annotations txt file. 

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
def prepare_tinyimagenet_validation_folder(val_img_dir):
    # check if the subfolders already exist, and if not, create them:
    n_subfolders = len([d for d in os.listdir(val_img_dir) if os.path.isdir(os.path.join(val_img_dir, d))])
    if n_subfolders == 0:
        print('Creating subfolders for validation images')
        # The val_annotation txt file comprises 6 tab separated columns of filename, 
        # class label, x and y coordinates, height, and width of bounding boxes
        val_data = pd.read_csv(f'{VALID_DIR}/val_annotations.txt', 
                            sep='\t', 
                            header=None, 
                            names=['File', 'Class', 'X', 'Y', 'H', 'W'])



        # Open and read val annotations text file
        fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
        data = fp.readlines()

        # Create dictionary to store img filename (word 0) and corresponding
        # label (word 1) for every line in the txt file (as key value pair)
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()

        # Create subfolders (if not present) for validation images based on label ,
        # and move images into the respective folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(val_img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))

    print('Validation subfolders ready')
    return

def get_class_to_name_dict(data_dir=DATA_DIR):
    # Save class names (for corresponding labels) as dict from words.txt file
    class_to_name_dict = dict()
    fp = open(os.path.join(DATA_DIR, 'words.txt'), 'r')
    data = fp.readlines()
    for line in data:
        words = line.strip('\n').split('\t')
        class_to_name_dict[words[0]] = words[1].split(',')[0]
    fp.close()
    return class_to_name_dict





# compute training data statistics
def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
    mean /= total_images_count
    std /= total_images_count
    return mean, std
    
# Set recompute_stats to True to recompute the statistics
def get_training_stats(recompute_stats=False,train_dir=TRAIN_DIR):
    if recompute_stats:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=T.Compose([T.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        trn_mean, trn_std = calculate_mean_std(train_loader)
    else:   # use precomputed statistics of TinyImageNet training data
        trn_mean = [0.4802, 0.4481, 0.3975]
        trn_std = [0.2296, 0.2263, 0.2255]
    return trn_mean, trn_std


class TinyImageNetDataModule(L.LightningModule):
    def __init__(self,config,transform=None):
        super().__init__()
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.val_split = config['val_split']
        self.transform = transform


    def prepare_data(self):
        '''
        - download the dataset
        - extract the dataset
        - create a validation folder
        '''
        val_img_dir = os.path.join(self.data_dir,'val','images')
        prepare_tinyimagenet_validation_folder(val_img_dir)

    def setup(self, stage=None):
        '''
        - create train, val, and test datasets
        - compute stats
        - apply transforms
        '''
        # Read image files to pytorch dataset using ImageFolder, a generic data 
        # loader where images are in format root/label/filename
        # See https://pytorch.org/vision/stable/datasets.html
        self.train_ds = datasets.ImageFolder(os.path.join(self.data_dir,'train'), transform=self.transform)
        self.val_ds = datasets.ImageFolder(os.path.join(self.data_dir,'val','images'), transform=self.transform)
        self.test_ds = datasets.ImageFolder(os.path.join(self.data_dir,'val','images'), transform=self.transform)

    def train_dataloader(self):
        '''
        generate the train dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 11}
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, **kwargs)
    
    def val_dataloader(self):
        '''
        generate the validation dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 11}
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, **kwargs)
    
    def test_dataloader(self):
        '''
        generate the test dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 11}
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, **kwargs)
    

    
############################################
# creating the model and LigningModule
def compute_layer_dimensions(input_size, stride):
    """Compute output spatial dimensions after a layer with given stride"""
    return (input_size + 1) // stride  # +1 to account for potential padding


def create_efficientnet_model(model_name, model_hparams, load_pretrained=False):
    # Extract parameters needed for dimension calculation
    # batch_size = model_hparams.pop('batch_size', 64)  # Default to 64 if not provided
    # image_size = model_hparams.pop('image_size', 224)  # Default to 224 if not provided
    
    # # Create a dict to store expected dimensions at each layer
    # dims = {'input': (batch_size, 3, image_size, image_size)}
    
    # # Initial stem conv layer dimensions
    # stem_stride = 2  # EfficientNet's stem uses stride 2
    # stem_h = compute_layer_dimensions(image_size, stem_stride)
    # dims['stem'] = (batch_size, 32, stem_h, stem_h)  # 32 is standard EfficientNet stem output channels
    
    # Create model with appropriate normalization layers based on dimensions
    if load_pretrained:
        print(f"Loading pretrained model {model_name} from torch hub")
        model = EfficientNetBW.from_pretrained(model_name, **model_hparams)
    else:
        model = EfficientNetBW.from_name(model_name, **model_hparams)
    return model


class TinyImageNetModule(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams,lr_scheduler_name, lr_scheduler_hparams,data_hparams):
        """TinyImageNetModule.

        Args:
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
            lr_scheduler_hparams: Hyperparameters for the learning rate scheduler, as dictionary. This includes step size, gamma, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.lr_scheduler_name = lr_scheduler_name        
        self.load_pretrained = model_hparams.pop('load_pretrained', False)
        
        # Add batch_size and image_size to model_hparams from data_hparams
        model_hparams['batch_size'] = data_hparams['batch_size']
        # model_hparams['image_size'] = data_hparams['image_size']      # not sure if we should enforce the image size here or scale the data according to the model 
        
        # Create model
        self.model = create_efficientnet_model(self.model_name, model_hparams, self.load_pretrained)
        
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, data_hparams['image_size'], data_hparams['image_size']), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        opt_name = self.optimizer_name
        if opt_name == "Adam":
            optimizer = optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        elif opt_name == "AdamW":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif opt_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        elif opt_name == "RMSprop":
            optimizer = optim.RMSprop(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: {opt_name}"

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        scheduler_name = self.lr_scheduler_name
        if scheduler_name is not None:
            if scheduler_name == "StepLR":
                scheduler = optim.lr_scheduler.StepLR(optimizer, **self.hparams.lr_scheduler_hparams)
            elif scheduler_name == "MultiStepLR":
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.hparams.lr_scheduler_hparams)
            elif scheduler_name == "CosineAnnealingWarmRestarts":
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.hparams.lr_scheduler_hparams)

            return [optimizer], [scheduler]
        else:
            return optimizer


    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        idx=0
        for name, module in self.model.named_modules():
            if hasattr(module, 'cov_cond_list'):
                idx+=1
                self.log(f"zl{idx:02}_cond_{name}",module.cov_cond_list[-1])

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (labels == preds.argmax(dim=-1)).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc,prog_bar=True,on_epoch=True,sync_dist=True)
        self.log("val_loss", loss,on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc,sync_dist=True)


############################################
# Training the model
def avg_corr_hook_fn(module, input, output):
    # Take the first element of the input tuple
    input_tensor = input[0]
    
    # Compute average cross-correlation
    avg_corr = comp_avg_corr(input_tensor)
    # Store the result
    if not hasattr(module, 'avg_corr_list'):
        module.avg_corr_list = []
    module.avg_corr_list.append(avg_corr.item())
    # module.avg_corr_list = [avg_corr.item()]


def cond_hook_fn(module, input, output):
    # Take the first element of the input tuple
    input_tensor = input[0]

    # compute condition number
    cov_cond = comp_cov_cond(input_tensor)
    # Store the result
    if not hasattr(module, 'cov_cond_list'):
        module.cov_cond_list = []
    # module.cov_cond_list.append(cov_cond.item())
    module.cov_cond_list = [cov_cond.item()]


def register_hooks(model):
    for name,module in model.named_modules():
        if isinstance(module, BatchWhiteningBlock) or isinstance(module, nn.BatchNorm2d):
            print(f'registering hook to {name}')
            module.register_forward_hook(cond_hook_fn)


def save_cov_stats(model):
    all_correlations = []
    # corr_stats=[]
    cond_stats=[]
    for name, module in model.named_modules():
        if isinstance(module, BatchWhiteningBlock):
            pass # TODO

    return


    




class CustomWarmUpCallback(L.Callback):
    def __init__(self, warmup_steps):
        super().__init__()
        self.warmup_steps = warmup_steps

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        current_step = trainer.global_step
        if current_step <= self.warmup_steps:
            # Adjust the behavior of the target layers during warm-up
            pl_module.model.set_bw_cov_warmup(True)
        else:
            pl_module.model.set_bw_cov_warmup(False)


def create_trainer(config,callbacks=None):
    # save_name = config['model']['name'] +'_'+ config['wandb']['name']
    save_name = config['model']['name']
    logger = None
    if config.get("wandb",None) is not None:
        logger = WandbLogger(**config["wandb"])
    # else:
    #     logger = TensorBoardLogger("lightning_logs", name=save_name)
    if callbacks is None:
        callbacks= [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                    LearningRateMonitor("epoch"),       # Log learning rate every epoch
                    CustomWarmUpCallback(5000)
                ]  
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models and tensorboard logs
        accelerator="auto",
        logger = logger,
        callbacks=callbacks,
        **config['trainer']
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    return trainer 



def create_model(config):
    model_name = config['model'].pop('name')
    optimizer_name = config['optimizer'].pop('opt_name')
    lr_scheduler_name = config['lr_scheduler'].pop('sched_name')
    # save_name = model_name +'_'+ config['wandb']['name']
    save_name = model_name
    if config['train']['eval_only'] and config['train']['ckpt'] is not None and os.path.isfile(os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")):
        print(f"Found pretrained model at {config['train']['ckpt']}, loading...")
        model = TinyImageNetModule.load_from_checkpoint(config['train']['ckpt'])
    else:
        model = TinyImageNetModule(model_name,config['model'],optimizer_name,config['optimizer'],lr_scheduler_name,config['lr_scheduler'],config['dataset'])
    # register_hooks(model)
    return model


def create_data_module(config):
    # check whether to use imagenet stats or TinyImageNet stats - depending on whether we use a pretrained model or not    
    if config['model'].get('load_pretrained',False):
        trn_mean = [0.485, 0.456, 0.406]
        trn_std = [0.229, 0.224, 0.225]
    else:
        trn_mean, trn_std = get_training_stats(recompute_stats=config['dataset']['recompute_stats'])
    # preprocess_transform = T.Compose([
    #             T.Resize(256), # Resize images to 256 x 256
    #             T.CenterCrop(config['dataset']['image_size']), # Center crop image
    #             T.RandomHorizontalFlip(),
    #             # T.TenCrop(config['dataset']['image_size']),
    #             T.ToTensor(),  # Converting cropped images to tensors
    #             T.Normalize(mean=trn_mean, std=trn_std)
    # ])

    preprocess_transform = T.Compose([
            T.Resize(256),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(degrees=45),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            T.CenterCrop(config['dataset']['image_size']),
            T.ToTensor(),
            T.Normalize(mean=trn_mean, std=trn_std)
    ])


    dataset_module = TinyImageNetDataModule(config['dataset'],transform=preprocess_transform)
    return dataset_module

        
def analyze_model_dimensions(model, input_size=(1, 3, 224, 224)):
    """Analyze and print the dimensions of each layer in the model.
    
    Args:
        model: The PyTorch model to analyze
        input_size: The input tensor size (batch_size, channels, height, width)
    """
    print("\nModel Layer Dimensions Analysis:")
    print("=" * 80)
    print(f"Input tensor shape: {input_size}")
    print("-" * 80)
    
    # Create a dummy input tensor
    x = torch.randn(input_size)
    
    # Register hooks to capture output shapes
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # Handle cases where output is a tuple
        print(f"{module.__class__.__name__:30} Output shape: {output.shape}")
    
    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if len(name) > 0:  # Skip the model itself
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
    # Forward pass with dummy input
    try:
        model(x)
    except Exception as e:
        print(f"Error during forward pass: {e}")
    finally:
        # Remove all hooks
        for hook in hooks:
            hook.remove()
    
    print("=" * 80)


def main(config):
    L.seed_everything(config['global_seed'])
    # if config['wandb']['mode'] == 'online':
    #     wandb.login()

    data_set = create_data_module(config)
    trainer = create_trainer(config)

    if config['lr_scheduler']['sched_name']=='CosineAnnealingWarmRestarts':   # need to compute T_0  
        n_epochs = config['trainer']['max_epochs']
        n_cycles = config['lr_scheduler'].pop('n_cycles',1)
        config['lr_scheduler']['T_0'] = n_epochs//n_cycles
        

    model = create_model(config)
    
    # Print model summary using torchsummary
    # print("\nModel Summary:")
    # print("=" * 80)
    # summary(model.model, input_size=(config['dataset']['batch_size'], 3, config['dataset']['image_size'], config['dataset']['image_size']))
    # print("=" * 80)
    
    # Analyze and print model dimensions
    # batch_size = config['dataset']['batch_size']
    # image_size = config['dataset']['image_size']
    # input_size = (batch_size, 3, image_size, image_size)
    # analyze_model_dimensions(model.model, input_size)  # Note: we use model.model because TinyImageNetModule wraps the actual model
    
    # model = torch.compile(model)

    trainer.fit(model, datamodule=data_set)
    model = TinyImageNetModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training

    if config['train']['test']:
        val_result = trainer.test(model, datamodule=data_set)
        result = {"val": val_result[0]["test_acc"]}

    print(f"Test accuracy: {result['val']:.3f}")
    return model, result



config_defaults = {'global_seed':42,
                'wandb':{'mode':'online',
                            'project':'bw-efcnt_tin',
                            'name':'bw_exp_b0',},
                    'dataset':{ 'data_dir':DATA_DIR,
                                'batch_size':64,
                                'image_size':224,
                                'recompute_stats':False,
                                'val_split':0.2},
                    'model':{ 'name':'efficientnet-b0', 'load_pretrained':False, 'num_classes':200,'dropout_rate':0.5,'conv_stem_type':1,'mbconv_type':1},
                    'optimizer':{ 'opt_name':'AdamW', 'lr':0.001, 'weight_decay':0.001},
                    'lr_scheduler':{ 'sched_name':'StepLR', 'step_size':5, 'gamma':0.98},
                    'trainer':{ 'max_epochs':300, 'devices':'auto','strategy':'auto','precision':32},
                    'train':{ 'ckpt':None, 'test':True, 'eval_only':False},
                    }



def set_trial_params(config,trial):
    config['optimizer']['lr'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, step=5e-5)
    config['optimizer']['weight_decay'] = trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    config['dataset']['batch_size'] = trial.suggest_categorical('batch_size', [32,64,128,256,512])
    config['model']['dropout_rate'] = trial.suggest_float('dropout', 0.2,0.8,step=0.1)
    config['lr_scheduler']['step_size'] = trial.suggest_int('lr_sched_step_size',5,105,step=10)
    config['lr_scheduler']['gamma'] = trial.suggest_categorical('lr_sched_gamma',[0.9,0.95,0.99])

    config['model']['conv_stem_type'] = trial.suggest_categorical('conv_stem_type',[1,2])
    config['model']['mbconv_type'] = trial.suggest_categorical('mbconv_type',[1,2,3])

    config['model']['batch_whitening_momentum'] = trial.suggest_float('batch_whitening_momentum',0.9,0.99,step=0.01)
    config['model']['batch_whitening_epsilon'] = trial.suggest_float('batch_whitening_epsilon', 1e-5, 1e-3, step=5e-5)
    return config

def objective(trial):

    config = copy.deepcopy(config_defaults)
    config.pop('wandb') 
    # Suggest values for hyperparameters
    config = set_trial_params(config,trial)
    print('='*50,'TRIAL START','='*50)
    print(config)
    print('='*101)

    data_set = create_data_module(config)
    
    config['trainer']['max_epochs'] = 50
    config['trainer']['enable_checkpointing']=False
    callbacks= [LearningRateMonitor("epoch"),       # Log learning rate every epoch
                CustomWarmUpCallback(5000)]  

    trainer = create_trainer(config,callbacks)

    if config['lr_scheduler']['sched_name']=='CosineAnnealingWarmRestarts':   # need to compute T_0  
        n_epochs = config['trainer']['max_epochs']
        n_cycles = config['lr_scheduler'].pop('n_cycles',1)
        config['lr_scheduler']['T_0'] = n_epochs//n_cycles
        

    model = create_model(config)
    # print the summary of the model
    print(model)

    trainer.fit(model, datamodule=data_set)

    # Return the final validation loss (or other metric)
    return trainer.callback_metrics['train_loss'].item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network with specified GPU")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use for training (default: 0)")
    args = parser.parse_args()

    # change config if needed
    
    # config_defaults['trainer']['strategy'] = 'ddp_find_unused_parameters_true'       # enable on multi gpu machine
    L.seed_everything(config_defaults['global_seed'])
    if HPARAM_OPT=='TRAIN':
        study_filename='study.pkl'
        print('='*20,f'HPARAM OPT TRAIN on {study_filename}','='*20)
        if os.path.exists(study_filename):
            print('continuing previous study')
            with open(study_filename, 'rb') as file:
                study=pickle.load(file)
        else:
            print('starting new study')
            study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        with open(study_filename, 'wb') as file:
            pickle.dump(study, file)        
        # Print the best hyperparameters found
        print("Best hyperparameters: ", study.best_trial.params)

    elif HPARAM_OPT=='INFER':
        study_filename='study.pkl'
        print('='*20,f'HPARAM OPT INFER on {study_filename}','='*20)
        if os.path.exists(study_filename):
            print('loading study')
            with open(study_filename, 'rb') as file:
                study=pickle.load(file)
        else:
            raise ValueError(f'couldnt find {study_filename}')
        print(f'best trial: {study.best_trial.number}')
        print(f'best params: {study.best_params}')
        config = copy.deepcopy(config_defaults)
        # config.pop('wandb') 
        # config['wandb']['name'] = 'nbw2_exp_b0_best_modified'
        config['wandb']['name'] = 'nbw2_exp_b0_dbw_blkdiag_best'

        # set best params 
        config['optimizer']['lr'] = study.best_params['learning_rate']
        config['optimizer']['weight_decay'] = study.best_params['weight_decay']
        # config['dataset']['batch_size'] = study.best_params['batch_size']
        config['dataset']['batch_size'] = 32    # override due to performance issues (batch whitening gets extremely slow on larger batch size)
        config['model']['dropout_rate'] = study.best_params['dropout']
        config['lr_scheduler']['step_size'] = study.best_params['lr_sched_step_size']
        config['lr_scheduler']['gamma'] = study.best_params['lr_sched_gamma']

        config['model']['conv_stem_type'] = study.best_params['conv_stem_type']
        # config['model']['conv_stem_type'] = 1       # temp experiment to disable bw on all layers except for MBConv blocks
        config['model']['mbconv_type'] = study.best_params['mbconv_type']
        # config['model']['mbconv_type']=0        # for debug - disable batch whitening. use bn .

        config['model']['batch_whitening_momentum'] = study.best_params['batch_whitening_momentum']
        config['model']['batch_whitening_epsilon'] = study.best_params['batch_whitening_epsilon']

        config['trainer']['max_epochs'] = 50
        if args.gpu>=0:
            config['trainer']['devices'] = [args.gpu]
        config['trainer']['accumulate_grad_batches'] = 4
        print(config)
        # run the training
        main(config)


    else:
        # Run the main function
        # delete 'wandb' from config    
        config = copy.deepcopy(config_defaults)
        config.pop('wandb') 
        # config['model']['name'] = 'efficientnet-b3'
        # config['model']['batch_whitening_momentum'] = 0.1   # higher value for faster update of running_mean (more weight on curent batch statistics)
        config['optimizer']['opt_name'] = 'AdamW'
        config['optimizer']['lr'] = 0.001
        # config['lr_scheduler']={'sched_name':'CosineAnnealingWarmRestarts', 'n_cycles':5, 'eta_min':0.1*config['optimizer']['lr']}
        # config['model']['mbconv_type']=0      # 0 to turn batch whitening off
        # config['model']['conv_stem_type']=1
        config['model']['mbconv_type']=2      
        config['model']['conv_stem_type']=1

        config['trainer']['max_epochs'] = 50
        if args.gpu>=0:
            config['trainer']['devices'] = [args.gpu]
        config['dataset']['batch_size'] = 32
        # config['trainer']['precision'] = 16
        config['trainer']['accumulate_grad_batches'] = 4
        # config['wandb']['name'] = f"nbw2_exp_b0_Itn_fix_off"
        print(config)
        main(config)

