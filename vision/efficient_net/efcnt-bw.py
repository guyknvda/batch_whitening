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
import ast

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from efficientnet_pytorch import EfficientNetBW,BatchWhiteningBlock,comp_avg_corr,get_rank,comp_cov_cond
import wandb
import optuna




# set directories

DATA_DIRS = {
    'tin': '/datasets/vision/tiny-imagenet-200',        # to download the dataset : wget http://cs231n.stanford.edu/tiny-imagenet-200.zip Original images come in shapes of [3,64,64]
    'imgnet': '/datasets/vision/imagenet/ILSVRC/Data/CLS-LOC/' # to download the dataset : wget http://www.image-net.org/download-images.php
}
# Define training and validation data paths
CHECKPOINT_PATH = "./checkpoints"
# HPARAM_OPT is deprecated. Use --mode CLI argument instead.
 

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
        val_annotations_path = os.path.join(os.path.dirname(val_img_dir), 'val_annotations.txt')
        val_data = pd.read_csv(val_annotations_path, 
                            sep='\t', 
                            header=None, 
                            names=['File', 'Class', 'X', 'Y', 'H', 'W'])



        # Open and read val annotations text file
        fp = open(val_annotations_path, 'r')
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
def get_training_stats(data_dir,recompute_stats=False):
    if recompute_stats:
        train_dir = os.path.join(data_dir,'train')
        train_dataset = datasets.ImageFolder(root=train_dir, transform=T.Compose([T.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        trn_mean, trn_std = calculate_mean_std(train_loader)
    elif data_dir == DATA_DIRS['tin']:
        # use precomputed statistics of TinyImageNet training data
        trn_mean = [0.4802, 0.4481, 0.3975]
        trn_std = [0.2296, 0.2263, 0.2255]
    elif data_dir == DATA_DIRS['imgnet']:
        # use precomputed statistics of ImageNet training data
        trn_mean = [0.485, 0.456, 0.406]
        trn_std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unknown dataset: {data_dir}")
    return trn_mean, trn_std


class TinyImageNetDataModule(L.LightningDataModule):
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
        kwargs = {"pin_memory": True, "num_workers": 24}
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, **kwargs)
    
    def val_dataloader(self):
        '''
        generate the validation dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 24}
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, **kwargs)
    
    def test_dataloader(self):
        '''
        generate the test dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 24}
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, **kwargs)
    
class ImageNetDataModule(L.LightningDataModule):
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
        return
    
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
        self.val_ds = datasets.ImageFolder(os.path.join(self.data_dir,'val'), transform=self.transform)
        self.test_ds = datasets.ImageFolder(os.path.join(self.data_dir,'val'), transform=self.transform)

    def train_dataloader(self):
        '''
        generate the train dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 24}
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, **kwargs)
    
    def val_dataloader(self):
        '''
        generate the validation dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 24}
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, **kwargs)
    
    def test_dataloader(self):
        '''
        generate the test dataloader
        '''
        kwargs = {"pin_memory": True, "num_workers": 24}
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


class ImgClsModel(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams,lr_scheduler_name, lr_scheduler_hparams,data_hparams):
        """ImgClsModel.

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

def get_wandb_logger(project, name):
    # name can be either run ID or run name. if there's a run with this ID, resume it, otherwise start a new run with this name
    try:
        # Check if name is a valid run ID 
        api = wandb.Api()
        run = api.run(f"{project}/{name}")
        
        # Resume existing run
        return WandbLogger(
            project=project,
            id=name,
            resume="must"  # Force resuming
        )
    except Exception:
        # If run doesn't exist, start new run with name as run name
        return WandbLogger(
            project=project,
            name=name,
            # log_model="all"  
        )

def create_trainer(config,callbacks=None):
    # save_name = config['model']['name'] +'_'+ config['wandb']['name']
    save_name = config['model']['name'] + '_' + config['dataset']['name']
    logger = None
    if config.get("wandb",None) is not None:
        logger = get_wandb_logger(config["wandb"]["project"],config["wandb"]["name"])
    # else:
    #     logger = TensorBoardLogger("lightning_logs", name=save_name)
    
    # Handle checkpoint directory - keep it consistent for proper resuming
    checkpoint_dir = os.path.join(CHECKPOINT_PATH, save_name)
    
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=save_name+'-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,  # Save the top 3 models
        mode='min',
        save_last=True  # Save the last checkpoint for resuming training
    )
    
    
    if callbacks is None:
        callbacks= [checkpoint_callback,  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                    LearningRateMonitor("epoch"),       # Log learning rate every epoch
                    CustomWarmUpCallback(5000)
                ]  
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        # default_root_dir=checkpoint_dir,  # Where to save models and tensorboard logs
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
    
    # Check if we should load from checkpoint
    if config['train']['ckpt'] is not None and os.path.isfile(config['train']['ckpt']):
        print(f"Loading model from checkpoint: {config['train']['ckpt']}")
        model = ImgClsModel.load_from_checkpoint(config['train']['ckpt'])
    else:
        model = ImgClsModel(model_name,config['model'],optimizer_name,config['optimizer'],lr_scheduler_name,config['lr_scheduler'],config['dataset'])
    # register_hooks(model)
    return model


def create_data_module(config):
    # check whether to use imagenet stats or TinyImageNet stats - depending on whether we use a pretrained model or not    
    if config['dataset']['name'] in DATA_DIRS:
        config['dataset']['data_dir'] = DATA_DIRS[config['dataset']['name']]
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']['name']}")

    if config['model'].get('load_pretrained',False):
        trn_mean = [0.485, 0.456, 0.406]
        trn_std = [0.229, 0.224, 0.225]
    else:
        trn_mean, trn_std = get_training_stats(data_dir=config['dataset']['data_dir'],
                                               recompute_stats=config['dataset']['recompute_stats'])

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

    if config['dataset']['name'] == 'tin':
        dataset_module = TinyImageNetDataModule(config['dataset'],transform=preprocess_transform)
    elif config['dataset']['name'] == 'imgnet':
        dataset_module = ImageNetDataModule(config['dataset'],transform=preprocess_transform)
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']['name']}")
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
    print(model)
    if config['train']['ckpt'] is not None:
        trainer.fit(model, datamodule=data_set, ckpt_path=config['train']['ckpt'])
    else:
        trainer.fit(model, datamodule=data_set)
    


    # Find the checkpoint callback and get the best model path
    checkpoint_callback = None
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            checkpoint_callback = callback
            break
    
    if checkpoint_callback:
        print(f"Training completed. Best checkpoint: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_path:
            model = ImgClsModel.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )  # Load best checkpoint after training
    else:
        print("Training completed. No checkpoint callback found.")

    if config['train']['test']:
        test_result = trainer.test(model, datamodule=data_set)
        result = {"test": test_result[0]["test_acc"]}

    print(f"Test accuracy: {result['test']:.3f}")
    if (trainer.logger is not None and 
        isinstance(trainer.logger, WandbLogger) and 
        hasattr(trainer.logger, 'experiment') and 
        trainer.logger.experiment is not None and
        hasattr(trainer.logger.experiment, 'id')):
        print(f"wandb run ID: {trainer.logger.experiment.id}")
    else:
        print("No wandb logger found")

    return model, result
    # return model, result, trainer.logger.experiment.id if trainer.logger is not None and hasattr(trainer.logger, 'experiment') and hasattr(trainer.logger.experiment, 'id') else None, trainer.checkpoint_callback.best_model_path



config_defaults = {'global_seed':42,
                'wandb':{'project':'bw-efcnt',
                            'name':'bw_exp_b0'},
                    'dataset':{ 'name':'tin', 'data_dir':DATA_DIRS['tin'],  # Will be set based on dataset choice
                                'batch_size':32,
                                'image_size':224,
                                'recompute_stats':False,
                                'val_split':0.2},
                    'model':{ 'name':'efficientnet-b0', 'load_pretrained':False, 'num_classes':200,'dropout_rate':0.5,'conv_stem_type':1,'mbconv_type':1},
                    'optimizer':{ 'opt_name':'AdamW', 'lr':0.001, 'weight_decay':0.001},
                    'lr_scheduler':{ 'sched_name':None},  # Default to no scheduler
                    'trainer':{ 'max_epochs':300, 'devices':'auto','strategy':'auto','precision':32},
                    'train':{ 'ckpt':None, 'test':True, 'eval_only':False},
                    }

# Dataset-specific configuration overrides
dataset_configs = {
    'tin': {
        'wandb': {'project': 'bw-efcnt_tin'},
        'model': {'num_classes': 200},
        # 'dataset': {'image_size': 64},  # TinyImageNet original size
        'dataset': {'image_size': 224},  
        'trainer': {'max_epochs': 50},
        'optimizer': {'lr': 0.001, 'weight_decay': 0.001},
        'lr_scheduler':{ 'sched_name':'StepLR', 'step_size':5, 'gamma':0.98},
    },
    'imgnet': {
        'wandb': {'project': 'efcnt_imgnet'},
        'model': {'num_classes': 1000, 'load_pretrained': False},
        'dataset': {'image_size': 224},  # ImageNet standard size
        'trainer': {'max_epochs': 100},  # ImageNet typically needs fewer epochs
        'optimizer': {'lr': 0.002, 'weight_decay': 0.001},  # Lower LR for pretrained
        # 'lr_scheduler':{'sched_name':'CosineAnnealingWarmRestarts', 'n_cycles':10, 'eta_min':0.000001},
        'lr_scheduler':{'sched_name':'CosineAnnealingWarmRestarts', 'n_cycles':1, 'eta_min':0.000001},  # for debugging
    }
}



def set_trial_params(config,trial):
    config['optimizer']['lr'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, step=5e-5)
    config['optimizer']['weight_decay'] = trial.suggest_categorical('weight_decay', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
    # config['dataset']['batch_size'] = trial.suggest_categorical('batch_size', [32,64,128,256,512])
    config['model']['dropout_rate'] = trial.suggest_float('dropout', 0.2,0.8,step=0.1)
    config['lr_scheduler']['step_size'] = trial.suggest_int('lr_sched_step_size',5,105,step=10)
    config['lr_scheduler']['gamma'] = trial.suggest_categorical('lr_sched_gamma',[0.9,0.95,0.99])

    config['model']['conv_stem_type'] = trial.suggest_categorical('conv_stem_type',[1,2])
    config['model']['mbconv_type'] = trial.suggest_categorical('mbconv_type',[1,2,3])

    config['model']['batch_whitening_momentum'] = trial.suggest_float('batch_whitening_momentum',0.9,0.99,step=0.01)
    config['model']['batch_whitening_epsilon'] = trial.suggest_float('batch_whitening_epsilon', 1e-5, 1e-3, step=5e-5)
    config['model']['bw_fix_factor'] = trial.suggest_float('bw_fix_factor',0.9,0.99,step=0.01)
    config['model']['bw_cov_err_threshold']=trial.suggest_categorical('bw_cov_err_threshold',[0.01,0.05,0.1,0.2,0.3])
    return config

def objective(trial):
    # Open the log file in append mode ('a+')
    log_file = 'optimization_trials.log'
    with open(log_file, 'a+') as f:
        # Write trial separator
        trial_header = f'\n{"="*50} TRIAL {trial.number} START {"="*50}\n'
        f.write(trial_header)
        print(trial_header)  # Also print to console

        # Note: This function is called from within the TRAIN mode, 
        # so it should use the same configuration that was set up there.
        # We'll need to pass the base config to this function, but for now
        # we'll reconstruct it here to maintain compatibility.
        
        # Get the dataset from the global context (set in main)
        # This is a bit of a hack, but maintains backward compatibility
        import sys
        if hasattr(sys.modules[__name__], '_current_dataset'):
            dataset = sys.modules[__name__]._current_dataset
        else:
            dataset = 'tin'  # default fallback
            
        # Reconstruct the config hierarchy
        trial_config = copy.deepcopy(config_defaults)
        if dataset in dataset_configs:
            trial_config = merge_configs(trial_config, dataset_configs[dataset])
        
        trial_config.pop('wandb', None) 
        # Suggest values for hyperparameters
        trial_config = set_trial_params(trial_config, trial)
        
        # Write config to file
        config_str = f'Configuration:\n{trial_config}\n{"="*101}\n'
        f.write(config_str)
        print(config_str)  # Also print to console

        data_set = create_data_module(trial_config)
        
        trial_config['trainer']['max_epochs'] = 30
        trial_config['trainer']['enable_checkpointing']=False
        callbacks= [LearningRateMonitor("epoch"),       # Log learning rate every epoch
                    CustomWarmUpCallback(5000)]  

        trainer = create_trainer(trial_config,callbacks)

        if trial_config['lr_scheduler']['sched_name']=='CosineAnnealingWarmRestarts':   # need to compute T_0  
            n_epochs = trial_config['trainer']['max_epochs']
            n_cycles = trial_config['lr_scheduler'].pop('n_cycles',1)
            trial_config['lr_scheduler']['T_0'] = n_epochs//n_cycles
            
        model = create_model(trial_config)
        
        # Redirect model summary to string
        import io
        from contextlib import redirect_stdout
        
        # Capture model summary
        summary_buffer = io.StringIO()
        with redirect_stdout(summary_buffer):
            print(model)
        model_summary = summary_buffer.getvalue()
        
        # Write model summary to file
        f.write(f'\nModel Summary:\n{model_summary}\n')
        print(f'\nModel Summary:\n{model_summary}')  # Also print to console

        trainer.fit(model, datamodule=data_set)

        # Write final metrics
        final_loss = trainer.callback_metrics['train_loss'].item()
        result_str = f'\nFinal training loss: {final_loss}\n'
        f.write(result_str)
        print(result_str)  # Also print to console

        # Write trial end separator
        trial_footer = f'{"="*50} TRIAL {trial.number} END {"="*50}\n'
        f.write(trial_footer)
        print(trial_footer)  # Also print to console

        return final_loss

# ------------------------------------------------------------
# Utility to override nested configuration entries from CLI
# ------------------------------------------------------------

def merge_configs(base_config, override_config):
    """Recursively merge override_config into base_config.
    
    This creates a deep copy of base_config and applies overrides from override_config.
    Nested dictionaries are merged recursively.
    """
    import copy
    merged = copy.deepcopy(base_config)
    
    def _merge_dict(base_dict, override_dict):
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                _merge_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    _merge_dict(merged, override_config)
    return merged

def apply_overrides(config, overrides):
    """Apply KEY=VALUE overrides to a (potentially nested) config dict.

    Keys can use dot-notation to address nested fields, e.g. "optimizer.lr".
    Values are parsed with ast.literal_eval when possible, falling back to
    strings if parsing fails.
    """
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        path, value_str = item.split('=', 1)
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str  # treat as plain string
        keys = path.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

if __name__ == "__main__":
    # -----------------------------
    # Command-line interface setup
    # -----------------------------
    parser = argparse.ArgumentParser(description="EfficientNet-BW training/inference script")
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # TRAIN subcommand
    train_parser = subparsers.add_parser('TRAIN', help='Run Optuna hyper-parameter sweep')
    train_parser.add_argument('pkl_file', type=str, help='Path to Optuna study pickle file')
    train_parser.add_argument('--dataset', choices=['tin', 'imgnet'], default='tin')
    train_parser.add_argument('--gpu', type=int, default=0)
    train_parser.add_argument('--wandb', nargs='?', const='', default=None, metavar='RUN_NAME',
                              help='Enable WandB logging; optionally set run name')
    train_parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials')
    train_parser.add_argument('-o', '--override', nargs='*', default=[], metavar='KEY=VALUE',
                              help='Override any default config parameter (dot-notation)')

    # INFER subcommand

    # python efcnt-bw.py INFER study.pkl -o trainer.max_epochs=2 --wandb bw_dbg2 
    # resume from checkpoint
    # python efcnt-bw.py INFER study.pkl -o trainer.max_epochs=4 --ckpt_path checkpoints/efficientnet-b0_tin//last.ckpt --wandb <run ID>
    #  imgnet
    # python efcnt-bw.py INFER dummy.pkl --dataset imgnet -o model.mbconv_type=0 --wandb bw_imgnet_off -o trainer.max_epochs=2
    # python efcnt-bw.py INFER dummy.pkl --dataset imgnet -o model.mbconv_type=0 --ckpt_path checkpoints/efficientnet-b0_imgnet/last.ckpt --wandb <run ID>
    infer_parser = subparsers.add_parser('INFER', help='Train/evaluate model with chosen trial parameters')
    infer_parser.add_argument('pkl_file', type=str, help='Path to Optuna study pickle file')
    infer_parser.add_argument('optuna_trial_id', nargs='?', type=int,
                              help='Optuna trial ID to use (defaults to best)')
    infer_parser.add_argument('--dataset', choices=['tin', 'imgnet'], default='tin')
    infer_parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint file')
    infer_parser.add_argument('--wandb', nargs='?', const='', default=None, metavar='RUN_NAME')
    infer_parser.add_argument('--gpu', type=int, default=0)
    infer_parser.add_argument('-o', '--override', nargs='*', default=[], metavar='KEY=VALUE')

    args = parser.parse_args()

    # -----------------------------
    # Configuration hierarchy setup
    # -----------------------------

    # Step 1: Start with base defaults
    config = copy.deepcopy(config_defaults)
    
    # Step 2: Apply dataset-specific overrides

    if args.dataset in dataset_configs:
        config = merge_configs(config, dataset_configs[args.dataset])
    config['dataset']['name'] = args.dataset
    
    # Step 3: Set device
    config['trainer']['devices'] = [args.gpu]


    # Step 5: Apply command-line overrides (highest priority)
    if args.override:
        apply_overrides(config, args.override)

    L.seed_everything(config['global_seed'])

    # -----------------------------
    # TRAIN mode
    # -----------------------------
    if args.mode == 'TRAIN':
        # Set global variable for objective function to access
        import sys
        setattr(sys.modules[__name__], '_current_dataset', args.dataset)
        
        study_filename = args.pkl_file
        print('='*20, f'HPARAM OPT TRAIN on {study_filename}', '='*20)
        if os.path.exists(study_filename):
            with open(study_filename, 'rb') as f:
                study = pickle.load(f)
            print('Continuing previous study')
        else:
            study = optuna.create_study(direction='minimize')
            print('Starting new study')

        study.optimize(objective, n_trials=args.n_trials)

        with open(study_filename, 'wb') as f:
            pickle.dump(study, f)

        print('Best hyperparameters:', study.best_trial.params)

    # -----------------------------
    # INFER mode
    # -----------------------------
    elif args.mode == 'INFER':
        study_filename = args.pkl_file
        cfg = copy.deepcopy(config)
        cfg['trainer']['accumulate_grad_batches'] = 4   
        print('='*20, f'HPARAM OPT INFER on {study_filename}', '='*20)
        if os.path.exists(study_filename):
            with open(study_filename, 'rb') as f:
                study = pickle.load(f)
            trial_ids = {t.number for t in study.trials}
            selected_id = args.optuna_trial_id if args.optuna_trial_id in trial_ids else study.best_trial.number
            print(f'Selected trial: {selected_id}')

            # Start with the already configured config (base + dataset-specific + CLI overrides)
            
            params = study.trials[selected_id].params

            # Apply trial-specific parameters
            mapping = {
                ('optimizer', 'lr'): 'learning_rate',
                ('optimizer', 'weight_decay'): 'weight_decay',
                ('model', 'dropout_rate'): 'dropout',
                ('lr_scheduler', 'step_size'): 'lr_sched_step_size',
                ('lr_scheduler', 'gamma'): 'lr_sched_gamma',
                ('model', 'conv_stem_type'): 'conv_stem_type',
                ('model', 'mbconv_type'): 'mbconv_type',
                ('model', 'batch_whitening_momentum'): 'batch_whitening_momentum',
                ('model', 'batch_whitening_epsilon'): 'batch_whitening_epsilon',
                ('model', 'bw_fix_factor'): 'bw_fix_factor',
                ('model', 'bw_cov_err_threshold'): 'bw_cov_err_threshold',
            }
            for (section, key), trial_key in mapping.items():
                if trial_key in params:
                    cfg[section][key] = params[trial_key]
        
        
        
        else:
            print(f'Study file {study_filename} not found. Using default configuration.')
        

        if args.ckpt_path:
            cfg['train']['ckpt'] = args.ckpt_path
            
        else:
            cfg['train']['ckpt'] = None

        if args.wandb is not None:  
            cfg['wandb']['name'] = args.wandb
        else:
            cfg.pop('wandb', None)
        
        main(cfg)

