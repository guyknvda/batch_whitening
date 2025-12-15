# Import dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import models, datasets
from torchvision import transforms as T

import os
from random import randint
import urllib
import zipfile

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from efficientnet_pytorch import EfficientNet
import wandb


# to download the dataset : wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# set directories
DATA_DIR = '/datasets/vision/imagenet/ILSVRC/Data/CLS-LOC' # Original images come in shapes of [3,224,224]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')
CHECKPOINT_PATH = "saved_models"


############################################
# ImageNet dataset module functions
# Unlike training folder where images are already arranged in sub folders based 
# on their labels, images in validation folder are all inside a single folder. 
# Validation folder comes with images folder and val_annotations txt file. 

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file


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
    else:   # use precomputed statistics of ImageNet training data
        trn_mean = [0.485, 0.456, 0.406]
        trn_std = [0.229, 0.224, 0.225]
    return trn_mean, trn_std


class ImageNetDataModule(L.LightningModule):
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
def create_efficientnet_model(model_name, model_hparams, load_pretrained=False):
    if load_pretrained:
        print(f"Loading pretrained model {model_name} from torch hub")
        model = EfficientNet.from_pretrained(model_name, **model_hparams)
    else:
        model = EfficientNet.from_name(model_name, **model_hparams)
    return model

class ImageNetModule(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams,lr_scheduler_name, lr_scheduler_hparams):
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
        # Create model
        self.model = create_efficientnet_model(self.model_name,model_hparams,self.load_pretrained)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

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

def create_trainer(config):
    # save_name = config['model']['name'] +'_'+ config['wandb']['name']
    save_name = config['model']['name']
    logger = None
    if config.get("wandb",None) is not None:
        logger = WandbLogger(**config["wandb"])
    # else:
    #     logger = TensorBoardLogger("lightning_logs", name=save_name)

    callbacks= [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                LearningRateMonitor("epoch"),       # Log learning rate every epoch
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
        model = ImageNetModule.load_from_checkpoint(config['train']['ckpt'])
    else:
        model = ImageNetModule(model_name,config['model'],optimizer_name,config['optimizer'],lr_scheduler_name,config['lr_scheduler'])
    return model


def create_data_module(config):
    # check whether to use imagenet stats or TinyImageNet stats - depending on whether we use a pretrained model or not
    if config['model']['load_pretrained']:
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


    dataset_module = ImageNetDataModule(config['dataset'],transform=preprocess_transform)
    return dataset_module

        
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

    trainer.fit(model, datamodule=data_set)
    model = ImageNetModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training

    if config['train']['test']:
        val_result = trainer.test(model, datamodule=data_set)
        result = {"val": val_result[0]["test_acc"]}

    print(f"Test accuracy: {result['val']:.3f}")
    return model, result



config_defaults = {'global_seed':42,
                   'wandb':{'mode':'online',
                            'project':'bw-efcnt_imgnet',
                            'name':'bw_exp_b0',},
                    'dataset':{ 'data_dir':'/datasets/vision/imagenet/ILSVRC/Data/CLS-LOC',
                                'batch_size':64,
                                'image_size':224,
                                'recompute_stats':False,
                                'val_split':0.2},
                    'model':{ 'name':'efficientnet-b0', 'load_pretrained':False, 'num_classes':1000,'dropout_rate':0.5},
                    'optimizer':{ 'opt_name':'AdamW', 'lr':0.001, 'weight_decay':0.001},
                    'lr_scheduler':{ 'sched_name':'StepLR', 'step_size':5, 'gamma':0.98},
                    'trainer':{ 'max_epochs':300, 'devices':'auto','strategy':'auto','precision':32},
                    'train':{ 'ckpt':None, 'test':True, 'eval_only':False},
                    }


if __name__ == "__main__":
    # change config if needed
    config = config_defaults.copy()
    # Run the main function
    config['wandb']['name'] = 'exp_b0_imgnet_bn'
    # config['model']['name'] = 'efficientnet-b3'
    config['model']['batch_whitening_momentum'] = 0.1   # higher value for faster update of running_mean (more weight on curent batch statistics)
    # config['lr_scheduler']['sched_name'] = None
    # config['lr_scheduler']={'sched_name':'CosineAnnealingWarmRestarts', 'n_cycles':5, 'eta_min':0.1*config['optimizer']['lr']}

    config['trainer']['max_epochs'] = 300

    config['dataset']['batch_size'] = 64
    # config['trainer']['precision'] = 16
    config['trainer']['accumulate_grad_batches'] = 1
    print(config)

    main(config)

