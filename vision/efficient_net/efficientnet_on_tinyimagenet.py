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
DATA_DIR = '/datasets/vision/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')
CHECKPOINT_PATH = "saved_models"


############################################
# prepare the validation folder
# Unlike training folder where images are already arranged in sub folders based 
# on their labels, images in validation folder are all inside a single folder. 
# Validation folder comes with images folder and val_annotations txt file. 

# Create separate validation subfolders for the validation images based on
# their labels indicated in the val_annotations txt file
def prepare_validation_folder(val_img_dir):
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



############################################
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
    else:
        trn_mean = [0.4802, 0.4481, 0.3975]
        trn_std = [0.2296, 0.2263, 0.2255]
    return trn_mean, trn_std



############################################
# Setup function to create dataloaders for image datasets
def generate_dataloader(data, name, transform, batch_size=64, use_cuda=True):
    if data is None: 
        return None
    
    # Read image files to pytorch dataset using ImageFolder, a generic data 
    # loader where images are in format root/label/filename
    # See https://pytorch.org/vision/stable/datasets.html
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=T.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    # Set options for device
    if use_cuda:
        kwargs = {"pin_memory": True, "num_workers": 1}
    else:
        kwargs = {}
    
    # Wrap image dataset (defined above) in dataloader 
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=(name=="train"), 
                        **kwargs)
    
    return dataloader

def create_data_loader(data_dir,split_name, batch_size=64, trn_mean=None, trn_std=None, use_cuda=True):
    # Define the transformation to be applied to the images
    if trn_mean is None or trn_std is None:  # use imagenet statistics
        preprocess_transform = T.Compose([
                    T.Resize(256), # Resize images to 256 x 256
                    T.CenterCrop(224), # Center crop image
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),  # Converting cropped images to tensors
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    else: # use custom statistics 
        preprocess_transform = T.Compose([
                        T.Resize(256), # Resize images to 256 x 256
                        T.CenterCrop(224), # Center crop image
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),  # Converting cropped images to tensors
                        T.Normalize(mean=trn_mean, std=trn_std)  
        ])

    # Create dataloader
    return generate_dataloader(data_dir, split_name,preprocess_transform, batch_size, use_cuda)




############################################
# creating the model and LigningModule
def create_efficientnet_model(model_name, model_hparams):
    load_pretrained = model_hparams.pop('load_pretrained', False)
    if load_pretrained:
        model = EfficientNet.from_pretrained(model_name, **model_hparams)
    else:
        model = EfficientNet.from_name(model_name, **model_hparams)
    return model

class TinyImageNetModule(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """TinyImageNetModule.

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_efficientnet_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)


############################################
# Training the model
def train_model(model_name, trainer_params, save_name=None, eval_only=False,use_cuda=True,**kwargs):
    """Train model.

    Args:
        model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
    """
    save_name = model_name+save_name

    # add wandb logger
    wandb_logger = WandbLogger(name=save_name, project="tiny-imagenet")

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models and tensorboard logs
        # We run on a single GPU (if possible)
        accelerator="auto",
        # devices=trainer_params['devices'],
        # strategy=trainer_params['strategy'],
        # # How many epochs to train for if no patience is set
        # max_epochs=trainer_params["max_epochs"],
        logger = wandb_logger,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
        **trainer_params
    )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    # trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need


    # Create dataloaders
    batch_size = 128
    trn_mean, trn_std = get_training_stats(recompute_stats=False)
    train_loader = create_data_loader(TRAIN_DIR, "train", batch_size,trn_mean, trn_std, use_cuda)
    val_loader = create_data_loader(VALID_DIR, "val", batch_size, trn_mean, trn_std, use_cuda)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if eval_only and os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = TinyImageNetModule.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducible
        model = TinyImageNetModule(model_name=model_name, **kwargs)

        trainer.fit(model, train_loader, val_loader)
        model = TinyImageNetModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    # test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    result = {"val": val_result[0]["test_acc"]}
    
    return model, result




def main():
    # Define device to use (CPU or GPU). CUDA = GPU support for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print("Device to use:", device)
    L.seed_everything(42)
    wandb.login()

    val_img_dir = os.path.join(VALID_DIR, 'images')
    prepare_validation_folder(val_img_dir)

    # imgs,lbls = next(iter(train_loader))
    # print('Image shape:', imgs.shape)
    # print('Label shape:', lbls.shape)


    model_hparams = {'load_pretrained': False, 'num_classes': 200}
    optimizer_name = "Adam"
    optimizer_hparams = {'lr': 0.01, 'weight_decay': 0.0001}
    # trainer_params = {'max_epochs': 10,'devices':'auto','strategy':'ddp'}
    trainer_params={'max_epochs': 300,'devices':'auto'}

    model,result = train_model('efficientnet-b0', trainer_params, save_name='exp4',use_cuda=use_cuda,model_hparams=model_hparams, optimizer_name=optimizer_name, optimizer_hparams=optimizer_hparams)

    print(f"Test accuracy: {result['val']:.3f}")
    wandb.finish()






if __name__ == "__main__":

    # Run the main function
    main()

