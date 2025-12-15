import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

DATA_DIR = '/datasets/vision/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
# VALID_DIR = os.path.join(DATA_DIR, 'val')
VALID_DIR = os.path.join(DATA_DIR, 'val')
CHECKPOINT_PATH = "saved_models"


# Define the ConvNet model
class SimpleConvNet(LightningModule):
    def __init__(self, num_classes=200, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Simple ConvNet architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


class TinyImageNetDataModule(LightningDataModule):
    def __init__(self,data_dir=DATA_DIR,batch_size=64,val_split=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def prepare_data(self):
        '''
        - download the dataset
        - extract the dataset
        - create a validation folder
        '''
        val_img_dir = os.path.join(self.data_dir,'val','images')
        

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
    


# Define the training function
def train_model(max_epochs=100, ckpt_path=None, run_id=None):
    # Initialize the model
    model = SimpleConvNet(num_classes=200)
    
    # Initialize the data module
    data_module = TinyImageNetDataModule()
    
    # Set up WandB logger
    if run_id:
        # Resume an existing run
        wandb_logger = WandbLogger(project="tiny_imagenet_training", id=run_id, resume="must")
    else:
        # Start a new run
        wandb_logger = WandbLogger(project="tiny_imagenet_training", log_model="all")
    
    # Set up checkpoint callback to save top k models
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='tiny-imagenet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,  # Save the top 3 models
        mode='min',
        save_last=True  # Save the last checkpoint for resuming training
    )
    
    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )
    
    # Train the model
    if ckpt_path:
        # Resume training from checkpoint
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)
    else:
        # Start training from scratch
        trainer.fit(model, datamodule=data_module)
    
    return model, trainer, wandb_logger.experiment.id

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a ConvNet on Tiny ImageNet')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--run_id', type=str, default=None, help='WandB run ID to resume')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    if args.resume:
        if not args.checkpoint or not args.run_id:
            print("Error: Both --checkpoint and --run_id are required when resuming training")
        else:
            model, trainer, run_id = train_model(
                max_epochs=args.epochs,
                ckpt_path=args.checkpoint,
                run_id=args.run_id
            )
            print(f"Training resumed and completed. WandB run ID: {run_id}")
    else:
        model, trainer, run_id = train_model(max_epochs=args.epochs)
        print(f"Training completed. WandB run ID: {run_id}")
        print(f"Last checkpoint path: {trainer.checkpoint_callback.last_model_path}")
