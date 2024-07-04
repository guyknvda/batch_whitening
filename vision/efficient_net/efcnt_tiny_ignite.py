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

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import LRScheduler, ModelCheckpoint, global_step_from_engine
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
import ignite.contrib.engines.common as common

# import opendatasets as od
import os
from random import randint
import urllib
import zipfile

# Define device to use (CPU or GPU). CUDA = GPU support for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


DATA_DIR = '/datasets/vision/tiny-imagenet-200' # Original images come in shapes of [3,64,64]
# Define training and validation data paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train') 
VALID_DIR = os.path.join(DATA_DIR, 'val')


# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)    
    imshow(make_grid(images)) # Using Torchvision.utils make_grid function
    
def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = randint(0, len(images)-1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')

# Setup function to create dataloaders for image datasets
def generate_dataloader(data, name, transform):
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


val_img_dir = os.path.join(VALID_DIR, 'images')

# Define transformation sequence for image pre-processing
# If not using pre-trained model, normalize with 0.5, 0.5, 0.5 (mean and SD)
# If using pre-trained ImageNet, normalize with mean=[0.485, 0.456, 0.406], 
# std=[0.229, 0.224, 0.225])
preprocess_transform = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2296, 0.2263, 0.2255]) # 
])

preprocess_transform_pretrain = T.Compose([
                T.Resize(256), # Resize images to 256 x 256
                T.CenterCrop(224), # Center crop image
                T.RandomHorizontalFlip(),
                T.ToTensor(),  # Converting cropped images to tensors
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
])


batch_size = 32

train_loader = generate_dataloader(TRAIN_DIR, "train",transform=preprocess_transform)
val_loader = generate_dataloader(val_img_dir, "val",transform=preprocess_transform)


# train_loader = generate_dataloader(TRAIN_DIR, "train",transform=preprocess_transform_pretrain)
# val_loader = generate_dataloader(val_img_dir, "val",transform=preprocess_transform_pretrain)








# Define model architecture (using efficientnet-b3 version)
from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=200)
model = EfficientNet.from_name('efficientnet-b0', num_classes=200)

# Move model to designated device (Use GPU when on Colab)
model = model.to(device)

# Define hyperparameters and settings
lr = 0.001  # Learning rate
num_epochs = 300  # Number of epochs
log_interval = 300  # Number of iterations before logging

# Set loss function (categorical Cross Entropy Loss)
loss_func = nn.CrossEntropyLoss()

# Set optimizer (using Adam as default)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Setup pytorch-ignite trainer engine
trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)

# Add progress bar to monitor model training
ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"Batch Loss": x})

# Define evaluation metrics
metrics = {
    "accuracy": Accuracy(), 
    "loss": Loss(loss_func),
}

# Setup pytorch-ignite evaluator engines. We define two evaluators as they do
# not have exactly similar roles. `evaluator` will save the best model based on 
# validation score, whereas `train_evaluator` logs metrics on training set only

# Evaluator for training data
train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Evaluator for validation data
evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

# Display message to indicate start of training
@trainer.on(Events.STARTED)
def start_message():
    print("Begin training")

# Log results from every batch
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_batch(trainer):
    batch = (trainer.state.iteration - 1) % trainer.state.epoch_length + 1
    print(
        f"Epoch {trainer.state.epoch} / {num_epochs}, "
        f"Batch {batch} / {trainer.state.epoch_length}: "
        f"Loss: {trainer.state.output:.3f}"
    )

# Evaluate and print training set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(trainer):
    print(f"Epoch [{trainer.state.epoch}] - Loss: {trainer.state.output:.2f}")
    train_evaluator.run(train_loader)
    epoch = trainer.state.epoch
    metrics = train_evaluator.state.metrics
    print(f"Train - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f} "
          )

# Evaluate and print validation set metrics
@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_loss(trainer):
    evaluator.run(val_loader)
    epoch = trainer.state.epoch
    metrics = evaluator.state.metrics
    print(f"Validation - Loss: {metrics['loss']:.3f}, "
          f"Accuracy: {metrics['accuracy']:.3f}"
          )
    print()
    print("-" * 60)
    print()

# Sets up checkpoint handler to save best n model(s) based on validation accuracy metric
common.save_best_model_by_val_score(
          output_path="best_models",
          evaluator=evaluator,
          model=model,
          metric_name="accuracy",
          n_saved=1,
          trainer=trainer,
          tag="val"
)

# Define a Tensorboard logger
tb_logger = TensorboardLogger(log_dir="logs")

# Using common module to setup tb logger (Alternative method)
# tb_logger = common.setup_tb_logging("tb_logs", trainer, optimizer, evaluators=evaluator)

# Attach handler to plot trainer's loss every n iterations
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=log_interval),
    tag="training",
    output_transform=lambda loss: {"Batch Loss": loss},
)

# Attach handler to dump evaluator's metrics every epoch completed
for tag, evaluator in [("training", train_evaluator), ("validation", evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )


# Start training
trainer.run(train_loader, max_epochs=num_epochs)

# Close Tensorboard
tb_logger.close()


print(evaluator.state.metrics)
