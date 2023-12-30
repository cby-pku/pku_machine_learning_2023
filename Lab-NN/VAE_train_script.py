import torch.optim as optim
import torch
from VarAutoencoder import VarAutoencoder, VAE_loss_function, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM
from utils.train import train
from utils.scheduler import exponential_decay
import numpy as np
from utils.data_processor import create_flower_dataloaders
from argparse import ArgumentParser
import torch.nn.functional as F
import os, sys
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="AE", choices=["VAE", "AE"])
args = parser.parse_args()

# seed
torch.manual_seed(0)
np.random.seed(0)


data_root = "./flowers"
vis_root = "./vis"
model_save_root = "./model"

batch_size = 16
num_epochs = 21
early_stopping_patience = 5
model_class = VarAutoencoder if args.model == "VAE" else Autoencoder




model = model_class(
    encoding_dim=AE_ENCODING_DIM if args.model == "AE" else VAE_ENCODING_DIM,
)
optimizer = optim.Adam(model.parameters(), lr=0.001) # TODO: You can change this in Part 3 Step 2, for faster and better convergence.
scheduler = exponential_decay(initial_learning_rate=0.001, decay_rate=0.085, decay_epochs=20) # TODO: You can change this in Part 3 Step 2, for faster and better convergence.
#scheduler = CosineAnnealingLR(optimizer, T_max=10)
training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, 24, 24)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Use {device} for training {args.model}")

# Start training
train(
    optimizer=optimizer,
    scheduler=scheduler,
    model=model,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    num_epochs=num_epochs,
    early_stopping_patience=early_stopping_patience,
    device=device,
    model_save_root=model_save_root,
    loss_fn=VAE_loss_function if args.model == "VAE" else F.mse_loss,
)







