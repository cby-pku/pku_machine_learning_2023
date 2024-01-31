from autograd.BaseGraph import Graph
from autograd.Nodes import relu, Linear, MSE, sigmoid
from utils.scheduler import exponential_decay
import numpy as np
from utils.data_processor import create_flower_dataloaders
import pickle
import torch
import os

# **数据维度变化如下：3\*H\*W -> 256 -> 128 -> 64 (Latent vector) -> 128 -> 256 -> 3\*H\*W；使用ReLU作为激活函数；确保输出值位于[0, 1]范围内
# set random seed
np.random.seed(42)
torch.manual_seed(42)

# Basic settings
data_root = "./flowers"
vis_root = "./vis"
model_save_path = "./model/Best_MLPAE.pkl"

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

batch_size = 16
num_epochs = 100
early_stopping_patience = 15
IMG_WIDTH, IMG_HEIGHT = 24, 24


scheduler = exponential_decay(initial_learning_rate=5.0, decay_rate=1.0, decay_epochs=15)
training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, IMG_WIDTH, IMG_HEIGHT)


model = Graph([
    # TODO: Please implement the MLP autoencoder model here (Both encoder and decoder).
    Linear(3* IMG_HEIGHT*IMG_WIDTH, 256),
    relu(),
    Linear(256,128),
    relu(),
    Linear(128,64),
    relu(),
    Linear(64,128),
    relu(),
    Linear(128,256),
    relu(),
    Linear(256,3*IMG_HEIGHT *IMG_WIDTH),
    sigmoid()
]
)
loss_fn_node = MSE()


save_model_name = f"Best_MLPAE.pkl"
min_valid_loss = float('inf')
avg_train_loss = 10000.
avg_valid_loss = 10000.

for epoch in range(num_epochs):
    
    train_losses = []
    
    # Adjust the learning rate
    lr = scheduler(epoch)
    step_num = len(training_dataloader)
    
    # Training all batches
    for images, _ in training_dataloader:
        images = images.detach().numpy() # (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        model.flush()
        loss_fn_node.flush()
        # TODO: Please implement the training loop for a MLP autoencoder, which can perform forward, backward and update the parameters.
        images_flatten = images.reshape(images.shape[0], -1)
        # Forward Pass
        output = model.forward(images_flatten)
        loss =loss_fn_node.forward(output,images_flatten)
        grad = loss_fn_node.backward()
        model.backward(grad)
        model.optimstep(lr)
        
        
        train_losses.append(loss.item())
    avg_train_loss = sum(train_losses) / len(train_losses)
    
    # Validation every 3 epochs
    if epoch % 3 == 0:
        valid_losses = []
        for images, _ in validation_dataloader:
            images = images.detach().numpy() # (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
            # TODO: Please implement code which calculates the validation loss.
            images_flatten = images.reshape(images.shape[0],-1)
            output = model.forward(images_flatten)
            val_loss = loss_fn_node.forward(output,images_flatten)
            
            
            valid_losses.append(val_loss.item())
        avg_valid_loss = sum(valid_losses) / len(valid_losses)

        # TODO: Save model if better validation loss is achieved. Save the model by calling pickle.dump(model, open(model_save_path, 'wb')).
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            #pickle.dump(model,open(model_save_path),'wb')
            # NOTE I have added a line in front of the code to make sure the save path is existed, so accordingly the codeline here is changed
            with open(model_save_path, 'wb') as model_file:
                pickle.dump(model, model_file)
        
        
        # TODO: Early stopping if validation loss does not decrease for <early_stopping_patience> validation checks.
        if epoch > early_stopping_patience and avg_valid_loss>=min_valid_loss:
            #print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")
            print(f"Early stopping at epoch {epoch}.")
            break
            
        
        
    print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")

