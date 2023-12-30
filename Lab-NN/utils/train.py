import torch
import torch.nn.functional as F
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(
    optimizer, 
    scheduler, 
    model, 
    training_dataloader, 
    validation_dataloader,
    num_epochs,
    early_stopping_patience,
    device,
    model_save_root,
    loss_fn=F.mse_loss,
    ):

    model.train()
    model_name = model.name
    save_model_name = f"Best_{model_name}.pth"
    
    min_valid_loss = float('inf')
    # Training Loop
    avg_train_loss = 10000.
    avg_valid_loss = 10000.
    for epoch in range(num_epochs):
        
        train_losses = []
        
        # Adjust the learning rate
        lr = scheduler(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        step_num = len(training_dataloader)
        
        # Training all batches
        for images, _ in training_dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            #print(f"debug for outputs.shape{outputs.shape}")
            #print(f"debug for images.shape {images.shape}")
            # Calculate the loss
            loss = loss_fn(outputs, images)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                for images, _ in validation_dataloader:
                    images = images.to(device)
                    
                    outputs = model(images)
                    #print(f"debug for outputsshape{outputs.shape}")
                    #print(f"debug for images.shape {images.shape}")
                    
                    loss = loss_fn(outputs, images)
                    
                    valid_losses.append(loss.item())
                avg_valid_loss = sum(valid_losses) / len(valid_losses)
            
            
            if avg_valid_loss < min_valid_loss:
                min_valid_loss = avg_valid_loss
                best_model = model.state_dict()  # Save the best model weights
                torch.save(best_model, f"{model_save_root}/{save_model_name}")
                    
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= early_stopping_patience:
                print("Early stopping triggered!")
                break
            
        print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")
        





    

    
    