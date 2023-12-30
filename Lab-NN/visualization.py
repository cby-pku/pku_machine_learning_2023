import numpy as np
from utils.data_processor import create_flower_dataloaders, show_recover_results
import pickle
from argparse import ArgumentParser



parser = ArgumentParser()
parser.add_argument("--model", type=str, default="MLPAE", choices=["VAE", "AE", "MLPAE"])
args = parser.parse_args()

# set random seed
np.random.seed(0)

data_root = "./flowers"
vis_root = "./vis"

model_ckpt = f"./model/Best_{args.model}.pth"

batch_size = 16

if args.model == "MLPAE":
    
    training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, 24, 24)
    
    model = pickle.load(open("./model/Best_MLPAE.pkl", "rb"))

    # visualize the results
    train_images_sampled = next(iter(training_dataloader))[0].detach().numpy() 
    valid_images_sampled = next(iter(validation_dataloader))[0].detach().numpy() 

    train_outputs = model.forward(train_images_sampled.reshape(-1, 24 * 24 * 3)).reshape(-1, 3, 24, 24)
    valid_outputs = model.forward(valid_images_sampled.reshape(-1, 24 * 24 * 3)).reshape(-1, 3, 24, 24)

    show_recover_results(train_images_sampled, train_outputs, f"{vis_root}/train_MLPAE.png")
    show_recover_results(valid_images_sampled, valid_outputs, f"{vis_root}/valid_MLPAE.png")
    
else:
    # load the best model
    import torch
    from Autoencoder import Autoencoder, AE_ENCODING_DIM
    from VarAutoencoder import VarAutoencoder, VAE_ENCODING_DIM
    
    torch.manual_seed(0)
    
    training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, 24, 24)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class = VarAutoencoder if args.model == "VAE" else Autoencoder

    model = model_class(
        encoding_dim=AE_ENCODING_DIM if args.model == "AE" else VAE_ENCODING_DIM,
    )
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()
    model.to(device)

    # visualize the results
    train_images_sampled = next(iter(training_dataloader))[0].to(device)
    valid_images_sampled = next(iter(validation_dataloader))[0].to(device)

    train_outputs = model(train_images_sampled)
    valid_outputs = model(valid_images_sampled)

    if args.model == "VAE":
        train_outputs = train_outputs[0]
        valid_outputs = valid_outputs[0]

    show_recover_results(train_images_sampled, train_outputs, f"{vis_root}/train_{args.model}.png")
    show_recover_results(valid_images_sampled, valid_outputs, f"{vis_root}/valid_{args.model}.png")