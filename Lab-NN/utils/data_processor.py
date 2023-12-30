from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



def create_flower_dataloaders(batch_size, root, IMG_WIDTH, IMG_HEIGHT, num_worker=0):

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # ensure 3 channels
    ])


    flower_dataset = datasets.ImageFolder(root=root, transform=transform)

    train_size = int(0.8 * len(flower_dataset))
    valid_size = len(flower_dataset) - train_size

    train_dataset, valid_dataset = random_split(flower_dataset, [train_size, valid_size])

    training_dataloaders = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, pin_memory=True)
    validation_dataloaders = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)
    
    return training_dataloaders, validation_dataloaders


def show_images(images, nmax=64):
    img_grid = make_grid(images[:nmax], nrow=8).permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()


def show_recover_results(images, recover_images, save_path):
    
    num_images = images.shape[0]
    
    # show on two rows, first row is the original images, second row is the recovered images
    images = torch.tensor(images)
    recover_images = torch.tensor(recover_images)
    img_grid = make_grid(torch.cat([images, recover_images]), nrow=num_images).permute(1, 2, 0)
    plt.figure(figsize=(20, 20))
    
    img_grid = img_grid.cpu().numpy()
    
    plt.imshow(img_grid)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=1000)
        