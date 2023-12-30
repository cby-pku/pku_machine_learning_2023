import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64
IMG_WIDTH, IMG_HEIGHT = 24, 24
# Define the Variational Encoder
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarEncoder, self).__init__()
        # TODO: implement the encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 *IMG_HEIGHT//4*IMG_WIDTH//4, encoding_dim)


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward pass
        x = F.relu(self.pool(self.conv1(x)))#(16,32,12,12)
        x = F.relu(self.pool(self.conv2(x)))#(16,64,6,6)
        x = F.relu(self.conv3(x)) #(16,128,6,6)
        x = x.view( x.size(0),-1 )
        mu = self.fc1(x)
        log_var =self.fc1(x)
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarDecoder, self).__init__()
        # TODO: implement the decoder
        self.fc2 = nn.Linear(encoding_dim, 128*IMG_HEIGHT//4*IMG_WIDTH//4)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)

    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        # TODO: implement the forward pass'
        x = F.relu(self.fc2(v)) # (16,4608)
        x = x.view(x.size(0),128,IMG_HEIGHT//4,IMG_WIDTH//4)#(16,128,6,6)
        # NOTE use conv + upsample to substitue for ConvTranspose2d
        x = F.relu(self.conv4(x)) #(16,64,6,6)
        x = self.upsample(x) #(16,64,2,2)
        x = F.relu(self.conv5(x)) #(16,32,12,12)
        x = self.upsample(x) #(16,32,4,4)
        #x = self.upsample2(x)
        x = torch.sigmoid(self.conv6(x))#(16,3,24,24)
        return x

# Define the Variational Autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        '''
        mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        return v: sampled latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        
        # TODO: implement the reparameterization trick to sample v
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps *std
        return z
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        mu,log_var = self.encoder(x)
        v = self.reparameterize(mu,log_var)
        x_recon = self.decoder(v)
        return x_recon, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    '''
    outputs: (x, mu, log_var)
    images: input/original images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    return loss: the loss value, dim: (1)
    '''
    # TODO: implement the loss function for VAE
    x_recon, mu, log_var = outputs
    recon_loss = F.mse_loss(x_recon, images, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence
