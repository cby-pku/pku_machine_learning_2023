import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64
IMG_WIDTH, IMG_HEIGHT = 24, 24
# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        
        # TODO: implement the encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 *IMG_HEIGHT//4*IMG_WIDTH//4, encoding_dim)


        

    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        
        # TODO: implement the forward pass
        x = F.relu(self.pool(self.conv1(x)))#(16,32,12,12)
        x = F.relu(self.pool(self.conv2(x)))#(16,64,6,6)
        x = F.relu(self.conv3(x)) #(16,128,6,6)
        x = x.view( x.size(0),-1 )
        v = self.fc1(x)
        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        # TODO: implement the decoder
        self.fc2 = nn.Linear(encoding_dim, 128*IMG_HEIGHT//4*IMG_WIDTH//4)
        #-----------------Code use transpose2d-----------------------------------
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.deconv6 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1,output_padding=1)

        #-----------------Code use upsampe + conv as substitue------------------------------
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        #self.upsample2 = nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)

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


# Combine the Encoder and Decoder to make the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        v = self.encoder(x)
        x = self.decoder(v)
        return x
    
    @property
    def name(self):
        return "AE"

