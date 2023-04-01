import torch
from torch import nn


# x -> hidden dim -> mean, std -> Parametrization trick -> decoder -> output image
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim = 200, z_dim = 20):
        super().__init__()
        
        # Encoder
        self.image_2hid = nn.Linear(input_dim, hidden_dim)
        self.hid_2mu = nn.Linear(hidden_dim, z_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.z_2hid = nn.Linear(z_dim, hidden_dim)
        self.hid_2img = nn.Linear(hidden_dim, input_dim)
        
        self.relu = nn.ReLU()
        
    def encoder(self, x):
        # q_phi(z/x)
        h = self.relu(self.image_2hid(x))
        mu = self.hid_2mu(h)
        sigma = self.hid_2sigma(h)
        
        return mu, sigma
        
    def decoder(self, z):
        # p_theta(x/z)
        h = self.relu(self.z_2hid(z))
        
        return torch.sigmoid((self.hid_2img(h)))
        
        
    def forward(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.rand_like(sigma)
        z = mu + sigma*eps
        x_recons = self.decoder(z)
        
        return x_recons, mu, sigma
        
        
if __name__ == '__main__':
    x = torch.randn(2, 784)
    vae = VAE(784)
    x_recons, mu, sigma = vae(x)
    print(x_recons.shape)
    print(mu.shape)
    print(sigma.shape)