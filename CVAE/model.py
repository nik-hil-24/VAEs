import torch
from torch import nn


# x -> hidden dim -> mean, std -> Parametrization trick -> decoder -> output image
class CVAE(nn.Module):
    def __init__(self, feature_dim, class_dim, hidden_dim = 200, z_dim = 20):
        super().__init__()
        # Sizes
        self.feature_dim = feature_dim
        self.class_dim = class_dim

        # Encoder
        self.image_2hid = nn.Linear(feature_dim+class_dim, hidden_dim)
        self.hid_2mu = nn.Linear(hidden_dim, z_dim)
        self.hid_2sigma = nn.Linear(hidden_dim, z_dim)
        
        # Decoder
        self.z_2hid = nn.Linear(z_dim+class_dim, hidden_dim)
        self.hid_2img = nn.Linear(hidden_dim, feature_dim)
        
        self.relu = nn.ReLU()
        
    def encoder(self, x, c):
        # q_phi(z/x, c)
        inputs = torch.cat([x, c], 1)
        h = self.relu(self.image_2hid(inputs))
        self.mu = self.hid_2mu(h)
        self.sigma = self.hid_2sigma(h)
        
    def decoder(self, z, c):
        # p_theta(x/z, c)
        inputs = torch.cat([z, c], 1)
        h = self.relu(self.z_2hid(inputs))

        return torch.sigmoid((self.hid_2img(h)))
        
    def reparametrize(self):
        # Reparametrization for Gaussian Distribution
        eps = torch.rand_like(self.sigma)
        
        return self.mu + self.sigma*eps
    
    def generate(self, c):
        # Generate Data
        z = self.reparametrize()
        
        return self.decoder(z, c)
         
    def forward(self, x, c):
        # Encode
        self.encoder(x, c)
        # Reparametrize
        z = self.reparametrize()
        # Decode
        x_recons = self.decoder(z, c)
        
        return x_recons, self.mu, self.sigma
        
        
if __name__ == '__main__':
    x = torch.randn(1, 784)
    c = torch.tensor([[0,0,1]])
    cvae = CVAE(784, 3)
    x_recons, mu, sigma = cvae(x,c)
    print(x_recons.shape)
    print(mu.shape)
    print(sigma.shape)