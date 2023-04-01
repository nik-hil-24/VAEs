import os
import torch
import numpy as np
from tqdm import tqdm
from model import VAE
from torch import nn, optim
from matplotlib import animation
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms, datasets
from matplotlib.animation import FuncAnimation

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Model Parameters
image_size = 28
input_dim = 784
h_dim = 200
z_dim = 20
# Training
EPOCHS = 50
BATCH_SIZE = 128
LR = 3e-4           # Karpathy Constant
TRANSFORM = transforms.ToTensor()

# Datatset
dataset = datasets.MNIST(root = '', train = True, transform = TRANSFORM, download = True)
train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)

# Training
def train(net, loader, opt, loss_fn):
    loop = tqdm(enumerate(loader))
    for batch_idx, (x, _) in loop:
        # Forward Pass
        x = x.to(device).view(x.shape[0], input_dim)
        x_recons, mu, sigma = net(x)

        # Loss
        recons_loss = loss_fn(x_recons, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        # Backward Pass
        loss = recons_loss + kl_div
        opt.zero_grad()
        loss.backward()
        opt.step()
        loop.set_postfix(loss = loss.item())


def main():
    model = VAE(input_dim, h_dim, z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = LR)
    loss = nn.BCELoss(reduction = 'sum')
    
    for epoch in range(EPOCHS):
        print(f'[{epoch+1}/{EPOCHS}] Epochs')
        train(model, train_loader, optimizer, loss)

    return model

model = main()

# Saving Generated Images
def inference(digit, num_examples=1):
    # Folder for Images to be Stored in
    if 'inference' not in os.listdir('VAE'):
        os.mkdir('inference')
    
    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encoder(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decoder(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"inference/generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=5)
