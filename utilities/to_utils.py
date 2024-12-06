from cv2 import connectedComponentsWithStats
import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

def eval_dpp_div(batch):
    batch = batch<=128
    x = batch.reshape(batch.shape[0], -1).astype(np.float64)
    x = x/np.sqrt(x.shape[1])
    r = np.sum(np.square(x), axis=1, keepdims=True)
    D = r - 2 * np.dot(x, x.T) + r.T
    S = np.exp(-0.5 * np.square(D))
    try:
        eig_val, _ = np.linalg.eigh(S)
    except: 
        eig_val = np.ones(x.shape[0])
    loss = -np.mean(np.log(np.maximum(eig_val, 1e-10)))
    return loss
def eval_batch_validity(batch):
    
    validity = []
    all_areas = []
    for i in range(len(batch)):
        num_labels, _,b,_ = connectedComponentsWithStats((batch[i]<=128).astype(np.uint8), connectivity=8)
        if num_labels <= 2:
            tot_area=0
        else:
            areas = []
            for i in range(1, num_labels):
                area = b[i,-1]
                areas.append(area)
            areas = np.array(areas)
            tot_area = sum(areas) - max(areas)
        valid = num_labels == 2
        validity.append(valid)
        all_areas.append(tot_area)
    return np.array(validity), np.array(all_areas)

def evaluate_n_batches(netG, device, nz, batches=1000, batch_size=128):
    with torch.no_grad():
        all_valid = []
        all_areas = []
        all_diversity = []
        for i in trange(batches):
            fake = netG(torch.randn(batch_size, nz, 1, 1, device=device)).detach().cpu()
            fake = (fake>0)*255
            valid, area = eval_batch_validity((fake[:,0,:,:]).numpy().astype(np.uint8))
            #flatten last two dims to one
            all_valid.append(valid)
            all_areas.append(area)
            all_diversity.append(eval_dpp_div(fake.numpy()))
        all_valid = np.concatenate(all_valid)
        all_areas = np.concatenate(all_areas)
        all_diversity = np.array(all_diversity)
    return all_valid, all_areas, all_diversity

class DataNotFoundError(Exception):
    """Custom exception for missing data files."""
    pass

def load_data(directory):
    try:
        all_images = torch.load(os.path.join(directory, "all_images.pth"))
        labels = np.load(os.path.join(directory, 'labels.npy'))

        P = all_images[labels == 1]
        N_procedural = all_images[labels == 0] 
        N_real = N_procedural[35000:]  # Any after the first 35k are real
        N_procedural = N_procedural[:35000]  # The first 35k are synthetic

        N_rejected = torch.load(os.path.join(directory, "GAN_generated_negatives.pth"))[:, 0, :, :]  # The GAN generated negatives
        N_rejected = (N_rejected > 0) * 255

        return P, N_real, N_procedural, N_rejected

    except FileNotFoundError as e:
        raise DataNotFoundError(
            f"Missing file: {e.filename}. Did you download the TO data? Please ensure all required data files are in the directory."
        ) from e

def augment_all(data):
    def augment(image):
        return torch.stack([image, image.flip(0), image.flip(1), image.flip(0).flip(1), image.transpose(0, 1), image.transpose(0, 1).flip(0), image.transpose(0, 1).flip(1), image.transpose(0, 1).flip(0).flip(1)], 0)
    return torch.cat([augment(data[i]) for i in range(data.shape[0])], 0)


class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        return self.main(input)
    
class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, nc, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input).view(input.size(0), -1)