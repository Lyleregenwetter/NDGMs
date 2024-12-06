from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np

class Down_Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=400, num_hidden_layers=1):
        super(Down_Model, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

class Up_Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=400, num_hidden_layers=1):
        super(Up_Model, self).__init__()
        
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim), nn.LeakyReLU()])
        
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.LeakyReLU())
        
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

def diversity_loss(x):
    # Compute pairwise squared Euclidean distances
    r = torch.sum(x ** 2, dim=1, keepdim=True)
    D = r - 2 * torch.matmul(x, x.T) + r.T
    
    # Compute the similarity matrix using RBF
    S = torch.exp(-0.5 * D ** 2)
    
    # Compute the eigenvalues of the similarity matrix
    try:
        eig_val = torch.linalg.eigvalsh(S)
    except:
        eig_val = torch.ones(x.size(0), device=x.device)
    
    # Compute the loss as the negative mean log of the eigenvalues
    loss = -torch.mean(torch.log(torch.clamp(eig_val, min=1e-7)))
    
    return loss

def GAN_step_vanilla(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    criterion = nn.BCEWithLogitsLoss()
    D.zero_grad()
    real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

    output = D(P_batch).view(-1)
    L_D_real = criterion(output, real_label)

    fake_data = G(noise_batch)
    output = D(fake_data.detach()).view(-1)
    L_D_fake = criterion(output, fake_label)

    L_D_tot = L_D_real + L_D_fake
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data).view(-1)
    L_G = criterion(output, real_label)

    if diversity_weight > 0:
        _, L_div = diversity_loss(fake_data)
        L_G_tot = L_G + diversity_weight * L_div
    else:
        L_G_tot = L_G
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    report = {"L_D_real": L_D_real.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report

def GAN_step_clf(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    criterion = nn.BCEWithLogitsLoss()
    D.zero_grad()
    real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

    output = D(P_batch).view(-1)
    L_D_real = criterion(output, real_label)

    fake_data = G(noise_batch)
    output = D(fake_data.detach()).view(-1)
    L_D_fake = criterion(output, fake_label)

    L_D_tot = L_D_real + L_D_fake
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data).view(-1)
    L_G = criterion(output, real_label)

    output_clf = A(fake_data).view(-1)
    L_A = criterion(output_clf, real_label)

    if diversity_weight > 0:
        _, L_div = diversity_loss(fake_data)
        L_G_tot = L_G + validity_weight * L_A + diversity_weight * L_div
    else:
        L_G_tot = L_G + validity_weight * L_A
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    report = {"L_D_real": L_D_real.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item(), "L_clf": L_A.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report

def GAN_step_cond(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    criterion = nn.BCEWithLogitsLoss()

    z = torch.full((batch_size,), 0, dtype=torch.float, device=device).view(-1, 1)
    o = torch.full((batch_size,), 1, dtype=torch.float, device=device).view(-1, 1)
    real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

    D.zero_grad()
    
    P_cond = torch.concat([P_batch, o], dim=1)
    output = D(P_cond).view(-1)
    L_D_real_pos = criterion(output, real_label)

    N_cond = torch.concat([N_batch, z], dim=1)
    output = D(N_cond).view(-1)
    L_D_real_neg = criterion(output, real_label)

    noise_pos = torch.concat([noise_batch, o], dim=1)
    fake_pos = G(noise_pos)
    fake_pos = torch.concat([fake_pos, o], dim=1)
    output = D(fake_pos.detach()).view(-1)
    L_D_fake_pos = criterion(output, fake_label)

    noise_neg = torch.concat([noise_batch, z], dim=1)
    fake_neg = G(noise_neg)
    fake_neg = torch.concat([fake_neg, z], dim=1)
    output = D(fake_neg.detach()).view(-1)
    L_D_fake_neg = criterion(output, fake_label)

    L_D_tot = L_D_real_pos + L_D_fake_pos + L_D_real_neg + L_D_fake_neg
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()

    noise_pos = torch.concat([noise_batch, o], dim=1)
    fake_data = G(noise_pos)
    fake_data = torch.concat([fake_data, o], dim=1)
    output = D(fake_data).view(-1)
    L_G_pos = criterion(output, real_label)

    noise_neg = torch.concat([noise_batch, z], dim=1)
    fake_data = G(noise_neg)
    fake_data = torch.concat([fake_data, z], dim=1)
    output = D(fake_data).view(-1)
    L_G_neg = criterion(output, real_label)



    if diversity_weight > 0:
        _, L_div = diversity_loss(fake_data)
        L_G_tot = L_G_pos + L_G_neg + diversity_weight * L_div
    else:
        L_G_tot = L_G_pos + L_G_neg
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    report = {"L_D_real_pos": L_D_real_pos.item(), "L_D_fake_pos": L_D_fake_pos.item(), "L_D_real_neg": L_D_real_neg.item(), "L_D_fake_neg": L_D_fake_neg.item(), "L_G_pos": L_G_pos.item(), "L_G_neg": L_G_neg.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report


def GAN_step_DO(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    criterion = nn.BCEWithLogitsLoss()
    D.zero_grad()

    negative_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)
    real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
    fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

    output = D(P_batch).view(-1)
    L_D_real = criterion(output, real_label)

    output = D(N_batch).view(-1)
    L_D_neg = criterion(output, negative_label)

    fake_data = G(noise_batch)
    output = D(fake_data.detach()).view(-1)
    L_D_fake = criterion(output, fake_label)

    L_D_tot = L_D_real + L_D_fake + L_D_neg
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data).view(-1)
    L_G = criterion(output, real_label)

    if diversity_weight > 0:
        L_div = diversity_loss(fake_data)
        L_G_tot = L_G + diversity_weight * L_div
    else:
        L_G_tot = L_G
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    # L_G.backward()
    # G_opt.step()

    report = {"L_D_real": L_D_real.item(), "L_D_neg": L_D_neg.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report


def GAN_step_DDD(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=0.5, diversity_weight=0):
    criterion = nn.BCEWithLogitsLoss()
    D.zero_grad()

    z = torch.full((batch_size,), 0, dtype=torch.float, device=device)
    o = torch.full((batch_size,), 1, dtype=torch.float, device=device)

    output = D(P_batch).view(-1)
    L_D_real = criterion(output, o)

    fake_data = G(noise_batch)
    output = D(fake_data.detach()).view(-1)
    L_D_fake = criterion(output, z)

    L_D_tot = L_D_real + L_D_fake
    L_D_tot.backward()
    D_opt.step()

    A.zero_grad()
    output = A(N_batch).view(-1)
    L_A_neg = criterion(output, o)

    fake_data = G(noise_batch)
    output = A(fake_data.detach()).view(-1)
    L_A_fake = criterion(output, z)

    L_A_tot = L_A_neg + L_A_fake
    L_A_tot.backward()
    A_opt.step()

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data).view(-1)
    L_G_D = criterion(output, o)

    output = A(fake_data).view(-1)
    L_G_A = validity_weight * criterion(output, z)

    if diversity_weight > 0:
        L_div = diversity_loss(fake_data)
        L_G_tot = L_G_D + L_G_A + diversity_weight * L_div
    else:
        L_G_tot = L_G_D + L_G_A
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    report = {"L_D_real": L_D_real.item(), "L_D_fake": L_D_fake.item(), "L_G_D": L_G_D.item(), "L_A_neg": L_A_neg.item(), "L_A_fake": L_A_fake.item(), "L_G_A": L_G_A.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report

def GAN_step_MDD(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    criterion = nn.CrossEntropyLoss()
    D.zero_grad()

    t = torch.full((batch_size,), 2, dtype=torch.long, device=device)
    o = torch.full((batch_size,), 1, dtype=torch.long, device=device)
    z = torch.full((batch_size,), 0, dtype=torch.long, device=device)

    output = D(P_batch)
    L_D_real = criterion(output, o)

    output = D(N_batch)
    L_D_neg = criterion(output, t)

    fake_data = G(noise_batch)
    output = D(fake_data.detach())
    L_D_fake = criterion(output, z)

    L_D_tot = L_D_real + L_D_fake + L_D_neg
    L_D_tot.backward()
    D_opt.step()

    G.zero_grad()
    fake_data = G(noise_batch)
    output = D(fake_data)
    L_G = criterion(output, o)

    if diversity_weight > 0:
        L_div = diversity_loss(fake_data)
        L_G_tot = L_G + diversity_weight * L_div
    else:
        L_G_tot = L_G
        L_div = None

    L_G_tot.backward()
    G_opt.step()

    report = {"L_D_real": L_D_real.item(), "L_D_neg": L_D_neg.item(), "L_D_fake": L_D_fake.item(), "L_G": L_G.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report


def VAE_step_vanilla(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    
    D.zero_grad()
    G.zero_grad()
    
    alpha = 0.2

    encoded = D(P_batch)
    latent_dim = encoded.shape[1] // 2
    mu = encoded[:, :latent_dim]
    logvar = encoded[:, latent_dim:]
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # z = mu + sigma * epsilon
    
    # Forward pass through decoder (G)
    reconstructed = G(z)
    
    # Compute losses
    L_R = nn.MSELoss()(reconstructed, P_batch)
    L_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / P_batch.size(0)

    if diversity_weight > 0:
        L_div = diversity_loss(z)
        L_tot = alpha * L_KL + L_R + diversity_weight * L_div
    else:
        L_div = None
        L_tot = alpha * L_KL + L_R

    L_tot.backward()
    
    D_opt.step()
    G_opt.step()
    
    report = {"L_KL": L_KL.item(), "L_R": L_R.item(), "L_tot": L_tot.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report

def VAE_step_cond(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    
    p = torch.full((batch_size,), 1, dtype=torch.float, device=device).view(-1, 1)
    n = torch.full((batch_size,), 0, dtype=torch.float, device=device).view(-1, 1)

    D.zero_grad()
    G.zero_grad()
    
    alpha = 0.2

    P_batch_cond = torch.concat([P_batch, p], dim=1)
    N_batch_cond = torch.concat([N_batch, n], dim=1)

    encoded_pos = D(P_batch_cond)
    latent_dim = encoded_pos.shape[1] // 2
    mu_pos = encoded_pos[:, :latent_dim]
    logvar_pos = encoded_pos[:, latent_dim:]
    
    std = torch.exp(0.5 * logvar_pos)
    eps = torch.randn_like(std)
    z = mu_pos + eps * std  # z = mu + sigma * epsilon

    z_pos = torch.concat([z, p], dim=1)
    
    reconstructed_pos = G(z_pos)
    
    L_R_pos = nn.MSELoss()(reconstructed_pos, P_batch)
    L_KL_pos = -0.5 * torch.sum(1 + logvar_pos - mu_pos.pow(2) - logvar_pos.exp()) / P_batch.size(0)

    encoded_neg = D(N_batch_cond)
    mu_pos = encoded_neg[:, :latent_dim]
    logvar_pos = encoded_neg[:, latent_dim:]

    std = torch.exp(0.5 * logvar_pos)
    eps = torch.randn_like(std)
    z = mu_pos + eps * std  # z = mu + sigma * epsilon

    z_neg = torch.concat([z, n], dim=1)

    reconstructed_neg = G(z_neg)

    L_R_neg = nn.MSELoss()(reconstructed_neg, N_batch)
    L_KL_neg = -0.5 * torch.sum(1 + logvar_pos - mu_pos.pow(2) - logvar_pos.exp()) / N_batch.size(0)

    L_R_tot = L_R_pos + L_R_neg
    L_KL_tot = L_KL_pos + L_KL_neg

    if diversity_weight > 0:
        L_div = diversity_loss(z)
        L_tot = alpha * L_KL_tot + L_R_tot + diversity_weight * L_div
    else:
        L_div = None
        L_tot = alpha * L_KL_tot + L_R_tot

    L_tot.backward()
    
    D_opt.step()
    G_opt.step()
    
    report = {"L_KL": L_KL_tot.item(), "L_R": L_R_tot.item(), "L_tot": L_tot.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report

def VAE_step_clf(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
    
    D.zero_grad()
    G.zero_grad()
    
    alpha = 0.2

    encoded = D(P_batch)
    latent_dim = encoded.shape[1] // 2
    mu = encoded[:, :latent_dim]
    logvar = encoded[:, latent_dim:]
    
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # z = mu + sigma * epsilon
    
    # Forward pass through decoder (G)
    reconstructed = G(z)
    
    # Compute losses
    L_R = nn.MSELoss()(reconstructed, P_batch)
    L_KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / P_batch.size(0)
    L_A = nn.BCEWithLogitsLoss()(A(reconstructed).view(-1), torch.full((batch_size,), 1, dtype=torch.float, device=device))

    if diversity_weight > 0:
        L_div = diversity_loss(z)
        L_tot = alpha * L_KL + L_R + validity_weight * L_A + diversity_weight * L_div
    else:
        L_div = None
        L_tot = alpha * L_KL + L_R + validity_weight * L_A

    L_tot.backward()
    
    D_opt.step()
    G_opt.step()
    
    report = {"L_KL": L_KL.item(), "L_R": L_R.item(), "L_tot": L_tot.item()}
    if L_div is not None:
        report["L_div"] = L_div.item()
    return report

class NoiseScheduler:
    def __init__(self, num_timesteps, beta_start=0.0001, beta_end=0.02, device="cpu"):

        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device(device)

        # Linear beta schedule
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device, dtype=self.betas.dtype), self.alpha_cumprod[:-1]])

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(torch.clamp(1.0 - self.alpha_cumprod, min=1e-8))

    def get_variance(self, t):
        if isinstance(t, torch.Tensor) and t.ndim > 0:  # Batched timesteps
            return torch.index_select(self.betas, 0, t).to(self.device)
        return self.betas[t].to(self.device)  # Single timestep


def DDPM_step_wrapper(scheduler):
    def DDPM_step(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
        t = torch.randint(0, scheduler.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(P_batch).to(device)

        sqrt_alpha_cumprod_t = scheduler.sqrt_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(alpha_t_bar)
        sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - alpha_t_bar)

        x_t = sqrt_alpha_cumprod_t * P_batch + sqrt_one_minus_alpha_cumprod_t * noise

        t_embedded = t.unsqueeze(-1).float() / scheduler.num_timesteps
        x_input = torch.cat([x_t, t_embedded], dim=-1)

        noise_pred = D(x_input)

        beta_t = scheduler.betas[t].unsqueeze(-1).to(device)  # Variance (beta_t)
        loss_weights = (1 / beta_t) / (1 / beta_t).mean()  # Normalize loss weights
        loss = (loss_weights * nn.MSELoss(reduction="none")(noise_pred, noise)).mean()

        D.zero_grad()
        loss.backward()
        D_opt.step()

        return {"loss": loss.item()}
    return DDPM_step

def DDPM_step_cond_wrapper(scheduler):
    def DDPM_step_cond(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=None, diversity_weight=0):
        P_labels = torch.ones(P_batch.size(0), 1, device=device)  # Class 1 for P_batch
        N_labels = torch.zeros(N_batch.size(0), 1, device=device)  # Class 0 for N_batch
        
        data_batch = torch.cat([P_batch, N_batch], dim=0)
        labels = torch.cat([P_labels, N_labels], dim=0)

        perm = torch.randperm(data_batch.size(0))
        data_batch = data_batch[perm]
        labels = labels[perm]

        t = torch.randint(0, scheduler.num_timesteps, (data_batch.size(0),), device=device)
        noise = torch.randn_like(data_batch).to(device)

        sqrt_alpha_cumprod_t = scheduler.sqrt_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(alpha_t_bar)
        sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - alpha_t_bar)

        x_t = sqrt_alpha_cumprod_t * data_batch + sqrt_one_minus_alpha_cumprod_t * noise

        t_embedded = t.unsqueeze(-1).float() / scheduler.num_timesteps

        x_input = torch.cat([x_t, labels, t_embedded], dim=-1)

        noise_pred = D(x_input)

        beta_t = scheduler.betas[t].unsqueeze(-1).to(device)  # Variance (beta_t)
        loss_weights = (1 / beta_t) / (1 / beta_t).mean()  # Normalize loss weights
        loss = (loss_weights * nn.MSELoss(reduction="none")(noise_pred, noise)).mean()

        # Backpropagation and optimization
        D.zero_grad()
        loss.backward()
        D_opt.step()

        return {"loss": loss.item()}
    return DDPM_step_cond


class ReusableDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        self.previous_indices = []

    def _shuffle_indices(self):
        self.indices = torch.randperm(len(self.dataset)).tolist()

    def get_batch(self):
        queued = self.previous_indices
        while len(queued) < self.batch_size:
            if self.shuffle:
                self._shuffle_indices()
            queued.extend(self.indices)  # Add individual elements to queued list
        
        self.previous_indices = queued[self.batch_size:]  # Store remaining indices for the next batch
        batch_indices = queued[:self.batch_size]  # Get the batch of the correct size
        return torch.stack([self.dataset[i][0] for i in batch_indices])


def train(D, G, A, D_opt, G_opt, A_opt, P_loader, N_loader, num_steps, batch_size, noise_dim, train_step_fn, device, validity_weight, diversity_weight=0):
    # Loss function
    
    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch = P_loader.get_batch().to(device)
        N_batch = N_loader.get_batch().to(device)
        noise_batch = torch.randn(batch_size, noise_dim).to(device)

        report = train_step_fn(D, G, A, D_opt, G_opt, A_opt, P_batch, N_batch, noise_batch, batch_size, device, validity_weight=validity_weight, diversity_weight=diversity_weight)
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)
    return D, G, A

def pretrain_clf(A, A_opt, P_loader, N_loader, num_steps, batch_size, device):
    steps_range = trange(num_steps, position=0, leave=True)
    for step in steps_range:
        P_batch = P_loader.get_batch().to(device)
        N_batch = N_loader.get_batch().to(device)

        criterion = nn.BCEWithLogitsLoss()
        A.zero_grad()
        pos_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        neg_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

        output = A(P_batch).view(-1)
        L_A_real = criterion(output, pos_label)

        output = A(N_batch).view(-1)
        L_A_fake = criterion(output, neg_label)

        L_A_tot = L_A_real + L_A_fake
        L_A_tot.backward()
        A_opt.step()

        report = {"L_A_real": L_A_real.item(), "L_A_fake": L_A_fake.item()}
        postfix = {key: "{:.4f}".format(value) for key, value in report.items()}
        steps_range.set_postfix(postfix)
    return A

def get_DDPM_generate(scheduler, data_dim, batch_size=64):
    def DDPM_generate(D, G, A, numgen, latent_dim, device, batch_size=batch_size):
        results = []

        for start_idx in range(0, numgen, batch_size):
            end_idx = min(start_idx + batch_size, numgen)
            current_batch_size = end_idx - start_idx

            x = torch.randn(current_batch_size, data_dim).to(device)

            for t in reversed(range(scheduler.num_timesteps)):
                t_embedded = torch.full((current_batch_size, 1), t, device=device).float() / scheduler.num_timesteps
                x_input = torch.cat([x, t_embedded], dim=-1)
                with torch.no_grad():
                    noise_pred = D(x_input)

                beta_t = scheduler.betas[t].to(device)  # Variance (beta_t)
                alpha_t = scheduler.alphas[t].to(device)  # Current alpha_t (not cumulative)
                sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)  # sqrt(alpha_t)
                sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - cumprod(alpha))

                z = torch.randn_like(x) if t > 0 else 0  # Add noise only if t > 0
                x = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * noise_pred) + torch.sqrt(beta_t) * z

            results.append(x.detach().cpu().numpy())

        return np.concatenate(results, axis=0)
    return DDPM_generate


def get_DDPM_generate_cond(scheduler, data_dim, batch_size=64):
    def DDPM_generate_cond(D, G, A, numgen, latent_dim, device, batch_size=batch_size):
        results = []

        for start_idx in range(0, numgen, batch_size):
            end_idx = min(start_idx + batch_size, numgen)
            current_batch_size = end_idx - start_idx

            x = torch.randn(current_batch_size, data_dim).to(device)

            class_label = torch.ones((current_batch_size, 1), device=device)

            for t in reversed(range(scheduler.num_timesteps)):
                t_embedded = torch.full((current_batch_size, 1), t, device=device).float() / scheduler.num_timesteps
                x_input = torch.cat([x, class_label, t_embedded], dim=-1)

                with torch.no_grad():
                    noise_pred = D(x_input)

                beta_t = scheduler.betas[t].to(device)  # Variance (beta_t)
                alpha_t = scheduler.alphas[t].to(device)  # Current alpha_t (not cumulative)
                sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)  # sqrt(alpha_t)
                sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - cumprod(alpha))

                z = torch.randn_like(x) if t > 0 else 0  # Add noise only if t > 0
                x = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * noise_pred) + torch.sqrt(beta_t) * z

            results.append(x.detach().cpu().numpy())

        return np.concatenate(results, axis=0)
    return DDPM_generate_cond

def get_DDPM_generate_guidance(scheduler, data_dim, guidance_scale=1.0, batch_size=64):
    def DDPM_generate_guidance(D, G, A, numgen, latent_dim, device, guidance_scale=guidance_scale, batch_size=batch_size):
        results = []

        for start_idx in range(0, numgen, batch_size):
            end_idx = min(start_idx + batch_size, numgen)
            current_batch_size = end_idx - start_idx

            # Start with pure noise for the current batch
            x = torch.randn(current_batch_size, data_dim).to(device)

            # Reverse diffusion process
            for t in reversed(range(scheduler.num_timesteps)):
                t_embedded = torch.full((current_batch_size, 1), t, device=device).float() / scheduler.num_timesteps
                x_input = torch.cat([x, t_embedded], dim=-1)

                with torch.no_grad():
                    noise_pred = D(x_input)

                beta_t = scheduler.betas[t].to(device)  # Variance (beta_t)
                alpha_t = scheduler.alphas[t].to(device)  # Current alpha_t (not cumulative)
                sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)  # sqrt(alpha_t)
                sqrt_one_minus_alpha_cumprod_t = scheduler.sqrt_one_minus_alpha_cumprod[t].unsqueeze(-1).to(device)  # sqrt(1 - cumprod(alpha))

                # Compute classifier guidance
                x.requires_grad_(True)  # Enable gradient computation for x
                class_prob = A(x).squeeze(-1)  # Binary classifier output P(A=1|x)
                class_grad = torch.autograd.grad(outputs=class_prob.sum(), inputs=x)[0]  # ∇x P(A=1|x)

                # Incorporate classifier guidance
                guided_noise_pred = noise_pred - guidance_scale * class_grad

                # Compute the denoised sample
                z = torch.randn_like(x) if t > 0 else 0  # Add noise only if t > 0
                x = (1 / sqrt_alpha_t) * (x - ((1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * guided_noise_pred) + torch.sqrt(beta_t) * z

            results.append(x.detach().cpu().numpy())

        return np.concatenate(results, axis=0)
    return DDPM_generate_guidance


def VAE_generate(D, G, A, numgen, latent_dim, device):
    z = torch.randn(numgen, latent_dim).to(device)
    generated_data = G(z).detach().cpu().numpy()
    return generated_data

def VAE_generate_cond(D, G, A, numgen, latent_dim, device):
    z = torch.randn(numgen, latent_dim).to(device)
    labels = torch.ones(numgen, 1).to(device)
    z = torch.cat([z, labels], dim=1)
    generated_data = G(z).detach().cpu().numpy()
    return generated_data
    

def GAN_generate(D, G, A, numgen, noise_dim, device):
    noise = torch.randn(numgen, noise_dim).to(device)
    generated_data = G(noise).detach().cpu().numpy()
    return generated_data

def GAN_generate_cond(D, G, A, numgen, noise_dim, device):
    noise = torch.randn(numgen, noise_dim).to(device)
    labels = torch.ones(numgen, 1).to(device)
    noise = torch.cat([noise, labels], dim=1)
    generated_data = G(noise).detach().cpu().numpy()
    return generated_data

def generate_with_rej(generate_fn):
    def wrapper(D, G, A, numgen, latent_dim, device, timeouts=1000):
        all_generated_data = []
        num_valid = 0
        count = 0
        while num_valid < numgen:
            count += 1
            generated_data = generate_fn(D, G, None, numgen, latent_dim, device)
            # Apply rejection criteria using the acceptance function `A`
            pred_logit = A(torch.tensor(generated_data, device=device)).view(-1)
            valid =  torch.sigmoid(pred_logit) > 0.5
            num_valid += valid.sum().item()
            all_generated_data.append(torch.tensor(generated_data, device=device)[valid].cpu())
            if count > timeouts:
                return torch.cat(all_generated_data, dim=0).numpy()
        return torch.cat(all_generated_data, dim=0).numpy()[:numgen]
    return wrapper


def train_model(X, N, Y, C, numgen, numanim, condition, train_params=None, config_params=None, savedir=None):
    batch_size, disc_lr, disc_aux_lr, gen_lr, noise_dim, num_epochs, n_hidden, layer_size= train_params
    aux_setting, validity_weight, diversity_weight = config_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_fn = None

    data_dim = X.shape[1]
    cond_dim = 0

    mode = aux_setting
    if mode.startswith("GAN"):
        generate_fn = GAN_generate
        D_in = data_dim
        D_out = 1
        G_in = noise_dim
        G_out = data_dim
    elif mode.startswith("VAE"):
        generate_fn = VAE_generate
        D_in = data_dim
        D_out = 2*noise_dim
        G_in = noise_dim
        G_out = data_dim
    elif mode.startswith("DDPM"):
        scheduler = NoiseScheduler(1000, device = device)
        generate_fn = get_DDPM_generate(scheduler, data_dim, batch_size=batch_size)
        D_in = data_dim + 1
        D_out = data_dim
        G_in = 1 #unused
        G_out = 1 #unused

    if mode in ["GAN_vanilla"]:
        train_step = GAN_step_vanilla
    elif mode in ["GAN_DO"]:
        train_step = GAN_step_DO
    elif mode in ["GAN_MDD"]:
        train_step = GAN_step_MDD
        D_out = 3
    elif mode in ["GAN_DDD"]:
        train_step = GAN_step_DDD
    elif mode in ["GAN_cond"]:
        generate_fn = GAN_generate_cond
        train_step = GAN_step_cond
        D_in = data_dim + 1
        G_in = noise_dim + 1
        batch_size = batch_size//2
    elif mode in ["GAN_pre"]:
        train_step = GAN_step_clf
        pretrain_fn = pretrain_clf
    elif mode in ["GAN_rej"]:
        generate_fn = generate_with_rej(GAN_generate)
        train_step = GAN_step_vanilla
        pretrain_fn = pretrain_clf
    elif mode in ["VAE_vanilla"]:
        train_step = VAE_step_vanilla
    elif mode in ["VAE_pre"]:
        train_step = VAE_step_clf
        pretrain_fn = pretrain_clf
    elif mode in ["VAE_rej"]:
        generate_fn = generate_with_rej(VAE_generate)
        train_step = VAE_step_vanilla
        pretrain_fn = pretrain_clf
    elif mode in ["VAE_cond"]:
        generate_fn = VAE_generate_cond
        train_step = VAE_step_cond
        D_in = data_dim + 1
        batch_size = batch_size//2
        G_in = noise_dim + 1
    elif mode in ["DDPM_vanilla"]:
        train_step = DDPM_step_wrapper(scheduler)
    elif mode in ["DDPM_cond"]:
        train_step = DDPM_step_cond_wrapper(scheduler)
        generate_fn = get_DDPM_generate_cond(scheduler, data_dim, batch_size=batch_size)
        D_in = data_dim + 2
    elif mode in ["DDPM_rej"]:
        generate_fn = generate_with_rej(get_DDPM_generate(scheduler, data_dim, batch_size=batch_size))
        train_step = DDPM_step_wrapper(scheduler)
        pretrain_fn = pretrain_clf
    elif mode in ["DDPM_guid"]:
        generate_fn = get_DDPM_generate_guidance(scheduler, data_dim, validity_weight, batch_size=batch_size)
        train_step = DDPM_step_wrapper(scheduler)
        pretrain_fn = pretrain_clf
    else:
        raise ValueError("Invalid mode")


    D = Down_Model(D_in, D_out, layer_size, n_hidden)
    G = Up_Model(G_in, G_out, layer_size, n_hidden)
    A = Down_Model(data_dim, 1, layer_size, n_hidden)

    
    D.to(device)
    G.to(device)
    A.to(device)
    D_opt = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=(0.5,0.999))
    G_opt = torch.optim.Adam(G.parameters(), lr=gen_lr, betas=(0.5,0.999))
    A_opt = torch.optim.Adam(A.parameters(), lr=disc_aux_lr, betas=(0.5,0.999))

    P = torch.tensor(X).float()
    N = torch.tensor(N).float()

    P_loader = ReusableDataLoader(TensorDataset(P), batch_size)
    N_loader = ReusableDataLoader(TensorDataset(N), batch_size)

    if num_epochs>0:
        num_steps = num_epochs*len(P)//batch_size
    else:
        num_steps = -num_epochs #hacky way to specify fixed number of steps rather than epochs

    if pretrain_fn is not None:
        A = pretrain_fn(A, A_opt, P_loader, N_loader, num_steps, batch_size, device)
    
    train(D, G, A, D_opt, G_opt, A_opt, P_loader, N_loader, num_steps, batch_size, noise_dim, train_step, device, validity_weight, diversity_weight)

    generated_data = generate_fn(D, G, A, numgen, noise_dim, device)
    return [generated_data], [num_epochs]

def train_wrapper(train_params=None, config_params=None):
    def model(X, N, Y=None, C=None, numgen=None, numanim=None, condition=None, savedir=None):
        return train_model(X, N, Y, C, numgen=numgen, numanim=numanim, condition=condition, train_params=train_params, config_params=config_params, savedir=savedir)
    return model