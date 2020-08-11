import os
from datetime import datetime
from scipy.stats import spearmanr, kendalltau
import numpy as np
from common import get_data, normalize_data, get_device, SYNTHETIC_N_SAMPLES
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from cdt.data import load_dataset
from collections import deque, defaultdict
from itertools import chain
from networkx import DiGraph, relabel_nodes
import pandas as pd
from copy import deepcopy


class LinearGenerator(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super(LinearGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, dim, bias=False), nn.BatchNorm1d(dim), nn.LeakyReLU()])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        return self.layers(z)


class GLO(nn.Module):
    def __init__(self, n_samples, latent_embed_dim, generator_kwargs=None, generator=None):
        super(GLO, self).__init__()
        self.latent_embed_dim = latent_embed_dim
        self.n_samples = n_samples
        self.z_logits = nn.Parameter(torch.randn(n_samples, latent_embed_dim))
        self.generator = generator or LinearGenerator(**generator_kwargs)

    def forward(self):
        # z = torch.softmax(self.z_logits, dim=-1)
        return self.generator(self.z_logits)


class NoiseCodeTranslator(nn.Module):
    def __init__(self, batch_size, noise_dim, generator_kwargs=None, generator=None):
        super(NoiseCodeTranslator, self).__init__()
        self.noise_dim = noise_dim
        self.generator = generator or LinearGenerator(**generator_kwargs)
        self.batch_size = batch_size

    def forward(self):
        e = torch.rand(self.batch_size, self.noise_dim, device=get_device(self.generator))
        return self.generator(e)


class GLANN(nn.Module):
    def __init__(self, **kwargs):
        super(GLANN, self).__init__()
        glo_kwargs = kwargs.get('glo_kwargs', {})
        translator_kwargs = kwargs.get('translator_kwargs', {})

        self.glo = GLO(**glo_kwargs)
        self.translator = NoiseCodeTranslator(**translator_kwargs)

    def forward(self):
        return self.glo.generator(self.translator.forward())


def loss_nct(Z_pred: torch.Tensor, Z: torch.Tensor):
    sample_batch_size = Z_pred.shape[0] * 10
    perm = torch.randperm(Z.shape[0])
    idx = perm[:sample_batch_size]
    Z_samples = Z[idx]
    Z_pred = Z_pred.unsqueeze(1)
    Z_samples = Z_samples.unsqueeze(0)
    L2 = torch.mean((Z_pred - Z_samples)**2, dim=-1)  # type: torch.Tensor
    # L2 ~ (Z_pred, Z_samples).mse()
    min_L2, idxs = torch.min(L2, dim=1)
    return torch.mean(min_L2), None


def loss_mse(X_pred, X_true):
    return nn.MSELoss()(X_pred, X_true), None


def train_glo(X: torch.Tensor, model: GLO, criterion, name: str,
              epochs=3000, alpha=1e-2, writer: SummaryWriter = None):
    optimizer = Adam(model.parameters(), lr=alpha)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    model = model.to(X.device)
    end_loss = float('inf')
    t = trange(epochs)
    for i in t:
        optimizer.zero_grad()
        X_pred = model()
        loss, loss_terms_dict = criterion(X_pred, X)
        t.set_description(f'Training: [{name}] \t loss={loss.item():.3e}')
        loss.backward()
        end_loss = loss.item()
        optimizer.step()
        lr_scheduler.step(loss)
        if writer:
            writer.add_scalar(f'{name}/Loss', loss.item(), i)
            loss_terms_dict = loss_terms_dict or {}
            for k, v in loss_terms_dict.items():
                writer.add_scalar(f'{name}/{k}', v.item(), i)
    return end_loss


def train_nct(Z: torch.Tensor, model: NoiseCodeTranslator, name,
              epochs=3000, alpha=1e-1, writer: SummaryWriter = None):
    optimizer = Adam(model.parameters(), lr=alpha)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    model = model.to(Z.device)
    end_loss = float('inf')
    t = trange(epochs)
    for i in t:
        optimizer.zero_grad()
        Z_pred = model.forward()
        loss, loss_terms_dict = loss_nct(Z_pred, Z)
        t.set_description(f'Training: [{name}] \t loss={loss.item():.3e}')
        loss.backward()
        end_loss = loss.item()
        optimizer.step()
        lr_scheduler.step(loss)
        if writer:
            writer.add_scalar(f'{name}/Loss', loss.item(), i)
            loss_terms_dict = loss_terms_dict or {}
            for k, v in loss_terms_dict.items():
                writer.add_scalar(f'{name}/{k}', v.item(), i)
    return end_loss


def train_glann(X: torch.Tensor, model: GLANN, perceptual_loss_criterion,
                stage_epochs=(10000, 20000, 3000),
                writer: SummaryWriter = None):
    """
    According to https://arxiv.org/pdf/1812.08985.pdf section 3.
    """
    # First Stage: Optimize GLO by minimizing perceptual loss with X:

    loss_stage1 = train_glo(X, model.glo, loss_mse,
                            name="stage1(glo)", epochs=stage_epochs[0], writer=writer)
    Z = model.glo.z_logits.data.clone().detach()
    loss_stage2 = train_nct(Z, model.translator,
                            name="stage2(translator)", epochs=stage_epochs[1], writer=writer)
    # We test histograms for the real data vs generative model.
    if writer:
        b_size = model.translator.batch_size
        model.translator.batch_size = X.shape[0] * 2
        X_generated = model()
        for i in range(X.shape[1]):
            col_real = X[:, i]
            col_generated = X_generated[:, i]
            writer.add_histogram(f'Histograms[:, {i}]/X_real', col_real, stage_epochs[1])
            writer.add_histogram(f'Histograms[:, {i}]/X_generated', col_generated, stage_epochs[1])
    # Stage 3: we add the transitivity loss and independence loss


DEVICE = 'cuda:0'
GLO_GENERATOR_KWARGS = {
    'input_dim': 4,
    'hidden_dims': [64, 32, 24, 16],
    'output_dim': 11
}
TRANSLATOR_GENERATOR_KWARGS = {
    'input_dim': 2,
    'hidden_dims': [48, 32, 24, 16, 8],
    'output_dim': 4
}
GLO_KWARGS = {
    'latent_embed_dim': 4,
    'generator_kwargs': GLO_GENERATOR_KWARGS
}
TRANSLATOR_KWARGS = {
    'batch_size': 200,
    'noise_dim': 2,
    'generator_kwargs': TRANSLATOR_GENERATOR_KWARGS
}
GLANN_KWARGS = {
    'glo_kwargs': GLO_KWARGS,
    'translator_kwargs': TRANSLATOR_KWARGS
}

DATASET = 'sachs'


def create_writer():
    EXPERIMENT_LOGDIR = f'logs_{os.path.basename(__file__)[:-3]}/{DATASET}.{datetime.now():%Y-%m-%d.%H-%M}'
    print(f"tensorboard: run 'tensorboard --logdir={'/'.join(EXPERIMENT_LOGDIR.split('/')[:-1])}'")
    return SummaryWriter(log_dir=EXPERIMENT_LOGDIR)


def main():
    data, actual_caus_ord = get_data(DATASET)  # type: pd.DataFrame, list
    # data = data.drop(columns=data.columns[2])
    X = torch.from_numpy(data.to_numpy()).to(device=DEVICE, dtype=torch.float)
    X = normalize_data(X)
    glann_kwargs = deepcopy(GLANN_KWARGS)
    glann_kwargs['glo_kwargs']['n_samples'] = X.size(0)
    model = GLANN(**glann_kwargs)
    writer = create_writer()
    train_glann(X, model, None, writer=writer)


if __name__ == '__main__':
    main()
