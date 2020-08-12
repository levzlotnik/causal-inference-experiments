import os
from datetime import datetime
from scipy.stats import spearmanr, kendalltau
import numpy as np
from common import get_data, normalize_data, get_device, linregress
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
from itertools import product


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

    def forward(self, N=None):
        e = torch.rand(N or self.batch_size, self.noise_dim, device=get_device(self.generator))
        return self.generator(e)


class GLANN(nn.Module):
    def __init__(self, **kwargs):
        super(GLANN, self).__init__()
        glo_kwargs = kwargs.get('glo_kwargs', {})
        translator_kwargs = kwargs.get('translator_kwargs', {})

        self.glo = GLO(**glo_kwargs)
        self.translator = NoiseCodeTranslator(**translator_kwargs)

    def forward(self, N=None):
        return self.glo.generator(self.translator.forward(N))


class SquaredCorrelation(nn.Module):
    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        dX, dY = X - X.mean(), Y - Y.mean()
        return (torch.sum(dX * dY)**2) / (torch.sum(dX**2) * torch.sum(dY**2))


class SquaredConditionalCorrelation(nn.Module):
    def __init__(self):
        super(SquaredConditionalCorrelation, self).__init__()
        self.squared_corr = SquaredCorrelation()

    def forward(self, X, Y, Z):
        with torch.no_grad():
            # Get linear regression coefficients
            Z_des = torch.stack([Z, torch.ones_like(Z)], dim=-1)
            a_x, b_x = linregress(Z_des, X)
            a_y, b_y = linregress(Z_des, Y)
        X_pred = Z * a_x + b_x
        Y_pred = Z * a_y + b_y
        res_X, res_Y = X - X_pred, Y - Y_pred
        return self.squared_corr.forward(res_X, res_Y)


class ConnectivityMatrix(nn.Module):
    def __init__(self, size):
        super(ConnectivityMatrix, self).__init__()
        self.size = size
        self.logits_matrix = nn.Parameter(torch.zeros(size, size))
        self.register_buffer('mask', 1 - torch.eye(size))

    def forward(self):
        C_logits = torch.stack([self.logits_matrix, self.logits_matrix.t()], dim=0)
        C = C_logits.softmax(dim=0)[0] * self.mask
        return C


def loss_transitivity(C: torch.Tensor):
    r"""
    Calculates transitivity loss:
      $
        L_{trans} = \sum_{ i \ne j, j \ne k, i \ne k } C_{ij} C_{jk} | 1 - C_{ik} |
      $
    We simplify it to:
      $
        L_{trans} = \sum_{i, j} (C^2 \odot C^T)_{ij}
      $
    """
    C2 = C @ C
    return torch.sum(C2 * C.t())


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


def loss_indep(X: torch.Tensor, C: torch.Tensor):
    idxs = list(range(X.shape[1]))
    loss = 0.
    cond_corr2 = SquaredConditionalCorrelation()
    corr2 = SquaredCorrelation()
    for i, j, k in product(idxs, idxs, idxs):
        if len({i, j, k}) < 3:
            continue
        X_i, X_j, X_k = X[:, i], X[:, j], X[:, k]
        scc = cond_corr2.forward(X_i, X_k, X_j)
        sc = corr2.forward(X_i, X_k)
        # Chain Loss
        loss += C[i, j] * C[j, k] * scc
        # Fork Loss
        loss += C[j, i] * C[j, k] * scc
        # Collider Loss
        loss += sc

    return loss


class CausalityChainModel(nn.Module):
    def __init__(self, size, glann: GLANN = None,
                 reconst_weight=1., transitivity_weight=1., independence_weight=1.,
                 **glann_kwargs):
        super(CausalityChainModel, self).__init__()
        if glann:
            self.glann = glann
        else:
            glann_kwargs['glo_kwargs']['generator_kwargs']['output_dim'] = size
            self.glann = GLANN(**glann_kwargs)
        self.conn_mat = ConnectivityMatrix(size)
        self.reconst_weight = reconst_weight
        self.transitivity_weight = transitivity_weight
        self.independence_weight = independence_weight

    def forward(self, X, N_samples=None):
        # Transitivity loss - defined in paper.
        loss_trans = loss_transitivity(self.conn_mat.forward())
        # MSE loss for data construction:
        X_pred = self.glann.glo.forward()
        loss_construction, _ = loss_mse(X_pred, X)
        # Translator Loss for generative process.
        Z = self.glann.glo.z_logits.data.clone().detach()
        Z_pred = self.glann.translator.forward()
        loss_translator, _ = loss_nct(Z_pred, Z)
        # Independence loss computation:
        X_pred = self.glann.forward(N_samples)
        loss_independence = loss_indep(X_pred, self.conn_mat.forward())
        loss = loss_trans * self.transitivity_weight + \
               (loss_construction + loss_translator) * self.reconst_weight + \
               loss_independence * self.independence_weight
        loss_terms_dict = {
            'loss_trans': loss_trans,
            'loss_translator': loss_translator,
            'loss_construction': loss_construction,
            'loss_independence': loss_independence
        }
        return loss, loss_terms_dict

    @property
    def C(self):
        with torch.no_grad():
            return self.conn_mat.forward()


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


def train_causality_chain(X: torch.Tensor, model: CausalityChainModel, name: str,
                          N_samples=15000,
                          epochs=3000, alpha=1e-2, writer: SummaryWriter = None):
    # noinspection PyUnusedLocal
    optimizer = Adam(model.parameters(), lr=alpha)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    model = model.to(X.device)
    end_loss = float('inf')
    t = trange(epochs)
    for i in t:
        optimizer.zero_grad()
        loss, loss_terms_dict = model.forward(X, N_samples)
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


def train_glann_causality_chain(X: torch.Tensor, model: GLANN,
                                stage_epochs=(10000, 20000, 30000),
                                weights_dict=None,
                                writer: SummaryWriter = None):
    """
    According to https://arxiv.org/pdf/1812.08985.pdf section 3.
    """
    # First Stage: Optimize GLO by minimizing perceptual loss with X:

    if weights_dict is None:
        weights_dict = {}
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
        model.translator.batch_size = b_size
    # Stage 3: we add the transitivity loss and independence loss
    weights_dict = weights_dict or {}
    causal_chain_glann = CausalityChainModel(X.shape[1], model, **weights_dict)
    train_causality_chain(X, causal_chain_glann,
                          name="stage3(causality_chain)",
                          N_samples=3000,
                          epochs=stage_epochs[2], writer=writer)


DEVICE = 'cuda:0'
GLO_GENERATOR_KWARGS = {
    'input_dim': 4,
    'hidden_dims': [64, 32, 24, 16],
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
    'batch_size': 100,
    'noise_dim': 2,
    'generator_kwargs': TRANSLATOR_GENERATOR_KWARGS
}
GLANN_KWARGS = {
    'glo_kwargs': GLO_KWARGS,
    'translator_kwargs': TRANSLATOR_KWARGS
}
WEIGHTS_DICT = {
    'reconst_weight': 1.,
    'transitivity_weight': 1e-4,
    'independence_weight': 1e-2
}

DATASET = 'synthetic'


def create_writer():
    EXPERIMENT_LOGDIR = f'logs_{os.path.basename(__file__)[:-3]}/{DATASET}.{datetime.now():%Y-%m-%d.%H-%M}'
    print(f"tensorboard: run 'tensorboard --logdir={'/'.join(EXPERIMENT_LOGDIR.split('/')[:-1])}'")
    return SummaryWriter(log_dir=EXPERIMENT_LOGDIR)


def main():
    data, actual_caus_ord = get_data(DATASET)  # type: pd.DataFrame, list
    X = torch.from_numpy(data.to_numpy()).to(device=DEVICE, dtype=torch.float)
    X = normalize_data(X)
    glann_kwargs = deepcopy(GLANN_KWARGS)
    glann_kwargs['glo_kwargs']['n_samples'] = X.size(0)
    glann_kwargs['glo_kwargs']['generator_kwargs']['output_dim'] = X.size(1)
    model = GLANN(**glann_kwargs)
    writer = create_writer()
    train_glann_causality_chain(X, model, weights_dict=WEIGHTS_DICT, writer=writer)


if __name__ == '__main__':
    main()
