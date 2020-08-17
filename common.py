import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
from networkx import (
    DiGraph, find_cycle, topological_sort, NetworkXError, NetworkXNoCycle, relabel_nodes
)
from cdt.data import load_dataset
import pandas as pd
import os
from tqdm import tqdm, trange
import platform


def matrix2list(mat):
    return [list(arr) for arr in mat]


def plot_conn_mat(C, title=""):
    fig, ax = plt.subplots(figsize=(9, 9))
    if isinstance(C, torch.Tensor):
        C = C.cpu().numpy()
    im = ax.imshow(C)
    ax.figure.colorbar(im)
    ax.set_xticks(np.arange(C.shape[0]))
    ax.set_yticks(np.arange(C.shape[0]))
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(C)):
        for j in range(len(C)):
            text = ax.text(j, i, "%.3f" % C[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    return fig


def sorted_digraph(graph: DiGraph):
    # make deterministic:
    import random
    random.seed(42)
    np.random.seed(42)
    graph = graph.copy()
    # remove cycles.
    while True:
        try:
            cycle = find_cycle(graph, orientation=None)
            last_edge = cycle[-1]  # type: (int, int)
            graph.remove_edge(*last_edge)
        except (NetworkXError, NetworkXNoCycle) as ex:
            break

    return list(topological_sort(graph))


def levenshtein_distance(l1: list, l2: list):
    def lev(i, j):
        if min(i, j) <= 0:
            return max(i, j)
        x1 = lev(i - 1, j) + 1
        x2 = lev(i, j - 1) + 1
        x3 = lev(i - 1, j - 1) + int(l1[i] != l2[j])
        return min(x1, x2, x3)

    return lev(len(l1)-1, len(l2)-1)


def leaky_relu(x):
    return np.maximum(x, 0) + 0.01 * np.minimum(x, 0)


SYNTHETIC_N_SAMPLES = 5000


def get_synthetic():
    with torch.no_grad():
        x1 = torch.randn(SYNTHETIC_N_SAMPLES)
        x2 = leaky_relu(x1 * 8 + 3 + torch.randn_like(x1))
        x3 = leaky_relu(x1 * -1 + x2 * 0.1 + 8 + torch.randn_like(x1)*0.1)
        x4 = leaky_relu(x1 * 0.2 + x3*0.01 - 5 + torch.randn_like(x1)*0.1)
        x5 = leaky_relu(x2 * 0.5 + x4 * -0.3+ torch.randn_like(x1)*0.1)
        x6 = leaky_relu(x5 * 0.2 + 3+ torch.randn_like(x1)*0.1)
        x7 = leaky_relu(x1 * -7 + 9 + x6 * 4 + torch.randn_like(x1)*0.1)
        x8 = leaky_relu(x4 * -7 + 9 + x7 * 4 + torch.randn_like(x1)*0.1)
        x = torch.stack([x1, x2, x3, x4, x5, x6, x7, x8], dim=-1)
    data = x.numpy()
    data = pd.DataFrame(data=data, columns=[str(i+1) for i in range(x.shape[1])])
    return data, list(range(x.shape[1]))


def get_data(name):
    if name == 'sachs':
        data, graph = load_dataset('sachs')  # type: pd.DataFrame, DiGraph
        relabel_nodes(graph, {col: i for i, col in enumerate(data.columns)})
        caus_ordering = sorted_digraph(graph)
        return data, caus_ordering
    if name == 'synthetic':
        return get_synthetic()
    raise KeyError


def normalize_data(X: torch.Tensor):
    return (X - X.min(dim=0)[0]) / (X.max(dim=0)[0] - X.min(dim=0)[0])


def get_device(model: nn.Module):
    try:
        return next(model.parameters()).device
    except:
        return "cpu"


def linregress(A: torch.Tensor, b: torch.Tensor):
    with torch.no_grad():
        return torch.pinverse(A) @ b


PRINT_EVERY = 100


class RangeLogger:
    def __init__(self, *args, **kwargs):
        if platform.system() == 'Windows':
            self.range = trange(*args, **kwargs)
            self.is_trange = True
        else:
            self.range = range(*args, **kwargs)
            self.is_trange = False
        self.description = ""

    def set_description(self, desc):
        if isinstance(self.range, tqdm):
            self.range.set_description(desc)
        else:
            self.description = desc

    def __iter__(self):
        if self.is_trange:
            yield from self.range
        else:
            for x in self.range:
                if x % PRINT_EVERY == 0:
                    print(f'[{x} / {len(self.range)}] : {self.description}')
                yield x


def xnor(x1, x2):
    return torch.abs(1 - x1 - x2 - x1*x2)

