import torch, os
from .utils import load_pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from torch import sparse
from typing import List, Optional


def get_adj_mx(path:str, graph_type):
    _,_,adj_mx = load_pickle(os.path.join(path,'adj_mx.pkl'))
    if graph_type == 'scalap':
        return sparse_scipy2torch(scaled_laplacian(adj_mx))
    elif graph_type =='origin':
        return torch.from_numpy(adj_mx).float()
    elif graph_type == 'alldistance':
        adj_mx = load_pickle(os.path.join(path,'graph_sensor_locations.pkl'))
        return torch.from_numpy(adj_mx/1000.0).float()                             #以km为单位


def sparse_scipy2torch(w: sp.coo_matrix):
    """
    build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    :return:
    """
    shape = w.shape
    i = torch.tensor(np.vstack((w.row, w.col)).astype(int)).long()
    v = torch.tensor(w.data).float()
    return sparse.FloatTensor(i, v, torch.Size(shape))


def normalized_laplacian(w: np.ndarray) -> sp.coo_matrix:
    w = sp.coo_matrix(w)
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return sp.eye(w.shape[0]) - w.dot(d_mat_inv_sqrt).T.dot(d_mat_inv_sqrt).tocoo()

def random_walk_matrix(w) -> sp.coo_matrix:
    w = sp.coo_matrix(w)
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv.dot(w).tocoo()

def scaled_laplacian(w: np.ndarray, lambda_max: Optional[float] = 2., undirected: bool = True) -> sp.coo_matrix:
    if undirected:
        w = np.maximum.reduce([w, w.T])
    lp = normalized_laplacian(w)
    if lambda_max is None:
        lambda_max, _ = eigsh(lp.todense(), 1, which='LM')
        lambda_max = lambda_max[0]
    lp = sp.csr_matrix(lp)
    m, _ = lp.shape
    i = sp.identity(m, format='csr', dtype=lp.dtype)
    lp = (2 / lambda_max * lp) - i
    return lp.astype(np.float32).tocoo()

