import os
import numpy as np
import faiss
import torch


def search_index_pytorch(index, x, k, D=None, I=None):
    """Search a FAISS index with a PyTorch tensor; returns PyTorch tensors.
    Compatible with FAISS >= 1.7.0 (numpy round-trip instead of SWIG pointers).
    """
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    x_np = x.cpu().numpy().astype('float32')
    D_np, I_np = index.search(x_np, k)

    D_out = torch.from_numpy(D_np).to(x.device)
    I_out = torch.from_numpy(I_np.astype('int64')).to(x.device)
    return D_out, I_out


def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    """Brute-force k-NN search on GPU; returns PyTorch tensors.
    Compatible with FAISS >= 1.7.0 (uses faiss.knn_gpu instead of
    the removed bruteForceKnn / cast_integer_to_long_ptr APIs).
    """
    assert xb.device == xq.device

    xq_np = xq.cpu().numpy().astype('float32')
    xb_np = xb.cpu().numpy().astype('float32')

    D_np, I_np = faiss.knn_gpu(res, xq_np, xb_np, k, metric=metric)

    D_out = torch.from_numpy(D_np).to(xb.device)
    I_out = torch.from_numpy(I_np.astype('int64')).to(xb.device)
    return D_out, I_out


def index_init_gpu(ngpus, feat_dim):
    flat_config = []
    for i in range(ngpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    res = [faiss.StandardGpuResources() for i in range(ngpus)]
    indexes = [faiss.GpuIndexFlatL2(res[i], feat_dim, flat_config[i]) for i in range(ngpus)]
    index = faiss.IndexShards(feat_dim)
    for sub_index in indexes:
        index.add_shard(sub_index)
    index.reset()
    return index


def index_init_cpu(feat_dim):
    return faiss.IndexFlatL2(feat_dim)