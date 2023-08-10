"""Test performance of Ab, where A is a single matrix, and b is a batch of vectors."""

import torch
import time

def bmv1(A, b):
    return (A.unsqueeze(0) @ b.unsqueeze(-1)).squeeze(-1)

def bmv2(A, b):
    return (A @ b.t()).t()

batch_size = 100000
n = 100
device = "cuda:0"

def benchmark(f):
    for i in range(1000):
        A = torch.randn((n, n), device=device, requires_grad=True)
        b = torch.randn((batch_size, n), device=device, requires_grad=True)
        loss = f(A, b).sum()
        loss.backward()

t = time.time(); benchmark(bmv1); print(time.time() - t)
t = time.time(); benchmark(bmv2); print(time.time() - t)
    