import os
import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, ".."))
import torch
import torch.nn as nn
import csv
import time
from modules.qp_unrolled_network import QPUnrolledNetwork

batch_size = 10000
input_size = 100
device = "cuda:0"

def mlp_builder(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_size),
    ).to(device)

model_shared = QPUnrolledNetwork(
    device=device,
    input_size=input_size,
    n_qp=10,
    m_qp=5,
    qp_iter=10,
    mlp_builder=mlp_builder,
    shared_PH=True,
)

model_not_shared = QPUnrolledNetwork(
    device=device,
    input_size=input_size,
    n_qp=10,
    m_qp=5,
    qp_iter=10,
    mlp_builder=mlp_builder,
    shared_PH=False,
)

def write_csv(prof, filename):

    # Extract key averages
    averages = prof.key_averages()

    # Export to CSV
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'Name', 'Self CPU total', 'CPU total', 'CPU time avg', 
            'Self CUDA total', 'CUDA total', 'CUDA time avg', 'Number of Calls'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for avg in averages:
            writer.writerow({
                'Name': avg.key,
                'Self CPU total': avg.self_cpu_time_total,
                'CPU total': avg.cpu_time_total,
                'Self CUDA total': avg.self_cuda_time_total,
                'CUDA total': avg.cuda_time_total,
                'Number of Calls': avg.count
            })


def profile(model, tag):

    outputs = []

    t = time.time()
    with torch.autograd.profiler.profile(use_cuda=True) as forward_prof:
        for i in range(10):
            x = torch.randn((batch_size, input_size), device=device)
            outputs.append(model(x))
    print(f"Forward Pass Profiling {tag}:", time.time() - t)
    write_csv(forward_prof, f"forward_prof_{tag}.csv")

    t = time.time()
    with torch.autograd.profiler.profile(use_cuda=True) as backward_prof:
        loss = sum(outputs).mean()
        loss.backward()
    print(f"Backward Pass Profiling {tag}:", time.time() - t)
    write_csv(backward_prof, f"backward_prof_{tag}.csv")

profile(model_shared, "shared")
profile(model_not_shared, "not_shared")
