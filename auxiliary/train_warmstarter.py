import numpy as np
from tqdm import tqdm
import os
import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, ".."))
from modules.warm_starter import WarmStarter
from modules.qp_solver import QPSolver
from utils.mpc_utils import generate_random_problem
import torch
from torch.nn import functional as F
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import copy
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=10000)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--m", type=int, default=5)
parser.add_argument("--fixed-PH", action='store_true')
args = parser.parse_args()

max_epochs = 50000
bs = args.batch_size
n = args.n
m = args.m
device = "cuda:0"

torch.manual_seed(42)
q0, b0, P0, H0 = generate_random_problem(1, n, m, device)
P0_np = P0.squeeze(0).cpu().numpy()
H0_np = H0.squeeze(0).cpu().numpy()
P0 = P0.broadcast_to((bs, -1, -1))
H0 = H0.broadcast_to((bs, -1, -1))

warm_starter = WarmStarter(device, n, m, fixed_P=args.fixed_PH, fixed_H=args.fixed_PH)
if not args.fixed_PH:
    oracle_solver = QPSolver(device, n, m)
else:
    oracle_solver = QPSolver(device, n, m, P=P0_np, H=H0_np)
optimizer = torch.optim.Adam(warm_starter.parameters())
losses = []
Path("runs").mkdir(parents=True, exist_ok=True)
writer = SummaryWriter('runs/' + "warmstarter" + datetime.now().strftime("_%y-%m-%d-%H-%M-%S"))


try:
    def restore_checkpoint():
        global loss_best, no_improvement_count
        warm_starter.load_state_dict(checkpoint[0])
        optimizer.load_state_dict(checkpoint[1])
        loss_best = 0
        no_improvement_count= 0
    loss_best = 0
    no_improvement_count = 0
    for i_ep in (pbar:= tqdm(range(max_epochs))):
        # Check for early stopping
        if i_ep > 0:
            if loss_best == 0:
                loss_best = losses[-1] + 1
            if losses[-1] < loss_best:
                no_improvement_count = 0
                loss_best = 0.95 * loss_best + 0.05 * losses[-1]
                checkpoint = [
                    copy.deepcopy(warm_starter.state_dict()),
                    copy.deepcopy(optimizer.state_dict()),
                ]
            else:
                no_improvement_count += 1
            if no_improvement_count >= 5:
                restore_checkpoint()
                optimizer.param_groups[0]['lr'] /= 10
                loss_best = 0
                if optimizer.param_groups[0]['lr'] < 1e-7:
                    break

        optimizer.zero_grad()
        q, b, P, H = generate_random_problem(bs, n, m, device)
        if not args.fixed_PH:
            oracle_Xb = oracle_solver(q, b, P, H)[0][:, -1, :]
            approx_X = warm_starter(q, b, P, H)
        else:
            oracle_Xb = oracle_solver(q, b)[0][:, -1, :]
            approx_X = warm_starter(q, b)
        loss = torch.log((approx_X - oracle_Xb).norm(dim=-1)).mean()
        if loss.isfinite():
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        else:
            restore_checkpoint()
        pbar.set_description(f"{optimizer.param_groups[0]['lr']:.2e}, {loss.item():.2f}/{loss_best:.2f}/{no_improvement_count}")
        writer.add_scalar("stat/loss", loss.item(), i_ep)
        writer.add_scalar("stat/lr", optimizer.param_groups[0]['lr'], i_ep)

except:
    traceback.print_exc()
finally:
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(warm_starter.state_dict(), f"models/warmstarter-{n}-{m}.pth")
