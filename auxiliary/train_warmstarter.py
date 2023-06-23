import setup_problem
import numpy as np
from tqdm import tqdm
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))
from modules.warm_starter import WarmStarter
from modules.qp_solver import QPSolver
from utils.utils import generate_random_problem
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=800)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--m", type=int, default=10)
args = parser.parse_args()

max_epochs = 50000
bs = args.batch_size
n = args.n
m = args.m
device = "cuda:0"
warm_starter = WarmStarter(device, n, m, fixed_P=False, fixed_H=False)
oracle_solver = QPSolver(device, n, m)
optimizer = torch.optim.Adam(warm_starter.parameters())
losses = []
Path("runs").mkdir(parents=True, exist_ok=True)
writer = SummaryWriter('runs/' + "warmstarter" + datetime.now().strftime("_%d-%H-%M-%S"))

try:
    loss_best = 0
    no_improvement_count = 0
    for i_ep in (pbar:= tqdm(range(max_epochs))):
        # Check for early stopping
        if i_ep > 0:
            if loss_best == 0:
                loss_best = losses[-1]
            if losses[-1] < loss_best:
                no_improvement_count = 0
                loss_best = 0.95 * loss_best + 0.05 * losses[-1]
            else:
                no_improvement_count += 1
            if no_improvement_count >= 5:
                optimizer.param_groups[0]['lr'] /= 10
                loss_best += 1
                if optimizer.param_groups[0]['lr'] < 1e-7:
                    break

        optimizer.zero_grad()
        P, H, q, b = generate_random_problem(bs, n, m, device)
        oracle_Xb, oracle_sol_b = oracle_solver(q, b, P, H)
        approx_X = warm_starter(qb, bb)
        loss = torch.log((approx_X - oracle_Xb).norm(dim=-1)).mean()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"{optimizer.param_groups[0]['lr']:.2e}, {loss.item():.2f}/{loss_best:.2f}/{no_improvement_count}")
        writer.add_scalar("losses/loss", loss.item())
        losses.append(loss.item())

except:
    traceback.print_exc()
finally:
    Path("models").mkdir(parents=True, exist_ok=True)
    torch.save(warm_starter.state_dict(), f"models/warmstarter-{n}-{m}.pth")
