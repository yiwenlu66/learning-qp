import torch
from torch import nn
from modules.qp_solver import QPSolver
from utils.utils import make_psd

class QPUnrolledNetwork(nn.Module):
    """
    Learn a QP problem from the input using a MLP, then solve the QP using fixed number of unrolled PDHG iterations.

    Form of QP:
    minimize    (1/2)x'Px + q'x
    subject to  Hx + b >= 0,
    where x in R^n, b in R^m.
    """
    def __init__(self, device, input_size, n_qp, m_qp, qp_iter, mlp_builder):
        """mlp_builder is a function mapping (input_size, output_size) to a nn.Sequential object."""

        super().__init__()

        # TODO: try shared P, H
        self.device = device
        self.input_size = input_size
        self.n_qp = n_qp
        self.m_qp = m_qp
        self.qp_iter = qp_iter

        self.n_P_param = n_qp * (n_qp + 1) // 2
        self.n_q_param = n_qp
        self.n_H_param = m_qp * n_qp
        self.n_b_param = m_qp
        self.n_qp_param = self.n_P_param + self.n_q_param + self.n_H_param + self.n_b_param

        self.mlp = mlp_builder(input_size, self.n_qp_param)
        # TODO: add warmstarter and preconditioner
        self.solver = QPSolver(device, n_qp, m_qp)

    def forward(self, x):
        qp_params = self.mlp(x).float()

        start = 0
        end = self.n_P_param
        P_params = qp_params[:, start:end]
        start = end
        end = start + self.n_q_param
        q = qp_params[:, start:end]
        start = end
        end = start + self.n_H_param
        H_params = qp_params[:, start:end]
        start = end
        end = start + self.n_b_param
        b = qp_params[:, start:end]

        # Reshape P, H vectors into matrices
        P = make_psd(P_params, min_eig=1e-4)
        H = H_params.view(-1, self.m_qp, self.n_qp)
        return self.solver(q, b, P, H, iters=self.qp_iter)[1][:, -1, :]
