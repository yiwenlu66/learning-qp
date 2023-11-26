import torch
from torch import nn
import torch.nn.functional as F
from rl_games.algos_torch.network_builder import NetworkBuilder, A2CBuilder
from ..modules.qp_unrolled_network import QPUnrolledNetwork
import atexit
from datetime import datetime
import os
import numpy as np

class A2CQPUnrolled(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        self.actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size', 1)

        NetworkBuilder.BaseNetwork.__init__(self)
        self.n_obs = input_shape[0]
        self.load(params)

        if self.separate:
            raise NotImplementedError()

        def mlp_builder(input_size, output_size):
            policy_mlp_args = {
                'input_size' : input_size,
                'units' : self.params["mlp"]["units"] + [output_size],
                'activation' : self.activation,
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            return self._build_mlp(**policy_mlp_args)

        self.policy_net = QPUnrolledNetwork(
            self.device,
            self.n_obs,
            self.n_qp,
            self.m_qp,
            self.qp_iter,
            mlp_builder,
            shared_PH=self.shared_PH,
            affine_qb=self.affine_qb,
            strict_affine_layer=self.strict_affine_layer,
            obs_has_half_ref=self.obs_has_half_ref,
            symmetric=self.symmetric,
            no_b=self.no_b,
            use_warm_starter=self.use_warm_starter,
            train_warm_starter=self.train_warm_starter,
            ws_loss_coef=self.ws_loss_coef,
            ws_update_rate=self.ws_update_rate,
            mpc_baseline=self.mpc_baseline,
            use_osqp_for_mpc=self.use_osqp_for_mpc,
            use_residual_loss=self.use_residual_loss,
            imitate_mpc=self.imitate_mpc,
            force_feasible=self.force_feasible,
            feasible_lambda=self.feasible_lambda,
            is_test=self.is_test,
        )

        # TODO: exploit structure in value function?
        value_mlp_args = {
            'input_size' : self.n_obs,
            'units' : self.params["mlp"]["units"] + [self.value_size],
            'activation' : self.activation,
            'norm_func_name' : self.normalization,
            'dense_func' : torch.nn.Linear,
            'd2rl' : self.is_d2rl,
            'norm_only_first_layer' : self.norm_only_first_layer
        }
        self.value_net = self._build_mlp(**value_mlp_args)

        sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
        self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        mlp_init = self.init_factory.create(**self.initializer)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        sigma_init(self.sigma)

        # Register the cleanup method to be called at exit
        atexit.register(self.cleanup)

    def cleanup(self):
        # Implement the housekeeping logic here
        # For example, dumping internal state to a file
        directory = 'test_results'
        if not os.path.exists(directory):
            os.makedirs(directory)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.mpc_baseline is not None and self.use_osqp_for_mpc:
            # When MPC is run using OSQP, dump the iteration counts (collected by QPUnrolledNetwork) to CSV
            tag = f"{self.run_name}_mpc_iter_count"
            filename = os.path.join(directory, f"{tag}_{timestamp}.csv")
            iter_counts = self.policy_net.info['osqp_iter_counts']
            np.savetxt(filename, iter_counts, fmt='%d')
        if self.mpc_baseline is not None and self.mpc_baseline.get("robust_method", None) is not None:
            # When robust MPC is used, dump the per-step times (collected by QPUnrolledNetwork) to CSV
            tag = f"{self.run_name}_running_time"
            filename = os.path.join(directory, f"{tag}_{timestamp}.csv")
            running_time = self.policy_net.info['running_time']
            np.savetxt(filename, running_time, fmt='%f')

    def forward(self, obs_dict):
        obs = obs_dict['obs']
        info = obs_dict.get('info', {})
        mu = self.policy_net(obs, info=info)[:, :self.actions_num]
        value = self.value_net(obs)
        sigma = self.sigma
        states = None   # reserved for RNN
        if self.policy_net.autonomous_losses:
            return mu, mu*0 + sigma, value, states, self.policy_net.autonomous_losses
        else:
            return mu, mu*0 + sigma, value, states

    def load(self, params):
        A2CBuilder.Network.load(self, params)
        self.params = params
        self.device = params["custom"]["device"]
        self.n_qp = params["custom"]["n_qp"]
        self.m_qp = params["custom"]["m_qp"]
        self.qp_iter = params["custom"]["qp_iter"]
        self.shared_PH = params["custom"]["shared_PH"]
        self.affine_qb = params["custom"]["affine_qb"]
        self.strict_affine_layer = params["custom"]["strict_affine_layer"]
        self.obs_has_half_ref = params["custom"]["obs_has_half_ref"]
        self.symmetric = params["custom"]["symmetric"]
        self.no_b = params["custom"]["no_b"]
        self.use_warm_starter = params["custom"]["use_warm_starter"]
        self.train_warm_starter = params["custom"]["train_warm_starter"]
        self.ws_loss_coef = params["custom"]["ws_loss_coef"]
        self.ws_update_rate = params["custom"]["ws_update_rate"]
        self.mpc_baseline = params["custom"]["mpc_baseline"]
        self.use_osqp_for_mpc = params["custom"]["use_osqp_for_mpc"]
        self.use_residual_loss = params["custom"]["use_residual_loss"]
        self.imitate_mpc = params["custom"]["imitate_mpc"]
        self.force_feasible = params["custom"]["force_feasible"]
        self.feasible_lambda = params["custom"]["feasible_lambda"]
        self.is_test = params["custom"]["train_or_test"] == "test"
        self.run_name = params["custom"]["run_name"]

class A2CQPUnrolledBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = A2CQPUnrolled(self.params, **kwargs)
        return net
