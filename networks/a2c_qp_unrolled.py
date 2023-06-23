import torch
from torch import nn
import torch.nn.functional as F
from rl_games.algos_torch.network_builder import NetworkBuilder, A2CBuilder

class A2CQPUnrolled(A2CBuilder.Network):
    """TODO: adapt to QP unrolled network."""
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.value_size = kwargs.pop('value_size', 1)

        NetworkBuilder.BaseNetwork.__init__(self)
        self.n_total_obs = input_shape[0]
        self.load(params)

        if self.separate:
            raise NotImplementedError()
        
        encoder_args = {
            'input_size' : self.n_teacher_obs, 
            'units' : [i * self.latent_size for i in [4, 2, 1]], 
            'activation' : self.activation, 
            'norm_func_name' : self.normalization,
            'dense_func' : torch.nn.Linear,
            'd2rl' : self.is_d2rl,
            'norm_only_first_layer' : self.norm_only_first_layer
        }
        self.encoder = self._build_mlp(**encoder_args)
        body_args = {
            'input_size' : self.n_common_obs + self.latent_size,
            'units' : self.units,
            'activation' : self.activation, 
            'norm_func_name' : self.normalization,
            'dense_func' : torch.nn.Linear,
            'd2rl' : self.is_d2rl,
        }
        self.body = self._build_mlp(**body_args)

        out_size = self.units[-1]
        self.value = torch.nn.Linear(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
        mu_init = self.init_factory.create(**self.space_config['mu_init'])
        self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
        sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
        if self.fixed_sigma:
            self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
        else:
            self.sigma = torch.nn.Linear(out_size, actions_num)

        mlp_init = self.init_factory.create(**self.initializer)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)    

        mu_init(self.mu.weight)
        if self.fixed_sigma:
            sigma_init(self.sigma)
        else:
            sigma_init(self.sigma.weight)  


    def forward(self, obs_dict):
        obs = obs_dict['obs']
        states = obs_dict.get('rnn_states', None)
        common_obs, teacher_obs = obs[:, :self.n_common_obs], obs[:, self.n_common_obs:]
        self.latent = self.encoder(teacher_obs)
        mlp_in = torch.cat((common_obs, self.latent), -1)
        mlp_out = self.body(mlp_in)
        value = self.value_act(self.value(mlp_out))
        mu = self.mu_act(self.mu(mlp_out))
        if self.fixed_sigma:
            sigma = self.sigma_act(self.sigma)
        else:
            sigma = self.sigma_act(self.sigma(mlp_out))
        return mu, mu*0 + sigma, value, states

    def load(self, params):
        A2CBuilder.Network.load(self, params)
        self.n_teacher_obs = params["custom"]["n_teacher_obs"]
        self.n_common_obs = self.n_total_obs - self.n_teacher_obs
        self.latent_size = params["custom"]["latent_size"]

class A2CTeacherBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = A2CTeacher(self.params, **kwargs)
        return net


