import torch
import torch.nn as nn
import latent_ode.utils as utils
import numpy as np

from s4.s4 import S4Block as S4
from s4.s4d import S4D
from torch.distributions import Normal
from dataclasses import dataclass
from latent_ode.latent_ode import ODEFunc, DiffeqSolver, Decoder, Encoder_z0_RNN
from latent_ode.latent_ode import ODEGRU, VanillaGRU, VAEGRU, ExpDecayGRU, DeltaTGRU, LatentODE, VanillaLSTM
from latent_ode.latent_ode import BaseRecurrentModel, BaseVAEModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class S4Model(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=256, n_layers=4, dropout=0.2, lr=0.001, prenorm=False, use_s4d=True):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 17 for walker)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            if use_s4d:
                self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr)))
            else:
                self.s4_layers.append(S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr)))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, times):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x
    
    def param_count(self):
        params = 0
        for param in self.parameters():
            params += np.prod(param.shape)
        return params


class BaselineModel(nn.Module):
    def __init__(self, model_name, d_input, d_output, d_model=400, ode_dim=400,
                 n_layers=1, enc_hidden_to_latent_dim=20, ode_tol=1e-5, eps_decay=0):
        super(BaselineModel, self).__init__()
        gen_ode_func = ODEFunc(ode_func_net=utils.create_net(d_model, d_model, n_layers=2, n_units=ode_dim,
                                                                nonlinear=nn.Tanh)).to(device)
        diffq_solver = DiffeqSolver(gen_ode_func, 'dopri5', odeint_rtol=ode_tol, odeint_atol=ode_tol/10)

        # encoder
        encoder = Encoder_z0_RNN(d_model, d_input, hidden_to_z0_units=enc_hidden_to_latent_dim,
                                 device=device).to(device)
        z0_prior = Normal(torch.tensor([0.]).to(device), torch.tensor([1.]).to(device))

        # decoder
        decoder = Decoder(d_model, d_output, n_layers=0).to(device)
        self.model_name = model_name

        if model_name == 'gru':
            self.model = VanillaGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                device=device).to(device)
        elif model_name == 'latentode':
            self.model = LatentODE(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=decoder,
                diffeq_solver=diffq_solver,
                z0_prior=z0_prior,
                device=device).to(device)
        elif model_name == 'expdecaygru':
            self.model = ExpDecayGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                device=device).to(device)
        elif model_name == 'odegru':
            self.model = ODEGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                diffeq_solver=diffq_solver,
                device=device).to(device)
        elif model_name == 'vaegru':
            self.model = VAEGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                encoder_z0=encoder,
                decoder=decoder,
                z0_prior=z0_prior,
                device=device).to(device)
        elif model_name == 'deltatgru':
            self.model = DeltaTGRU(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                device=device).to(device)
        elif model_name == 'lstm':
            self.model = VanillaLSTM(
                input_dim=d_input,
                latent_dim=d_model,
                eps_decay=eps_decay,
                decoder=decoder,
                n_layers=n_layers,
                device=device).to(device)
    
    def forward(self, x, times):
        # x: (B, L, D_state)
        if isinstance(self.model, BaseVAEModel) :
            next_states, _, _, _ = self.model.predict_next_states(x, times.squeeze(-1).to(device),
                                        torch.full([x.shape[0], ], fill_value=x.shape[1]).to(device))
        elif isinstance(self.model, BaseRecurrentModel):
            next_states, _ = self.model.predict_next_states(x, times.squeeze(-1).to(device))
        else:
            raise Exception("Unknown model:" + self.model_name )
        return next_states
    
    def __str__(self):
        return self.model_name
    
    def __repr__(self):
        return self.model_name

    def param_count(self):
        params = 0
        for param in self.model.parameters():
            params += np.prod(param.shape)
        return params
