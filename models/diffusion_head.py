'''
Implements following classes:
- Sinusoidal Position Embedding
- Diffusion policy (DDPM)
    - Forward and reverse process
    - Loss function
'''
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DiffusionConfig:
    T: int = 16 # number of diffusion steps
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    action_dim: int = 4
    cond_dim: int = 128 # conditional input dim

def make_beta_schedule(cfg: DiffusionConfig):
    betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.T)
    sqrt_betas =  torch.sqrt(betas)
    alphas = 1 - betas
    alpha_cumulative = torch.cumprod(alphas, dim=0)
    sqrt_alpha_cumulative = torch.sqrt(alpha_cumulative)
    one_by_sqrt_alpha = 1. / torch.sqrt(alphas)
    sqrt_one_minus_alpha_cumulative = torch.sqrt(1. - alpha_cumulative)
    return betas, sqrt_betas, alpha_cumulative, one_by_sqrt_alpha, sqrt_alpha_cumulative, sqrt_one_minus_alpha_cumulative


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, emb_dim=128, timesteps=1000):
        super().__init__()
        self.emb_dim = emb_dim

        half_dim  = self.emb_dim // 2 # half for sin and half for cosine waves
        # compute (1 / 10000^(i/d))
        # 10000^(i/d) = exp(log(10000) * i/d)
        emb = math.log(10000) / (half_dim - 1)
        i = torch.arange(half_dim, dtype=torch.float32) # i = [0, ..., 63]
        # inverse frequencies (or wavelengths): 1 / 10000^(i/d)
        freq = torch.exp(i * -emb)
        # compute  (pos / 10000^(i/d))
        # create timesteps (number of positions)
        # for every position we have a d-dimension vector(128)
        pos = torch.arange(timesteps, dtype=torch.float32) # (1000,)
        emb = torch.unsqueeze(pos, dim=-1) * torch.unsqueeze(freq, dim=0) # (1000, 1) * (1, 64)
        # (sin, cos, sin, cos, ...)
        emb_vec = torch.cat((emb.sin(), emb.cos()), dim=-1) # (1000, 128)

        # Define MLP
        self.net = nn.Sequential(
            nn.Embedding.from_pretreained(emb_vec), # creates a lookup table from t -> row in emb_vec: When you pass in time step t (an integer), it returns row t from our matrix
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
        )

    
    def forward(self, t):
        # t: (B, ) integer timesteps in [0, T-1]
        # return (pos, emb_dim) vector
        return self.net(t)


class ActionDenoiseModel(nn.Module):
    '''
    Inputs
        - noisy actions (B, action_dim)
        - fused context vector (B, d_model)
        - position/time embeddings: (B, d_model)
    Output
        - predicted noise (e_p)
    '''
    def __init__(self, cfg: DiffusionConfig, time_emb=32, hidden_dim=128):
        super().__init__()
        self.cfg = cfg
        self.pos_emb = SinusoidalPositionEmbedding(emb_dim=time_emb, timesteps=self.cfg.T)
        in_dim = time_emb + self.cfg.action_dim + cfg.cond_dim
        # model goes from (t, a, c) -> action_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.cfg.action_dim)
        )
    
    def forward(self, x_t, t, cond):
        """        
        :param self: Description
        :param t: timestep
        :param x: noisy action
        :param c: conditional embedding (fused)
        """
        t_emb = self.pos_emb(t)
        comb = torch.cat((x_t, t_emb, cond), dim=-1)
        eps_pred = self.net(comb)
        return eps_pred


class DiffusionPolicyHead(nn.Module):
    """
    Runs training and inference
    Training:
    - compute loss
        - forward process (q_sample): add noise to action based on t: eps_gt
        - Run ActiondenoiseModel(t, noisy_action, cond) to get eps_pred
        - compute MSE_LOSS(eps_gt, eps_pred)

    Inference (Sampling):
        - Compute denoised action conditioned on encoded inputs (image, text, state)

    """
    def __init__(self):
        super().__init__()
        self.cfg = DiffusionConfig
        betas, sqrt_betas, alpha_cumulative, one_by_sqrt_alpha, \
        sqrt_alpha_cumulative, sqrt_one_minus_alpha_cumulative = make_beta_schedule(self.cfg)
        self.denoise_action_model = ActionDenoiseModel(self.cfg, time_emb=32)
        '''
        - register a non-trainable tensor as part of a module's state
        - buffer is automatically included in the model's state_dict() when saving the model
        - automatically moved to the specified device
        '''
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", alpha_cumulative)
        self.register_buffer("sqrt_alpha_bar", sqrt_alpha_cumulative)
        self.register_buffer("sqrt_one_minus_alpha_bar", sqrt_one_minus_alpha_cumulative)
        self.register_buffer("one_by_sqrt_alpha", one_by_sqrt_alpha)
        self.register_buffer("sqrt_betas", sqrt_betas)

    def q_sample(self, x0, t, noise):
        '''
        Forward process for diffusion: take a clean action and adds noise based on t
        
        :param self: Description
        :param x0: action without noise (B, action_dim)
        :param t: timestep until to iteratively add noise (B,)
        :param noise: noise to add to the action
        '''
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].unsqueeze(dim=-1) # (B, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].unsqueeze(dim=-1)
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt
    
    def loss(self, actions, cond):
        B = actions.size(0)
        device = actions.device
        # random noise for action
        noise = torch.rand_like(actions) # (B, action_dim)
        # random timesteps
        t = torch.randint(0, self.cfg.T, (B,), device=device)
        # generate noisy actions
        xt = self.q_sample(actions, t=t, noise=noise)
        # predict noise from noisy actions conditioned over fused inputs (context)
        eps_pred = self.denoise_action_model(xt, t, cond)
        # compute loss
        loss = F.mse_loss(noise, eps_pred)
        return loss

    @torch.no_grad()
    def sample(self, cond, n_samples=None):
        '''        
        :param self: Description
        :param cond: fused context vector (B, model_dim) or (1, model_dim)
        returns: actions (B, action_dim)
        '''
        self.eval()
        if n_samples is None:
            B = cond.size(0)
        else:
            B = n_samples
            cond = cond.expand(B, -1)
        # random noisy actions (B, action_dim)
        x_t = torch.randn(B, self.cfg.action_dim, device=cond.device)
        # iteratively denoise
        for t_step in reversed(range(self.cfg.T)):
            # create timestep vector
            t = torch.ones((B,), dtype=torch.long, device=cond.device) * t_step
            # predict noise
            eps_pred = self.denoise_action_model(x_t, t, cond)
            # denoise action (reverse diffusion)
            # mean = (1 / torch.sqrt(alpha_t)) * (x_t - beta_t / torch.sqrt(1 - alpha_bar_t) * eps_pred) # original
            x0_pred = (x_t * self.sqrt_one_minus_alpha_bar[t] * eps_pred) * (1.0 / self.sqrt_alpha_bar[t]) # simplified
            # add nosie for stochasticity and prevent mode collapse
            if t_step > 0:
                noise = torch.randn_like(x_t)
                # x_t = mean + torch.sqrt(beta_t) * noise # original
                x_t = self.alphas[t] * x0_pred + self.sqrt_betas[t] * noise
            else:
                x_t = x0_pred
        
        return x_t
                