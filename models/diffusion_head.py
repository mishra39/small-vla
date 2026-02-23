'''
Implements following classes:
- Sinusoidal Position Embedding
- Diffusion policy (DDPM)
    - Forward and reverse process
    - Loss function
'''
import torch
import torch.nn as nn
import math

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
    one_by_sqrt_alpha = 1. / torch.sqrt(alphas)
    sqrt_one_minus_alpha_cumulative = torch.sqrt(1. - alpha_cumulative)
    return betas, sqrt_betas, alpha_cumulative, one_by_sqrt_alpha, sqrt_one_minus_alpha_cumulative


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
        Docstring for forward
        
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
    Docstring for DiffusionPolicyHead
    Runs training and inference
    Training:
    - compute loss
        - forward process (q_sample): add noise to action based on t: eps_gt
        - Run ActiondenoiseModel(t, noisy_action, cond) to get eps_pred
        - compute MSE_LOSS(eps_gt, eps_pred)

    Inference (Sampling):
        - Compute denoised action conditioned on encoded inputs (image, text, state)

    """