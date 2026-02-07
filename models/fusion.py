"""Fusion module to combine image, text, and state tokens"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionMLP(nn.Module):
    def __init__(self, context_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self, img_token, text_token, state_token):
        x = torch.cat(img_token, text_token, state_token, dim=0)
        x = self.net(x)
        x = self.ln(x)
        return x