"""Encoders for images, text, and state"""

import torch
import torch.nn as nn
import torch.nn.functional as F

''' Network Architecture
- Input: RGBImage batch (B, 3, H, W)
- Output: Encoded image vector (d_model)
- rgb image (3, H, W) -> conv1 (3, 32) + ReLU -> conv2(32,64) + ReLU -> conv3(64, 128) + ReLU -> GAP -> Projection () -> LayerNorm
'''

class TinyCNNImageEncoder(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # input: (B, 3, H, W)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.gap   = nn.AdaptiveAvgPool2d(output_size=d_model)
        self.proj  = nn.Linear(128, d_model) # Shape (B, d_model, H', W')
        self.ln    = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)) # (128, H', W')
        x = self.gap(x) # GAP (128,)
        x = self.proj(x) # 128 -> d_model
        x = self.ln(x)

        return x # (B, d_model)
        
class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_word)
        self.gru   = nn.GRU(input_size=d_word, hidden_size=d_model, batch_first=True)
        self.ln    = nn.LayerNorm(d_model)
    
    def forward(self,x):
        x = self.embed(x)
        _, h_last = self.gru(x)
        x = h_last[0]
        x = self.ln(x)

        return x


class RobotStateEncoder(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model))
        self.ln = nn.LayerNorm(d_model)
    
    def forward(self,x):
        x = self.net(x)
        x = self.ln(x)