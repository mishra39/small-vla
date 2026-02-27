import torch.nn as nn
from .encoders import TinyCNNImageEncoder, TextEncoderTinyGRU, RobotStateEncoder
from .fusion import FusionMLP
from .diffusion_head import DiffusionConfig, DiffusionPolicyHead

class VLADiffusionPolicy(nn.Module):
    def __init__(self, vocab_size, state_dim, action_dim, 
                 d_model=128, diffusion_T=16):
        super().__init__()
        # Initialize encoders
        self.img_encoder = TinyCNNImageEncoder(d_model=d_model)
        self.text_encoder = TextEncoderTinyGRU(vocab_size=vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = RobotStateEncoder(d_model=d_model)
        self.fusion = FusionMLP(d_model=d_model)

        # Diffusion Config
        self.cfg = DiffusionConfig(
                diffusion_T,
                action_dim=action_dim,
                cond_dim=d_model
                )
        
        self.diffusion_head = DiffusionPolicyHead(self.cfg)

    def encode_obs(self, img, text_tokens, state):
        '''
        Encodes observations and returns fused context
        
        :param self: Description
        '''
        img_token = self.img_encoder(img)
        txt_token = self.text_encoder(text_tokens)
        state_token = self.state_encoder(state)
        fused_context = self.fusion(img_token, txt_token, state_token)
        return fused_context
    
    def loss(self, img, text_tokens, state, actions):
        cond = self.encode_obs(img, text_tokens, state)
        loss = self.diffusion_head.loss(actions, cond)
        return loss
    
    def act(self, img, text_tokens, state):
        cond = self.encode_obs(img, text_tokens, state)
        actions = self.diffusion_head.sample(cond)
        return actions
