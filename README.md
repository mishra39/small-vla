# VLA: Pixel to Actions
VLAs fine-tune VLMs for robot actions
- Part 2: https://medium.com/@keivalyap/building-vla-models-from-scratch-ii-0180020dbc85

- Objective: Learn conditional action distribution given image(s), instruction text (W), and robot state (s)

- Inputs
    - RGB Images
    - Text
    - State: (joint positions, etc.)
- Output
    - Action
    - Auxiliary info

- Information Flow
    - Raw input -> Token -> Embedding -> Fusion -> Action
    - Raw RGB Image (3, H, W) -> CNN encoder
    - Text -> Tokens -> Embedding

- Implementation Plan
    - Encoders
        - Vision
            - (3 Conv + ReLU layers) + GAP + 1 Projection layer + LayerNorm
        - Language
            - Text -> Token -> Learned vector in embedding space -> GRU -> Layernorm
        - State
            - Linear(state_dim → 64) → ReLU → Linear(64 → d_model) → LayerNorm
    - Fusion
        - Fuses vision token v, text token t, and state token s into one single context embedding
        - Context embedding becomes the condition for the action head
        - MLP: Linear(3*d_model, d_model) -> ReLU -> Linear(d_model, d_model) -> LayerNorm
        - Cross-attention would be better for spatial grounding
    - Diffusion Model for actions
        - Conditional denoising diffusion probabilistic model (DDPM)-style policy head
        - Input contect vector (c) from fusion model -> stochastic policy over actions a
        - Models action generation as iterative denoising
        - Training objective: minimize MSE between predictive and true noise
    - Training/testing
    
## Why predict noise instead of directly predicting actions?
- Noise distribution is stationary and gaussian
- Learning problem is well conditioned across timesteps
- Model naturally learns a smooth conditional action manifold

    
- Vision Encoder: Extracts visual features from image observations into embeddings
- Vision-language projector: maps these embeddings into language space (LLM's representation space)


