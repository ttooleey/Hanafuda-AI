"""
Neural network models for Koi-Koi AI.

This module defines the Transformer-based neural network architecture
used for learning Koi-Koi game strategies.

Architecture:
    Input -> Conv1d (FF) -> LayerNorm -> TransformerEncoder -> Conv1d (FF) -> Output

The network uses a Transformer encoder to process card token representations,
allowing it to learn relationships between cards and game states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from koikoi.core.constants import (
    FEATURE_INPUT_DIM,
    FEATURE_EMBEDDING_DIM,
    FEATURE_FEEDFORWARD_DIM,
    ATTENTION_HEADS,
    ENCODER_LAYERS,
)


# Default network parameters (new naming convention)
NET_PARAMETERS = {
    'n_input': FEATURE_INPUT_DIM,
    'n_emb': FEATURE_EMBEDDING_DIM,
    'n_feedforward': FEATURE_FEEDFORWARD_DIM,
    'n_attn_head': ATTENTION_HEADS,
    'n_layer': ENCODER_LAYERS,
}

# Legacy parameter names for backward compatibility with saved models
NetParameter = {
    'nInput': FEATURE_INPUT_DIM,
    'nEmb': FEATURE_EMBEDDING_DIM,
    'nFw': FEATURE_FEEDFORWARD_DIM,
    'nAttnHead': ATTENTION_HEADS,
    'nLayer': ENCODER_LAYERS,
}


class KoiKoiEncoderBlock(nn.Module):
    """
    Transformer encoder block for Koi-Koi card representation.
    
    Architecture:
        1. Feedforward expansion: input_dim -> feedforward_dim (Conv1d + ReLU)
        2. Projection: feedforward_dim -> embedding_dim (Conv1d)
        3. Layer normalization
        4. Transformer encoder layers
    
    The input is treated as a sequence of card tokens, where each token
    represents a card with its associated features.
    
    Args:
        n_input/nInput: Input feature dimension
        n_emb/nEmb: Embedding dimension for transformer
        n_feedforward/nFw: Feedforward dimension
        n_attn_head/nAttnHead: Number of attention heads
        n_layer/nLayer: Number of transformer encoder layers
    
    Input shape: (batch_size, n_input, sequence_length)
    Output shape: (batch_size, n_emb, sequence_length)
    """
    
    def __init__(
        self,
        # New parameter names
        n_input: int = None,
        n_emb: int = None,
        n_feedforward: int = None,
        n_attn_head: int = None,
        n_layer: int = None,
        # Legacy parameter names (for backward compatibility)
        nInput: int = None,
        nEmb: int = None,
        nFw: int = None,
        nAttnHead: int = None,
        nLayer: int = None,
    ) -> None:
        super().__init__()
        
        # Support both new and legacy parameter names
        n_input = n_input or nInput or NET_PARAMETERS['n_input']
        n_emb = n_emb or nEmb or NET_PARAMETERS['n_emb']
        n_feedforward = n_feedforward or nFw or NET_PARAMETERS['n_feedforward']
        n_attn_head = n_attn_head or nAttnHead or NET_PARAMETERS['n_attn_head']
        n_layer = n_layer or nLayer or NET_PARAMETERS['n_layer']
        
        # Feedforward layers (implemented as 1x1 convolutions)
        # Note: Using original attribute names (f1, f2, attn_encoder)
        # for compatibility with pre-trained model weights
        self.f1 = nn.Conv1d(n_input, n_feedforward, kernel_size=1)
        self.f2 = nn.Conv1d(n_feedforward, n_emb, kernel_size=1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_attn_head,
            dim_feedforward=n_feedforward,
        )
        self.attn_encoder = nn.TransformerEncoder(encoder_layer, n_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder block.
        
        Args:
            x: Input tensor of shape (batch_size, n_input, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, n_emb, seq_len)
        """
        # Feedforward: expand then project
        x = self.f2(F.relu(self.f1(x)))
        
        # Layer normalization
        x = F.layer_norm(x, [x.size(-1)])
        
        # Transformer expects (seq_len, batch_size, n_emb)
        x = x.permute(2, 0, 1)
        x = self.attn_encoder(x)
        x = x.permute(1, 2, 0)
        
        return x


class DiscardModel(nn.Module):
    """
    Model for discard action decision.
    
    Given the current game state, predicts the value of discarding
    each card from hand. The output is a score for each of 48 cards.
    
    Architecture:
        KoiKoiEncoderBlock -> Conv1d (1) -> Squeeze
    
    Input shape: (batch_size, n_input, 48)
    Output shape: (batch_size, 48)
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder_block = KoiKoiEncoderBlock(**NET_PARAMETERS)
        # Note: Using 'out' for compatibility with pre-trained model weights
        self.out = nn.Conv1d(NET_PARAMETERS['n_emb'], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for discard decision.
        
        Args:
            x: Game state tensor of shape (batch_size, n_input, 48)
            
        Returns:
            Score tensor of shape (batch_size, 48)
        """
        x = self.encoder_block(x)
        x = self.out(x).squeeze(1)
        return x


class PickModel(nn.Module):
    """
    Model for pick action decision.
    
    Given the current game state, predicts the value of picking
    each field card to pair with the shown card.
    
    Architecture:
        KoiKoiEncoderBlock -> Conv1d (1) -> Squeeze
    
    Input shape: (batch_size, n_input, 48)
    Output shape: (batch_size, 48)
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder_block = KoiKoiEncoderBlock(**NET_PARAMETERS)
        # Note: Using 'out' for compatibility with pre-trained model weights
        self.out = nn.Conv1d(NET_PARAMETERS['n_emb'], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pick decision.
        
        Args:
            x: Game state tensor of shape (batch_size, n_input, 48)
            
        Returns:
            Score tensor of shape (batch_size, 48)
        """
        x = self.encoder_block(x)
        x = self.out(x).squeeze(1)
        return x


class KoiKoiModel(nn.Module):
    """
    Model for koi-koi decision (continue or stop).
    
    Given the current game state, predicts the value of stopping
    vs. continuing (calling koi-koi).
    
    Architecture:
        KoiKoiEncoderBlock -> Select first 2 tokens -> Conv1d (1) -> Squeeze
    
    Input shape: (batch_size, n_input, 50)
        Note: For koi-koi decisions, the input has 50 columns
        (2 decision tokens + 48 card tokens)
    
    Output shape: (batch_size, 2)
        Index 0: Value of stopping
        Index 1: Value of continuing (koi-koi)
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder_block = KoiKoiEncoderBlock(**NET_PARAMETERS)
        # Note: Using 'out' for compatibility with pre-trained model weights
        self.out = nn.Conv1d(NET_PARAMETERS['n_emb'], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for koi-koi decision.
        
        Args:
            x: Game state tensor of shape (batch_size, n_input, 50)
            
        Returns:
            Score tensor of shape (batch_size, 2)
                [:, 0] = value of stopping
                [:, 1] = value of continuing
        """
        x = self.encoder_block(x)
        # Select only the first 2 tokens (stop and continue)
        x = self.out(x[:, :, [0, 1]]).squeeze(1)
        return x


class TargetQNet(nn.Module):
    """
    Target Q-network for value estimation in reinforcement learning.
    
    This model estimates the expected value of a game state,
    used as a target network in DQN-style training.
    
    Architecture:
        KoiKoiEncoderBlock -> Select first token -> Conv1d (1) -> Squeeze
    
    Input shape: (batch_size, n_input, 48 or 50)
    Output shape: (batch_size, 1)
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.encoder_block = KoiKoiEncoderBlock(**NET_PARAMETERS)
        # Note: Using 'out' for compatibility with pre-trained model weights
        self.out = nn.Conv1d(NET_PARAMETERS['n_emb'], 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for value estimation.
        
        Args:
            x: Game state tensor of shape (batch_size, n_input, seq_len)
            
        Returns:
            Value tensor of shape (batch_size, 1)
        """
        x = self.encoder_block(x)
        # Select only the first token for value estimation
        x = self.out(x[:, :, 0].unsqueeze(2)).squeeze(1)
        return x
