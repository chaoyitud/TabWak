"""
Variational Autoencoder (VAE) implementation for tabular data.
This module contains the core components for a VAE model including tokenization,
attention mechanisms, and reconstruction layers.
"""

import math
import typing as ty
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
from torch import Tensor

class Tokenizer(nn.Module):
    """Tokenizes numerical and categorical features into a common embedding space.
    
    Args:
        d_numerical (int): Number of numerical features
        categories (ty.Optional[ty.List[int]]): List of category sizes for categorical features
        d_token (int): Dimension of token embeddings
        bias (bool): Whether to use bias in tokenization
    """
    
    def __init__(self, d_numerical: int, categories: ty.Optional[ty.List[int]], 
                 d_token: int, bias: bool):
        super().__init__()
        
        # Handle categorical features
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # Initialize tokenization weights
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))  # +1 for [CLS] token
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        
        # Initialize weights using Kaiming initialization
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        """Returns the total number of tokens including [CLS] and categorical tokens."""
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: ty.Optional[Tensor], x_cat: ty.Optional[Tensor]) -> Tensor:
        """Forward pass for tokenization.
        
        Args:
            x_num (ty.Optional[Tensor]): Numerical features
            x_cat (ty.Optional[Tensor]): Categorical features
            
        Returns:
            Tensor: Tokenized features
        """
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None, "At least one of x_num or x_cat must be provided"
        
        # Add [CLS] token
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
    
        x = self.weight[None] * x_num[:, :, None]

        # Add categorical embeddings if present
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
            
        # Add bias if enabled
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x

class MLP(nn.Module):
    """Simple Multi-Layer Perceptron with ReLU activation and dropout.
    
    Args:
        input_dim (int): Input dimension
        hidden_dim (int): Hidden layer dimension
        output_dim (int): Output dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

class MultiheadAttention(nn.Module):
    """Multi-head attention mechanism.
    
    Args:
        d (int): Input dimension
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability
        initialization (str): Weight initialization method ('xavier' or 'kaiming')
    """
    
    def __init__(self, d: int, n_heads: int, dropout: float, initialization: str = 'kaiming'):
        if n_heads > 1:
            assert d % n_heads == 0, "d must be divisible by n_heads"
        assert initialization in ['xavier', 'kaiming'], "initialization must be 'xavier' or 'kaiming'"

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        # Initialize weights
        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        """Reshape input for multi-head attention.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, n_tokens, d)
            
        Returns:
            Tensor: Reshaped tensor for multi-head attention
        """
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(self, x_q: Tensor, x_kv: Tensor, 
                key_compression: ty.Optional[nn.Module] = None,
                value_compression: ty.Optional[nn.Module] = None) -> Tensor:
        """Forward pass for multi-head attention.
        
        Args:
            x_q (Tensor): Query tensor
            x_kv (Tensor): Key-value tensor
            key_compression (ty.Optional[nn.Module]): Optional key compression module
            value_compression (ty.Optional[nn.Module]): Optional value compression module
            
        Returns:
            Tensor: Attention output
        """
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        
        # Validate dimensions
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, "tensor dimension must be divisible by n_heads"
            
        # Apply compression if provided
        if key_compression is not None:
            assert value_compression is not None, "value_compression must be provided with key_compression"
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None, "value_compression must be None if key_compression is None"

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        # Reshape for multi-head attention
        q = self._reshape(q)
        k = self._reshape(k)

        # Compute attention scores
        a = q @ k.transpose(1, 2)
        b = math.sqrt(d_head_key)
        attention = F.softmax(a/b, dim=-1)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            attention = self.dropout(attention)
            
        # Compute output
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        
        # Apply output projection if needed
        if self.W_out is not None:
            x = self.W_out(x)

        return x

class Transformer(nn.Module):
    """Transformer encoder with multi-head attention and feed-forward networks.
    
    Args:
        n_layers (int): Number of transformer layers
        d_token (int): Token dimension
        n_heads (int): Number of attention heads
        d_out (int): Output dimension
        d_ffn_factor (int): Factor for feed-forward network dimension
        attention_dropout (float): Dropout probability for attention
        ffn_dropout (float): Dropout probability for feed-forward network
        residual_dropout (float): Dropout probability for residual connections
        activation (str): Activation function ('relu' or 'gelu')
        prenormalization (bool): Whether to use pre-normalization
        initialization (str): Weight initialization method
    """
    
    def __init__(
        self,
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_out: int,
        d_ffn_factor: int,
        attention_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        activation: str = 'relu',
        prenormalization: bool = True,
        initialization: str = 'kaiming',      
    ):
        super().__init__()

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        
        # Create transformer layers
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(d_token, d_hidden),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
   
            self.layers.append(layer)

        # Initialize activation functions
        self.activation = nn.ReLU()
        self.last_activation = nn.ReLU()
        
        # Store configuration
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x: Tensor, layer: nn.ModuleDict, norm_idx: int) -> Tensor:
        """Start residual connection with optional normalization.
        
        Args:
            x (Tensor): Input tensor
            layer (nn.ModuleDict): Layer containing normalization modules
            norm_idx (int): Index of normalization module
            
        Returns:
            Tensor: Normalized tensor if prenormalization is enabled
        """
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x: Tensor, x_residual: Tensor, 
                     layer: nn.ModuleDict, norm_idx: int) -> Tensor:
        """End residual connection with optional normalization and dropout.
        
        Args:
            x (Tensor): Original input tensor
            x_residual (Tensor): Residual tensor
            layer (nn.ModuleDict): Layer containing normalization modules
            norm_idx (int): Index of normalization module
            
        Returns:
            Tensor: Output tensor after residual connection
        """
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the transformer.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention block
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](x_residual, x_residual)
            x = self._end_residual(x, x_residual, layer, 0)

            # Feed-forward block
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
            
        return x

class AE(nn.Module):
    """Autoencoder with multi-head attention.
    
    Args:
        hid_dim (int): Hidden dimension
        n_head (int): Number of attention heads
    """
    
    def __init__(self, hid_dim: int, n_head: int):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head

        self.encoder = MultiheadAttention(hid_dim, n_head)
        self.decoder = MultiheadAttention(hid_dim, n_head)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get the encoded representation of input.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Encoded representation
        """
        return self.encoder(x, x).detach() 

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the autoencoder.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Reconstructed output
        """
        z = self.encoder(x, x)
        h = self.decoder(z, z)
        return h

class VAE(nn.Module):
    """Variational Autoencoder with transformer-based encoder and decoder.
    
    Args:
        d_numerical (int): Number of numerical features
        categories (ty.List[int]): List of category sizes for categorical features
        num_layers (int): Number of transformer layers
        hid_dim (int): Hidden dimension
        n_head (int): Number of attention heads
        factor (int): Factor for feed-forward network dimension
        bias (bool): Whether to use bias in tokenization
    """
    
    def __init__(self, d_numerical: int, categories: ty.List[int], num_layers: int, 
                 hid_dim: int, n_head: int = 1, factor: int = 4, bias: bool = True):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.hid_dim = hid_dim
        d_token = hid_dim
        self.n_head = n_head
 
        # Initialize components
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder_mu = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.encoder_logvar = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)
        self.decoder = Transformer(num_layers, hid_dim, n_head, hid_dim, factor)

    def get_embedding(self, x: Tensor) -> Tensor:
        """Get the encoded representation of input.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Encoded representation
        """
        return self.encoder_mu(x, x).detach() 

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling from the latent space.
        
        Args:
            mu (Tensor): Mean of the latent distribution
            logvar (Tensor): Log variance of the latent distribution
            
        Returns:
            Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_num: Tensor, x_cat: Tensor) -> ty.Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the VAE.
        
        Args:
            x_num (Tensor): Numerical features
            x_cat (Tensor): Categorical features
            
        Returns:
            ty.Tuple[Tensor, Tensor, Tensor]: Reconstructed output, mean, and log variance
        """
        # Tokenize input
        x = self.Tokenizer(x_num, x_cat)

        # Encode to latent space
        mu_z = self.encoder_mu(x)
        std_z = self.encoder_logvar(x)
        z = self.reparameterize(mu_z, std_z)
        
        # Decode from latent space
        h = self.decoder(z[:, 1:])
        
        return h, mu_z, std_z

class Reconstructor(nn.Module):
    """Reconstructs numerical and categorical features from latent representations.
    
    Args:
        d_numerical (int): Number of numerical features
        categories (ty.List[int]): List of category sizes for categorical features
        d_token (int): Token dimension
    """
    
    def __init__(self, d_numerical: int, categories: ty.List[int], d_token: int):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        # Initialize numerical reconstruction weights
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))  
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        
        # Initialize categorical reconstruction layers
        self.cat_recons = nn.ModuleList()
        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h: Tensor) -> ty.Tuple[Tensor, ty.List[Tensor]]:
        """Forward pass for reconstruction.
        
        Args:
            h (Tensor): Latent representation
            
        Returns:
            ty.Tuple[Tensor, ty.List[Tensor]]: Reconstructed numerical and categorical features
        """
        # Split latent representation
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]

        # Reconstruct numerical features
        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        
        # Reconstruct categorical features
        recon_x_cat = []
        for i, recon in enumerate(self.cat_recons):
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class Model_VAE(nn.Module):
    """Complete VAE model combining VAE and reconstruction components.
    
    Args:
        num_layers (int): Number of transformer layers
        d_numerical (int): Number of numerical features
        categories (ty.List[int]): List of category sizes for categorical features
        d_token (int): Token dimension
        n_head (int): Number of attention heads
        factor (int): Factor for feed-forward network dimension
        bias (bool): Whether to use bias in tokenization
    """
    
    def __init__(self, num_layers: int, d_numerical: int, categories: ty.List[int], 
                 d_token: int, n_head: int = 1, factor: int = 4, bias: bool = True):
        super().__init__()
        
        # Initialize VAE and reconstruction components
        self.VAE = VAE(d_numerical, categories, num_layers, d_token, 
                       n_head=n_head, factor=factor, bias=bias)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def get_embedding(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Get the encoded representation of input.
        
        Args:
            x_num (Tensor): Numerical features
            x_cat (Tensor): Categorical features
            
        Returns:
            Tensor: Encoded representation
        """
        x = self.Tokenizer(x_num, x_cat)
        return self.VAE.get_embedding(x)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> ty.Tuple[Tensor, ty.List[Tensor], Tensor, Tensor]:
        """Forward pass through the complete model.
        
        Args:
            x_num (Tensor): Numerical features
            x_cat (Tensor): Categorical features
            
        Returns:
            ty.Tuple[Tensor, ty.List[Tensor], Tensor, Tensor]: Reconstructed numerical features,
                reconstructed categorical features, mean, and log variance
        """
        # Encode and decode through VAE
        h, mu_z, std_z = self.VAE(x_num, x_cat)

        # Reconstruct features
        recon_x_num, recon_x_cat = self.Reconstructor(h)

        return recon_x_num, recon_x_cat, mu_z, std_z


class Encoder_model(nn.Module):
    """Encoder model for extracting latent representations.
    
    Args:
        num_layers (int): Number of transformer layers
        d_numerical (int): Number of numerical features
        categories (ty.List[int]): List of category sizes for categorical features
        d_token (int): Token dimension
        n_head (int): Number of attention heads
        factor (int): Factor for feed-forward network dimension
        bias (bool): Whether to use bias in tokenization
    """
    
    def __init__(self, num_layers: int, d_numerical: int, categories: ty.List[int], 
                 d_token: int, n_head: int, factor: int, bias: bool = True):
        super().__init__()
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.VAE_Encoder = Transformer(num_layers, d_token, n_head, d_token, factor)

    def load_weights(self, Pretrained_VAE: Model_VAE) -> None:
        """Load weights from a pretrained VAE model.
        
        Args:
            Pretrained_VAE (Model_VAE): Pretrained VAE model
        """
        self.Tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(Pretrained_VAE.VAE.encoder_mu.state_dict())

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        """Forward pass through the encoder.
        
        Args:
            x_num (Tensor): Numerical features
            x_cat (Tensor): Categorical features
            
        Returns:
            Tensor: Encoded representation
        """
        x = self.Tokenizer(x_num, x_cat)
        z = self.VAE_Encoder(x)
        return z


class Decoder_model(nn.Module):
    """Decoder model for reconstructing features from latent representations.
    
    Args:
        num_layers (int): Number of transformer layers
        d_numerical (int): Number of numerical features
        categories (ty.List[int]): List of category sizes for categorical features
        d_token (int): Token dimension
        n_head (int): Number of attention heads
        factor (int): Factor for feed-forward network dimension
        bias (bool): Whether to use bias in tokenization
    """
    
    def __init__(self, num_layers: int, d_numerical: int, categories: ty.List[int], 
                 d_token: int, n_head: int, factor: int, bias: bool = True):
        super().__init__()
        self.VAE_Decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def load_weights(self, Pretrained_VAE: Model_VAE) -> None:
        """Load weights from a pretrained VAE model.
        
        Args:
            Pretrained_VAE (Model_VAE): Pretrained VAE model
        """
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(Pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z: Tensor) -> ty.Tuple[Tensor, ty.List[Tensor]]:
        """Forward pass through the decoder.
        
        Args:
            z (Tensor): Latent representation
            
        Returns:
            ty.Tuple[Tensor, ty.List[Tensor]]: Reconstructed numerical and categorical features
        """
        h = self.VAE_Decoder(z)
        x_hat_num, x_hat_cat = self.Detokenizer(h)
        return x_hat_num, x_hat_cat
