from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from src.recommendation.common_layers.AdditiveAttention import AdditiveAttention


class DeBERTaNewsEncoder(nn.Module):
    """
    News Encoder using DeBERTa-v3-base with CNN and Additive Attention.
    Compatible with NAML architecture.
    """
    def __init__(self, pretrained="microsoft/deberta-v3-base", conv_kernel_num=400, kernel_size=3, query_dim=200):
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained, use_safetensors=True)
        hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size
        self.cnn = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=conv_kernel_num,
            kernel_size=kernel_size,
            padding="same"
        )
        self.additive_attention = AdditiveAttention(conv_kernel_num, query_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) - input_ids tensor
            attention_mask: (batch_size, seq_len) - attention mask tensor
        
        Returns:
            (batch_size, conv_kernel_num) - encoded news representation
        """
        # Get PLM hidden states with attention_mask
        plm_output = self.plm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Apply CNN
        e = plm_output.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
        c = self.cnn(e)  # (batch_size, conv_kernel_num, seq_len)
        c = c.transpose(1, 2)  # (batch_size, seq_len, conv_kernel_num)
        
        # Apply Additive Attention
        c_att = self.additive_attention(c)  # (batch_size, seq_len, conv_kernel_num)
        
        # Sum over sequence dimension
        vec = torch.sum(c_att, dim=1)  # (batch_size, conv_kernel_num)
        return vec
