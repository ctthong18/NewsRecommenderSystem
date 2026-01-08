from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput


class NAML(nn.Module):
    """
    NAML model for news recommendation.
    
    Combines a news encoder and user encoder to predict user preferences
    for candidate news articles based on their reading history.
    """
    
    def __init__(
        self,
        news_encoder: nn.Module,
        user_encoder: nn.Module,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialize NAML model.
        
        Args:
            news_encoder: Module to encode news articles
            user_encoder: Module to encode user reading history
            loss_fn: Loss function for training (default: CrossEntropyLoss)
        """
        super().__init__()
        self.news_encoder: nn.Module = news_encoder
        self.user_encoder: nn.Module = user_encoder
        self.loss_fn: nn.Module = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()

    def forward(
        self,
        candidate_news: torch.Tensor,
        news_histories: torch.Tensor,
        user_id: torch.Tensor,
        target: torch.Tensor,
        candidate_attention_mask: Optional[torch.Tensor] = None,
        news_histories_attention_mask: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward pass through NAML model.
        
        Args:
            candidate_news: Candidate news tensor [batch_size, candidate_num, seq_len]
            news_histories: User history tensor [batch_size, history_size, seq_len]
            user_id: User ID tensor [batch_size]
            target: Target labels (index for training, one-hot for validation)
            candidate_attention_mask: Attention mask for candidate news [batch_size, candidate_num, seq_len]
            news_histories_attention_mask: Attention mask for news histories [batch_size, history_size, seq_len]
            
        Returns:
            ModelOutput containing logits, loss, and labels
            
        Note:
            During validation (self.training == False), loss is not calculated.
            Multiple positive labels may exist in validation mode.
        """
        # Encode candidate news
        batch_size, candidate_num, seq_len = candidate_news.size()
        candidate_news_flat = candidate_news.view(batch_size * candidate_num, seq_len)
        
        # Flatten attention_mask if provided
        candidate_attention_mask_flat = None
        if candidate_attention_mask is not None:
            candidate_attention_mask_flat = candidate_attention_mask.view(batch_size * candidate_num, seq_len)
        
        # Shape: [batch_size * candidate_num, seq_len] -> [batch_size * candidate_num, conv_kernel_num]
        news_candidates_encoded = self.news_encoder(candidate_news_flat, attention_mask=candidate_attention_mask_flat)
        conv_kernel_num = news_candidates_encoded.size(-1)
        
        # Reshape: [batch_size * candidate_num, conv_kernel_num] -> [batch_size, candidate_num, conv_kernel_num]
        news_candidates_encoded = news_candidates_encoded.view(
            batch_size, candidate_num, conv_kernel_num
        )

        # Encode user history
        # Shape: [batch_size, history_size, seq_len] -> [batch_size, conv_kernel_num]
        user_encoded = self.user_encoder(news_histories, self.news_encoder, attention_mask=news_histories_attention_mask)
        
        # Reshape for batch matrix multiplication
        # Shape: [batch_size, conv_kernel_num] -> [batch_size, conv_kernel_num, 1]
        user_encoded = user_encoded.unsqueeze(-1)
        
        # Compute scores via dot product
        # Shape: [batch_size, candidate_num, conv_kernel_num] x [batch_size, conv_kernel_num, 1]
        #     -> [batch_size, candidate_num, 1]
        output = torch.bmm(news_candidates_encoded, user_encoded)
        output = output.squeeze(-1)  # [batch_size, candidate_num]

        # During validation, don't compute loss
        if not self.training:
            return ModelOutput(logits=output, loss=torch.Tensor([-1]), labels=target)

        # Compute training loss
        loss = self.loss_fn(output, target)
        return ModelOutput(logits=output, loss=loss, labels=target)
