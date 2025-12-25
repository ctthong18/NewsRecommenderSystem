import torch
import torch.nn as nn

class NewsRecModel(nn.Module):
    def __init__(self, news_encoder, user_encoder, device="cpu"):
        super().__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.device = device

    def forward(self, batch):
        # batch: {candidate_news, user_history}
        cand_vecs = self.news_encoder(batch["cand_input_ids"], batch["cand_attention_mask"])
        hist_vecs = self.news_encoder(batch["hist_input_ids"], batch["hist_attention_mask"])
        user_vec = self.user_encoder(hist_vecs)
        scores = torch.matmul(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        return scores
