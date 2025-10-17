import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class ONCEEncoder(nn.Module):
    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        entity_dim: int = 100,
        hidden_dim: int = 512,
        output_dim: int = 256,
        device: str = None,
    ):
        super().__init__()
        print(f"[ONCEEncoder] Initializing with model: {text_model_name}")

        # Load pre-trained sentence model
        self.text_encoder = SentenceTransformer(text_model_name)

        d_text = self.text_encoder.get_sentence_embedding_dimension()
        print(f"[ONCEEncoder] Text embedding dimension: {d_text}")

        self.fusion = nn.Sequential(
            nn.Linear(d_text + entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.to(self.device)
        print(f"[ONCEEncoder] Running on device: {self.device}")

    def forward(self, texts, entity_vecs):
        # Encode text bằng SentenceTransformer (trả về numpy array)
        text_emb = self.text_encoder.encode(texts, convert_to_tensor=True, device=self.device)

        if not isinstance(entity_vecs, torch.Tensor):
            entity_vecs = torch.tensor(entity_vecs, dtype=torch.float32, device=self.device)

        fused = torch.cat([text_emb, entity_vecs], dim=1)
        output = self.fusion(fused)

        return output

    @torch.no_grad()
    def encode_batch(self, texts, entity_vecs):
        self.eval()
        return self.forward(texts, entity_vecs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)


if __name__ == "__main__":
    encoder = ONCEEncoder(text_model_name="all-MiniLM-L6-v2", entity_dim=100)
    sample_texts = ["Microsoft launches new AI model", "OpenAI releases GPT-5"]
    sample_entities = torch.randn(2, 100)

    out = encoder.encode_batch(sample_texts, sample_entities)
    print("Output shape:", out.shape)
