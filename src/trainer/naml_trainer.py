import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer

class NAMLTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, device, evaluator=None):
        super().__init__(model, optimizer, loss_fn, device, evaluator)

    def train_epoch(self, dataloader, epoch=0, log_interval=100):
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            news_vecs = batch["news_vecs"].to(self.device)         # (B, num_news, emb_dim)
            user_hist = batch["user_hist"].to(self.device)         # (B, hist_len, emb_dim)
            labels = batch["labels"].to(self.device)               # (B,)

            self.optimizer.zero_grad()
            scores = self.model(news_vecs, user_hist)              # (B,)
            loss = self.loss_fn(scores, labels.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if (step + 1) % log_interval == 0:
                avg = total_loss / (step + 1)
                print(f"Step {step+1}: loss={avg:.4f}")

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                news_vecs = batch["news_vecs"].to(self.device)
                user_hist = batch["user_hist"].to(self.device)
                labels = batch["labels"].to(self.device)
                scores = self.model(news_vecs, user_hist)
                preds = torch.sigmoid(scores).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        if self.evaluator:
            metrics = self.evaluator.evaluate(all_preds, all_labels)
            print("Validation metrics:", metrics)
            return metrics
        else:
            return {}
