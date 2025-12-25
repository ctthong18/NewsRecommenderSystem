# src/recommendation/trainer/base_trainer.py
import torch
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, device, evaluator=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.evaluator = evaluator

    def train_epoch(self, dataloader, epoch=0, log_interval=100):
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            inputs, labels = batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = self.loss_fn(outputs.squeeze(-1), labels.float())
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
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                outputs = self.model(**inputs)
                preds = torch.sigmoid(outputs).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy().tolist())

        if self.evaluator:
            metrics = self.evaluator.evaluate(all_preds, all_labels)
            print("Validation metrics:", metrics)
            return metrics
        else:
            return {}
