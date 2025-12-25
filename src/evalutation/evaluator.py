from ..utils.metrics import RecEvaluator

def evaluate_model(model, dataloader, device):
    evaluator = RecEvaluator()
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            for k in batch:
                batch[k] = batch[k].to(device)
            scores = model(batch)
            evaluator.add_batch(batch["labels"].cpu(), scores.cpu())
    return evaluator.compute()
