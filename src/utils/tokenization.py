from typing import Callable, Union, Dict

import torch
from transformers import PreTrainedTokenizer


def create_transform_fn_from_pretrained_tokenizer(
    tokenizer: PreTrainedTokenizer, max_length: int, padding: bool = True
) -> Callable[[list[str | list[str]]], Dict[str, torch.Tensor]]:
    """
    Create a transform function that tokenizes text.
    Supports both single text strings and lists of [title, description] pairs.
    
    Args:
        tokenizer: Pre-trained tokenizer
        max_length: Maximum sequence length
        padding: Whether to pad sequences
    
    Returns:
        Transform function that takes list of texts and returns dict with input_ids and attention_mask
    """
    def transform(texts: list[Union[str, list[str]]]) -> Dict[str, torch.Tensor]:
        # If texts is list of [title, description] pairs, combine them
        processed_texts = []
        for text in texts:
            if isinstance(text, list):
                # Combine title and description with separator
                title = text[0] if len(text) > 0 else ""
                description = text[1] if len(text) > 1 else ""
                combined = f"{title} [SEP] {description}".strip()
                processed_texts.append(combined)
            else:
                processed_texts.append(text)
        
        # Tokenize with attention_mask
        encoded = tokenizer(
            processed_texts,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=True,
            return_attention_mask=True
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"]
        }

    return transform
