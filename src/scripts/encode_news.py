import torch
from transformers import AutoTokenizer, AutoModel
import json, argparse, os, sys
from tqdm import tqdm
import numpy as np
from pathlib import Path
import hashlib
from datetime import datetime
from typing import Dict, Set, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger, setup_logger


class EmbeddingCache:
    """
    Disk-based embedding cache with invalidation and statistics tracking.
    """
    
    def __init__(self, cache_path: str, model_name: str, max_length: int, logger=None):
        """
        Initialize embedding cache.
        
        Args:
            cache_path: Path to cache file
            model_name: Name of the model used for encoding
            max_length: Maximum sequence length
            logger: Logger instance
        """
        self.cache_path = cache_path
        self.model_name = model_name
        self.max_length = max_length
        self.logger = logger
        
        # Cache metadata
        self.metadata_path = cache_path.replace('.pt', '_metadata.json')
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'new_encodings': 0,
            'total_cached': 0
        }
        
        # Load existing cache
        self.embeddings = {}
        self.metadata = {}
        self._load_cache()
    
    def _compute_content_hash(self, text: str) -> str:
        """Compute hash of text content for cache invalidation."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_cache(self):
        """Load existing cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                self.embeddings = torch.load(self.cache_path)
                if self.logger:
                    self.logger.info(f"Loaded {len(self.embeddings)} embeddings from cache")
                self.stats['total_cached'] = len(self.embeddings)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load cache: {str(e)}, starting fresh")
                self.embeddings = {}
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    
                # Validate cache compatibility
                if not self._is_cache_valid():
                    if self.logger:
                        self.logger.warning("Cache invalidated due to model/config change")
                    self.embeddings = {}
                    self.metadata = {}
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load metadata: {str(e)}")
                self.metadata = {}
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is valid for current model and config."""
        if not self.metadata:
            return False
        
        return (self.metadata.get('model_name') == self.model_name and
                self.metadata.get('max_length') == self.max_length)
    
    def get(self, news_id: str, content: str) -> np.ndarray:
        """
        Get embedding from cache if available and valid.
        
        Args:
            news_id: News article ID
            content: News article content
            
        Returns:
            Cached embedding or None if not found/invalid
        """
        if news_id not in self.embeddings:
            self.stats['cache_misses'] += 1
            return None
        
        # Check if content has changed (cache invalidation)
        content_hash = self._compute_content_hash(content)
        cached_hash = self.metadata.get('content_hashes', {}).get(news_id)
        
        if cached_hash != content_hash:
            # Content changed, invalidate this entry
            if self.logger:
                self.logger.debug(f"Cache invalidated for {news_id} (content changed)")
            self.stats['cache_misses'] += 1
            return None
        
        self.stats['cache_hits'] += 1
        return self.embeddings[news_id]
    
    def put(self, news_id: str, content: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            news_id: News article ID
            content: News article content
            embedding: Embedding vector
        """
        self.embeddings[news_id] = embedding
        
        # Store content hash for invalidation
        if 'content_hashes' not in self.metadata:
            self.metadata['content_hashes'] = {}
        self.metadata['content_hashes'][news_id] = self._compute_content_hash(content)
        
        self.stats['new_encodings'] += 1
    
    def save(self):
        """Save cache and metadata to disk."""
        # Create directory if needed
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        
        # Save embeddings
        torch.save(self.embeddings, self.cache_path)
        
        # Update and save metadata
        self.metadata.update({
            'model_name': self.model_name,
            'max_length': self.max_length,
            'last_updated': datetime.now().isoformat(),
            'total_embeddings': len(self.embeddings)
        })
        
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        if self.logger:
            file_size = os.path.getsize(self.cache_path) / (1024 * 1024)
            self.logger.info(
                f"Cache saved: {len(self.embeddings)} embeddings ({file_size:.2f} MB)",
                num_embeddings=len(self.embeddings),
                file_size_mb=file_size
            )
    
    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'hit_rate_percent': hit_rate,
            'total_requests': total_requests
        }
    
    def get_missing_ids(self, news_desc: Dict[str, str]) -> Set[str]:
        """
        Get IDs of news articles that need encoding.
        
        Args:
            news_desc: Dictionary of news_id -> description
            
        Returns:
            Set of news IDs that are not in cache or have changed
        """
        missing = set()
        for news_id, content in news_desc.items():
            if self.get(news_id, content) is None:
                missing.add(news_id)
        return missing


def get_optimal_batch_size(model, tokenizer, device, sample_text: str, max_length: int, 
                          initial_batch_size: int, logger=None) -> int:
    """
    Automatically determine optimal batch size based on available GPU memory.
    
    Args:
        model: The model to test
        tokenizer: Tokenizer
        device: Device to test on
        sample_text: Sample text for testing
        max_length: Maximum sequence length
        initial_batch_size: Starting batch size
        logger: Logger instance
        
    Returns:
        Optimal batch size
    """
    if device.type == 'cpu':
        return initial_batch_size
    
    # Try progressively larger batch sizes
    batch_size = initial_batch_size
    max_batch_size = initial_batch_size * 4
    
    if logger:
        logger.info(f"Testing optimal batch size (starting from {initial_batch_size})")
    
    with torch.no_grad():
        while batch_size <= max_batch_size:
            try:
                # Test encoding with this batch size
                test_batch = [sample_text] * batch_size
                inputs = tokenizer(
                    test_batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length
                ).to(device)
                
                _ = model(**inputs).last_hidden_state[:,0,:]
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Success, try larger batch
                if logger:
                    logger.debug(f"Batch size {batch_size} successful")
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM, use previous successful batch size
                    optimal_size = max(batch_size // 2, initial_batch_size)
                    if logger:
                        logger.info(f"Optimal batch size determined: {optimal_size}")
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    return optimal_size
                else:
                    raise
    
    # If we got here, use the max tested size
    optimal_size = batch_size // 2
    if logger:
        logger.info(f"Optimal batch size determined: {optimal_size}")
    return optimal_size


def encode_all(args):
    """
    Encode all news articles using a pre-trained model with caching support and optimizations.
    
    Args:
        args: Command-line arguments
    """
    # Setup logger
    logger_instance = setup_logger(
        name="encode_news",
        log_dir=args.log_dir if hasattr(args, 'log_dir') and args.log_dir else None,
        log_level=args.log_level if hasattr(args, 'log_level') else "INFO",
        console_output=True
    )
    logger = logger_instance.get_logger("encode")
    
    logger.info("=" * 60)
    logger.info("Starting news encoding with caching and optimizations")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"News descriptions: {args.news_desc}")
    logger.info(f"Output path: {args.out_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Use cache: {args.use_cache}")
    logger.info(f"Force recompute: {args.force_recompute}")
    logger.info(f"Auto batch size: {args.auto_batch_size}")
    logger.info(f"Multi-GPU: {args.multi_gpu}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        logger.info("No GPU available, using CPU")
    
    # Load news descriptions
    logger.info(f"Loading news descriptions from {args.news_desc}")
    try:
        news_desc = json.load(open(args.news_desc, encoding="utf-8"))
        logger.info(f"Loaded {len(news_desc)} news descriptions", num_news=len(news_desc))
    except Exception as e:
        logger.error(f"Failed to load news descriptions: {str(e)}")
        raise
    
    # Initialize cache
    cache = None
    if args.use_cache and not args.force_recompute:
        cache = EmbeddingCache(args.out_path, args.model_name, args.max_length, logger)
        logger.info("Cache initialized")
        
        # Get missing IDs (incremental encoding)
        missing_ids = cache.get_missing_ids(news_desc)
        logger.info(
            f"Incremental encoding: {len(missing_ids)} new/changed, "
            f"{len(news_desc) - len(missing_ids)} cached",
            new_items=len(missing_ids),
            cached_items=len(news_desc) - len(missing_ids)
        )
        
        # Filter to only encode missing items
        news_to_encode = {nid: desc for nid, desc in news_desc.items() if nid in missing_ids}
    else:
        news_to_encode = news_desc
        logger.info(f"Encoding all {len(news_to_encode)} news articles (cache disabled or force recompute)")
    
    # If nothing to encode, just save existing cache
    if len(news_to_encode) == 0:
        logger.info("All embeddings are cached, nothing to encode")
        if cache:
            stats = cache.get_statistics()
            logger.info("Cache statistics:", **stats)
        return
    
    # Load model and tokenizer
    logger.info(f"Loading tokenizer and model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModel.from_pretrained(args.model_name)
        
        # Multi-GPU support
        if args.multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)
        
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    
    # Determine optimal batch size
    bs = args.batch_size
    if args.auto_batch_size and torch.cuda.is_available():
        # Use first news description as sample
        sample_text = next(iter(news_to_encode.values())) if news_to_encode else "Sample text for testing"
        bs = get_optimal_batch_size(model, tokenizer, device, sample_text, args.max_length, bs, logger)
        logger.info(f"Using optimized batch size: {bs}")
    
    # Encode news
    batch = []
    ids = []
    total_batches = (len(news_to_encode) + bs - 1) // bs
    
    logger.info(f"Starting encoding with {total_batches} batches")
    
    # GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared before encoding")
    
    with torch.no_grad():
        batch_count = 0
        for nid, desc in tqdm(news_to_encode.items(), desc="Encoding news"):
            ids.append(nid)
            batch.append(desc)
            
            if len(batch) >= bs:
                try:
                    inputs = tokenizer(
                        batch, 
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt", 
                        max_length=args.max_length
                    ).to(device)
                    out = model(**inputs).last_hidden_state[:,0,:]  # CLS pooling
                    
                    # Store in cache or dict
                    for i, nid_ in enumerate(ids):
                        embedding = out[i].cpu().numpy()
                        if cache:
                            cache.put(nid_, news_desc[nid_], embedding)
                        else:
                            if not hasattr(encode_all, 'emb_dict'):
                                encode_all.emb_dict = {}
                            encode_all.emb_dict[nid_] = embedding
                    
                    batch, ids = [], []
                    batch_count += 1
                    
                    # Periodic GPU memory cleanup
                    if torch.cuda.is_available() and batch_count % 100 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"GPU OOM at batch {batch_count}. Try reducing batch size.")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise
                    else:
                        raise
        
        # Process last batch
        if batch:
            logger.debug(f"Processing final batch with {len(batch)} items")
            try:
                inputs = tokenizer(
                    batch, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=args.max_length
                ).to(device)
                out = model(**inputs).last_hidden_state[:,0,:]
                
                for i, nid_ in enumerate(ids):
                    embedding = out[i].cpu().numpy()
                    if cache:
                        cache.put(nid_, news_desc[nid_], embedding)
                    else:
                        if not hasattr(encode_all, 'emb_dict'):
                            encode_all.emb_dict = {}
                        encode_all.emb_dict[nid_] = embedding
                        
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("GPU OOM on final batch. Try reducing batch size.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                else:
                    raise
    
    # Final GPU cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared after encoding")
    
    # Save embeddings
    logger.info(f"Saving embeddings to {args.out_path}")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    try:
        if cache:
            cache.save()
            stats = cache.get_statistics()
            logger.info("Cache statistics:", **stats)
            logger.info(
                f"Cache hit rate: {stats['hit_rate_percent']:.1f}%",
                cache_hits=stats['cache_hits'],
                cache_misses=stats['cache_misses'],
                new_encodings=stats['new_encodings']
            )
        else:
            emb_dict = getattr(encode_all, 'emb_dict', {})
            torch.save(emb_dict, args.out_path)
            file_size = os.path.getsize(args.out_path) / (1024 * 1024)  # MB
            logger.info(
                f"Successfully saved embeddings ({file_size:.2f} MB)",
                num_embeddings=len(emb_dict),
                file_size_mb=file_size
            )
    except Exception as e:
        logger.error(f"Failed to save embeddings: {str(e)}")
        raise
    
    logger.info("=" * 60)
    logger.info("News encoding completed successfully!")
    logger.info("=" * 60)
    print("Saved news embeddings to", args.out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode news articles using pre-trained models with caching and optimizations")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base", help="Pre-trained model name")
    parser.add_argument("--news_desc", required=True, help="Path to news descriptions JSON file")
    parser.add_argument("--out_path", default="outputs/news_embeddings.pt", help="Output path for embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Initial batch size for encoding")
    parser.add_argument("--max_length", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Use disk-based cache (default: True)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable caching")
    parser.add_argument("--force-recompute", action="store_true", help="Force recompute all embeddings, ignoring cache")
    parser.add_argument("--auto-batch-size", action="store_true", help="Automatically determine optimal batch size")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs if available (DataParallel)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for log files")
    args = parser.parse_args()
    encode_all(args)
