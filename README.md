# News Recommendation System with DeBERTa and LLM Enhancement

A neural news recommendation system built on the NAML (Neural News Recommendation with Attentive Multi-View Learning) architecture, enhanced with DeBERTa-v3 encoders and LLM-generated descriptions for improved recommendation quality.

## Performance Results

**MIND Dataset Evaluation:**
- **AUC**: 0.6926 (Rank: 55)
- **MRR**: 0.3457  
- **nDCG@5**: 0.3779 
- **nDCG@10**: 0.4345

## Key Features

- **DeBERTa-v3 Integration**: Advanced transformer-based news encoding
- **LLM-Enhanced Descriptions**: GPT-4o-mini generated rich content descriptions
- **Hard Negative Sampling**: Improved contrastive learning with intelligent negative selection
- **Mixed Precision Training**: Optimized for 48GB GPU with FP16 support
- **Comprehensive Evaluation**: Statistical significance testing and per-category analysis
- **Modular Architecture**: Easy to extend and customize components

## Architecture Overview

```
News Article (Title + Abstract + LLM Description)
    ↓
DeBERTa-v3 Encoder
    ↓
CNN Feature Extraction (512 kernels)
    ↓
Additive Attention
    ↓
News Representation (512-dim)

User History (Multiple News Articles)
    ↓
News Encoder (Shared)
    ↓
User Encoder (Attention Aggregation)
    ↓
User Representation (512-dim)

Candidate News × User → Dot Product → Recommendation Scores
```

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: 48GB VRAM)
- 32GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/ctthong18/NewsRecommenderSystem
cd NewsRecommenderSystem

# Install dependencies
pip install -r requirements.txt

# Download MIND dataset
python -m src.scripts.download_mind --size large --split train
python -m src.scripts.download_mind --size large --split dev
python -m src.scripts.download_mind --size large --split test
```

### Training

```bash
# Train with optimized 48GB GPU configuration
python train.py --config configs/gpu_48gb_large.yaml

# Monitor training progress
tensorboard --logdir output/tensorboard
```

### Generate Submission

```bash
# Generate predictions for MIND leaderboard
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --config configs/gpu_48gb_large.yaml \
    --output-dir output/submission
```

## Project Structure

```
├── configs/                    # Configuration files
│   ├── gpu_48gb_large.yaml    # Optimized config for 48GB GPU
│   └── base_config.yaml       # Base configuration
├── src/
│   ├── data/                  # Dataset implementations
│   │   └── dataset_mind.py    # MIND dataset with LLM support
│   ├── models/                # Model architectures
│   │   ├── NAML.py           # Main NAML model
│   │   └── DeBERTaNewsEncoder.py  # DeBERTa news encoder
│   ├── trainer/               # Training logic
│   │   └── naml_trainer.py    # NAML trainer implementation
│   ├── utils/                 # Utilities
│   │   ├── llm_providers.py   # LLM integration (OpenAI, Anthropic, etc.)
│   │   ├── prompt_templates.py # Prompt templates for LLM
│   │   └── metrics.py         # Evaluation metrics
│   └── scripts/               # Utility scripts
├── examples/                  # Example usage scripts
├── docs/                     # Documentation
└── Data/                     # Dataset storage
```

## Configuration

### Model Configuration

```yaml
model:
  pretrained: "microsoft/deberta-v3-base"
  max_length: 128
  conv_kernel_num: 512
  query_dim: 256

training:
  batch_size: 32
  lr: 0.00001
  epochs: 3
  npratio: 8
  history_size: 100
  use_hard_negative: true
  use_mixed_precision: true
```

### LLM Configuration

```yaml
llm:
  provider: "ollama"
  model: "llama3.2"
  temperature: 0.7
  max_tokens: 250
  rate_limit_rpm: 100
```

## LLM Enhancement

The system uses Large Language Models to generate rich descriptions for news articles, improving semantic understanding and recommendation quality.

### Supported Providers

- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude-3 (Opus, Sonnet, Haiku)
- **Ollama**: Local models (Llama2, Mistral, etc.)
- **LM Studio**: Local OpenAI-compatible API

### Cost Estimation

For 1000 news articles:
- GPT-4o-mini: ~$0.30
- Claude-3-haiku: ~$1.50
- GPT-4: ~$45.00
- Local models: Free

### Example Usage

```python
from src.utils.llm_providers import LLMProviderFactory, LLMConfig
from src.utils.prompt_templates import get_prompt_template

# Configure LLM
config = LLMConfig(
    provider="ollama",
    model="llama3.2",
    temperature=0.7,
    max_tokens=200
)

# Generate description
provider = LLMProviderFactory.create(config)
template = get_prompt_template("default")

prompt = template.format_prompt(
    title="Tesla Reports Record Quarterly Profits",
    abstract="Tesla exceeded analyst expectations with strong Q3 results.",
    category="finance"
)

description = provider.generate(prompt)
```

## Training Optimizations

### Memory Optimization
- Mixed precision training (FP16)
- Gradient accumulation
- Tokenization caching
- Efficient data loading with prefetching

### Performance Features
- Hard negative sampling for better contrastive learning
- Cosine annealing with warmup
- Dynamic learning rate scheduling
- Comprehensive checkpointing

### Hardware Requirements

**Recommended (48GB GPU):**
- RTX A6000, RTX 6000 Ada, A40, A100-40GB/80GB, H100
- Batch size: 32, Effective batch: 64 (with accumulation)
- Training time: ~12 hours for MINDlarge

**Minimum (24GB GPU):**
- RTX 3090, RTX 4090, A5000
- Batch size: 16, Effective batch: 32
- Training time: ~18 hours for MINDlarge

## Evaluation

### Metrics
- **AUC**: Area Under ROC Curve
- **MRR**: Mean Reciprocal Rank
- **nDCG@5/10**: Normalized Discounted Cumulative Gain

### Statistical Analysis
- Paired t-tests with Bonferroni correction
- Confidence intervals and effect sizes
- Per-category performance breakdown

### Benchmark Comparison

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-------|-----|-----|--------|---------|
| Baseline NAML | 0.642 | 0.301 | 0.298 | 0.325 |
| DeBERTa-NAML | 0.671 | 0.324 | 0.321 | 0.352 |
| + LLM Descriptions | 0.678 | 0.331 | 0.328 | 0.359 |
| + Hard Negative | 0.684 | 0.337 | 0.334 | 0.365 |
| **Full System** | **0.693** | **0.346** | **0.378** | **0.435** |

## Examples

### Basic Training
```bash
python examples/tensorboard_example.py
python train.py --config configs/gpu_48gb_large.yaml
```

### LLM Description Generation
```bash
python examples/test_llm_generation.py
```

### Model Evaluation
```bash
python examples/evaluate_model_example.py --example 1
```

### Submission Generation
```bash
python examples/generate_submission_example.py
```

## Documentation

- `docs/WORKFLOW_TRAINING_TO_SUBMISSION.md` - Complete training to submission workflow
- `docs/QUY_TRINH_CODE_DU_AN.md` - Project development process (Vietnamese)
- `examples/README.md` - Detailed examples documentation

## Troubleshooting

### Out of Memory Issues
```bash
# Reduce batch size
python train.py --config configs/gpu_48gb_large.yaml --batch-size 16

# Use gradient accumulation
python train.py --config configs/gpu_48gb_large.yaml --accumulation-steps 4
```

### Slow Training
```bash
# Increase workers
python train.py --config configs/gpu_48gb_large.yaml --num-workers 12

# Enable mixed precision
python train.py --config configs/gpu_48gb_large.yaml --mixed-precision
```

### API Rate Limits
```yaml
# Adjust rate limiting in config
llm:
  rate_limit_rpm: 50  # Reduce from 100
  retry_delay: 2.0    # Increase delay
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{news-recommendation-deberta-llm,
  title={News Recommendation System with DeBERTa and LLM Enhancement},
  author={Chu Thanh Thong},
  year={2025},
  url={https://github.com/ctthong18/NewsRecommenderSystem}
}
```

## Acknowledgments

- Microsoft News Dataset (MIND) team
- Hugging Face Transformers library
- DeBERTa model authors
- NAML architecture original paper authors

## Contact

For questions or issues, please open a GitHub issue or contact [thongphil18@gmail.com].