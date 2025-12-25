# Examples Directory

This directory contains example scripts demonstrating various features of the news recommendation pipeline.

## Available Examples

### tensorboard_example.py
Demonstrates real-time training monitoring with TensorBoard.

**Features:**
- Basic metric logging (loss, learning rate)
- Gradient monitoring and visualization
- Sample prediction logging
- Comparing multiple training runs
- Context manager usage

**Usage:**
```bash
# Run examples
python examples/tensorboard_example.py

# View in TensorBoard
tensorboard --logdir=output/tensorboard
# Then open http://localhost:6006
```

**Output:**
- TensorBoard logs in `output/tensorboard/`
- Multiple example runs demonstrating different features
- Real-time visualization of training metrics

**Requirements:**
- tensorboard>=2.14.0

**See Also:**
- `docs/TENSORBOARD_MONITORING.md` - TensorBoard integration guide

---

### test_llm_generation.py
Demonstrates LLM-based description generation for news articles.

**Features:**
- Multiple LLM provider support (OpenAI, Anthropic, Local models)
- Batch processing with rate limiting
- Cost estimation
- Error handling and retries

**Usage:**
```bash
python examples/test_llm_generation.py
```

**Requirements:**
- API keys for LLM providers (set in environment variables)
- See `docs/LLM_DESCRIPTION_GENERATION.md` for setup

---

### test_sampling.py
Demonstrates improved hard negative sampling functionality.

**Features:**
- Multiple sampling strategies (hardest, mixed, semi-hard)
- Statistics tracking and monitoring
- Performance benchmarking
- Visualization generation

**Usage:**
```bash
# Set environment variable to avoid OpenMP warning
$env:KMP_DUPLICATE_LIB_OK="TRUE"  # Windows PowerShell
# or
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac

python examples/test_sampling.py
```

**Output:**
- Console output with statistics
- JSON files in `output/sampling_logs/`
- Visualization PNGs (requires matplotlib)

**Requirements:**
- numpy
- matplotlib (optional, for visualizations)

---

### test_dataloader_optimization.py
Benchmarks and demonstrates data loading pipeline optimizations.

**Features:**
- Prefetching with multiple workers
- Pin memory optimization
- Optimized collate function
- Performance comparison between configurations

**Usage:**
```bash
python examples/test_dataloader_optimization.py
```

**Output:**
- Benchmark results comparing different dataloader configurations
- Performance metrics (throughput, batch time, speedup)

**Requirements:**
- MIND dataset (small version)
- torch
- transformers

**See Also:**
- `docs/DATA_LOADING_OPTIMIZATION.md` - Detailed optimization documentation

---

## Running Examples

### Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

For visualization features:
```bash
pip install matplotlib
```

### Environment Setup

Some examples require environment variables:

```bash
# For LLM generation
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# For avoiding OpenMP warnings
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Output Directories

Examples create output in:
- `output/sampling_logs/` - Sampling statistics and visualizations
- `output/llm_logs/` - LLM generation logs
- `Data/generated/` - Generated data files

## Documentation

For detailed documentation, see:
- `docs/SAMPLING_IMPROVEMENTS.md` - Sampling system documentation
- `docs/LLM_DESCRIPTION_GENERATION.md` - LLM generation documentation
- `docs/LLM_QUICK_START.md` - Quick start guide for LLM features
- `docs/DATA_LOADING_OPTIMIZATION.md` - Data loading optimization guide

## Troubleshooting

### OpenMP Warning
If you see "OMP: Error #15: Initializing libiomp5md.dll":
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"  # Windows
export KMP_DUPLICATE_LIB_OK=TRUE  # Linux/Mac
```

### Import Errors
Make sure you're running from the project root:
```bash
cd /path/to/project
python examples/test_sampling.py
```

### Missing Dependencies
Install missing packages:
```bash
pip install numpy matplotlib torch
```

## Contributing

When adding new examples:
1. Create a descriptive filename (e.g., `test_feature_name.py`)
2. Add comprehensive docstrings
3. Include usage examples in comments
4. Update this README
5. Add corresponding documentation in `docs/`


---

### evaluate_model_example.py
Demonstrates comprehensive model evaluation with statistical analysis.

**Features:**
- Basic model evaluation with comprehensive metrics
- Per-category performance analysis
- Statistical model comparison with significance testing
- Confidence intervals and effect size analysis

**Usage:**
```bash
# Example 1: Basic evaluation
python examples/evaluate_model_example.py --example 1

# Example 2: Per-category analysis
python examples/evaluate_model_example.py --example 2

# Example 3: Model comparison
python examples/evaluate_model_example.py --example 3
```

**Output:**
- Evaluation reports in `output/evaluation/`
- JSON format for programmatic access
- Text reports for human readability
- Statistical comparison results

**Requirements:**
- Trained model checkpoint(s)
- MIND dataset (validation set)
- scipy (for statistical tests)

**See Also:**
- `docs/EVALUATION_GUIDE.md` - Comprehensive evaluation documentation
- `docs/TASK_16_IMPLEMENTATION_SUMMARY.md` - Implementation details

---

### checkpoint_example.py
Demonstrates checkpoint management system.

**Features:**
- Save checkpoints with metadata
- Load checkpoints with validation
- Track best models
- Resume training from checkpoint

**Usage:**
```bash
python examples/checkpoint_example.py
```

**See Also:**
- `docs/CHECKPOINT_MANAGEMENT.md` - Checkpoint system documentation

---

### test_training_optimization.py
Demonstrates training optimization features.

**Features:**
- Learning rate scheduling (warmup + cosine decay)
- Gradient accumulation
- Mixed precision training (AMP)
- Performance benchmarking

**Usage:**
```bash
python examples/test_training_optimization.py
```

**See Also:**
- `docs/TRAINING_OPTIMIZATION.md` - Training optimization guide
- `docs/TRAINING_OPTIMIZATION_QUICK_START.md` - Quick start guide

---

### inference_example.py
Demonstrates inference pipeline for news recommendations.

**Features:**
- Batch inference on validation/test sets
- Single-user inference for real-time recommendations
- Model ensemble support
- Optimized inference speed

**Usage:**
```bash
# Batch inference
python examples/inference_example.py --example 1

# Single-user inference
python examples/inference_example.py --example 2

# Ensemble inference
python examples/inference_example.py --example 3
```

**See Also:**
- `docs/INFERENCE_GUIDE.md` - Comprehensive inference documentation
- `docs/INFERENCE_QUICK_START.md` - Quick start guide

---

### generate_submission_example.py
Demonstrates submission file generation for MIND leaderboard.

**Features:**
- End-to-end submission generation workflow
- Load model and run inference on test set
- Format predictions in MIND leaderboard format
- Generate submission.zip package

**Usage:**
```bash
python examples/generate_submission_example.py
```

**Interactive Menu:**
1. Generate submission for MINDsmall (testing)
2. Generate submission for MINDlarge (leaderboard)
3. Generate submission with custom checkpoint

**Output:**
- `prediction.txt` - Submission file in MIND format
- `prediction_metadata.json` - Submission metadata
- `submission.zip` - Complete submission package

**Requirements:**
- Trained model checkpoint
- MIND test dataset
- Optional: LLM descriptions

**See Also:**
- `docs/SUBMISSION_GENERATION.md` - Submission generation guide
- `docs/INFERENCE_GUIDE.md` - Inference pipeline documentation
