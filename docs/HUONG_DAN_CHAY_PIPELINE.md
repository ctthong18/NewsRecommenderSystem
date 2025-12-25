# Hướng Dẫn Chạy Pipeline Huấn Luyện

## Tổng Quan

Pipeline này bao gồm 4 bước chính:
1. **Download dataset** - Tải MIND dataset
2. **Preprocess** - Tiền xử lý dữ liệu
3. **Generate LLM descriptions** - Tạo mô tả bằng LLM (tùy chọn)
4. **Train model** - Huấn luyện mô hình

## Cài Đặt

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Nếu dùng LLM (OpenAI/Anthropic)
pip install openai anthropic
```

## Cách Chạy Pipeline Hoàn Chỉnh

### Bước 1: Download MIND Dataset

```bash
# Download dataset nhỏ (khuyến nghị cho test)
python -m src.scripts.download_mind --size small --output-dir Data/raw

# Hoặc dataset lớn (cho production)
python -m src.scripts.download_mind --size large --output-dir Data/raw
```

**Output**: Các file sẽ được lưu trong `Data/raw/MINDsmall_train/` và `Data/raw/MINDsmall_dev/`

### Bước 2: Preprocess Dữ Liệu

```bash
# Preprocess training data
python -m src.scripts.preprocess \
  --news Data/raw/MINDsmall_train/news.tsv \
  --behaviors Data/raw/MINDsmall_train/behaviors.tsv \
  --out_dir Data/processed/train

# Preprocess validation data
python -m src.scripts.preprocess \
  --news Data/raw/MINDsmall_dev/news.tsv \
  --behaviors Data/raw/MINDsmall_dev/behaviors.tsv \
  --out_dir Data/processed/dev
```

**Output**: 
- `Data/processed/train/news_meta.json`
- `Data/processed/train/impressions.json`
- `Data/processed/dev/news_meta.json`
- `Data/processed/dev/impressions.json`

### Bước 3: Generate LLM Descriptions (Tùy Chọn)

#### Option 1: Dùng Local (Không cần API key, miễn phí)

```bash
python -m src.scripts.gen_descriptions \
  --news_meta Data/processed/train/news_meta.json \
  --provider local \
  --out_dir Data/generated
```

#### Option 2: Dùng OpenAI GPT-4

```bash
# Set API key
export OPENAI_API_KEY="sk-your-api-key-here"

# Generate descriptions
python -m src.scripts.gen_descriptions \
  --news_meta Data/processed/train/news_meta.json \
  --provider openai \
  --model gpt-4o-mini \
  --out_dir Data/generated
```

#### Option 3: Dùng Anthropic Claude

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-your-api-key-here"

# Generate descriptions
python -m src.scripts.gen_descriptions \
  --news_meta Data/processed/train/news_meta.json \
  --provider anthropic \
  --model claude-3-haiku-20240307 \
  --out_dir Data/generated
```

#### Option 4: Dùng Ollama (Local LLM)

```bash
# Cần cài Ollama trước: https://ollama.ai
# Pull model: ollama pull llama2

python -m src.scripts.gen_descriptions \
  --news_meta Data/processed/train/news_meta.json \
  --provider ollama \
  --model llama2 \
  --out_dir Data/generated
```

**Output**: `Data/generated/news_descriptions.json`

### Bước 4: Train Model

#### Tạo Config File

Tạo file `configs/my_config.yaml`:

```yaml
# Data paths
data:
  train_news: "Data/raw/MINDsmall_train/news.tsv"
  train_behaviors: "Data/raw/MINDsmall_train/behaviors.tsv"
  val_news: "Data/raw/MINDsmall_dev/news.tsv"
  val_behaviors: "Data/raw/MINDsmall_dev/behaviors.tsv"
  llm_description: "Data/generated/news_descriptions.json"  # Hoặc null nếu không dùng LLM

# Model settings
model:
  pretrained: "microsoft/deberta-v3-base"
  conv_kernel_num: 400
  query_dim: 200
  max_length: 64

# Training settings
training:
  batch_size: 8
  lr: 2e-5
  epochs: 3
  npratio: 4
  history_size: 50
  num_workers: 4
  
  # Optimizations
  use_scheduler: true
  scheduler_type: "cosine"
  warmup_ratio: 0.1
  gradient_accumulation_steps: 4
  use_mixed_precision: true
  
  # Checkpoint settings
  keep_last_n_checkpoints: 3
  metric_for_best_model: "ndcg_at_10"
  resume_from_checkpoint: false
  early_stopping_patience: 3

# Output paths
paths:
  checkpoint_dir: "output/checkpoints"
  output_dir: "output/models"
  log_dir: "output/logs"
```

#### Chạy Training

```bash
# Train với config file
python3 train.py --config configs/gpu_48gb_large.yaml

# Hoặc override một số parameters
python train.py --config configs/my_config.yaml --override \
  training.batch_size=16 \
  training.epochs=5 \
  training.lr=1e-5
```

**Output**:
- Checkpoints: `output/checkpoints/checkpoint_epoch_*.pt`
- Best model: `output/checkpoints/best_model.pt`
- Final model: `output/models/deberta_naml_final.pt`
- Logs: `output/logs/`

## Pipeline Đầy Đủ - Một Lệnh

Nếu muốn chạy tất cả các bước liên tiếp:

```bash
# 1. Download
python -m src.scripts.download_mind --size small

# 2. Preprocess train
python -m src.scripts.preprocess \
  --news Data/raw/MINDsmall_train/news.tsv \
  --behaviors Data/raw/MINDsmall_train/behaviors.tsv \
  --out_dir Data/processed/train

# 3. Preprocess dev
python -m src.scripts.preprocess \
  --news Data/raw/MINDsmall_dev/news.tsv \
  --behaviors Data/raw/MINDsmall_dev/behaviors.tsv \
  --out_dir Data/processed/dev

# 4. Generate LLM descriptions (local - nhanh và miễn phí)
python -m src.scripts.gen_descriptions \
  --news_meta Data/processed/train/news_meta.json \
  --provider local \
  --out_dir Data/generated

# 5. Train
python3 train.py --config configs/gpu_48gb_large.yaml
```

## Script Tự Động (Bash)

Tạo file `run_full_pipeline.sh`:

```bash
#!/bin/bash
set -e  # Exit on error

echo "=== Step 1: Download Dataset ==="
python -m src.scripts.download_mind --size small

echo "=== Step 2: Preprocess Training Data ==="
python -m src.scripts.preprocess \
  --news Data/raw/MINDsmall_train/news.tsv \
  --behaviors Data/raw/MINDsmall_train/behaviors.tsv \
  --out_dir Data/processed/train

echo "=== Step 3: Preprocess Validation Data ==="
python -m src.scripts.preprocess \
  --news Data/raw/MINDsmall_dev/news.tsv \
  --behaviors Data/raw/MINDsmall_dev/behaviors.tsv \
  --out_dir Data/processed/dev

echo "=== Step 4: Generate LLM Descriptions ==="
python -m src.scripts.gen_descriptions \
  --news_meta Data/processed/train/news_meta.json \
  --provider local \
  --out_dir Data/generated

echo "=== Step 5: Train Model ==="
python train.py --config configs/my_config.yaml

echo "=== Pipeline Complete! ==="
```

Chạy script:

```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

## Script Tự Động (Windows - PowerShell)

Tạo file `run_full_pipeline.ps1`:

```powershell
Write-Host "=== Step 1: Download Dataset ===" -ForegroundColor Green
python -m src.scripts.download_mind --size small

Write-Host "=== Step 2: Preprocess Training Data ===" -ForegroundColor Green
python -m src.scripts.preprocess `
  --news Data/raw/MINDsmall_train/news.tsv `
  --behaviors Data/raw/MINDsmall_train/behaviors.tsv `
  --out_dir Data/processed/train

Write-Host "=== Step 3: Preprocess Validation Data ===" -ForegroundColor Green
python -m src.scripts.preprocess `
  --news Data/raw/MINDsmall_dev/news.tsv `
  --behaviors Data/raw/MINDsmall_dev/behaviors.tsv `
  --out_dir Data/processed/dev

Write-Host "=== Step 4: Generate LLM Descriptions ===" -ForegroundColor Green
python -m src.scripts.gen_descriptions `
  --news_meta Data/processed/train/news_meta.json `
  --provider local `
  --out_dir Data/generated

Write-Host "=== Step 5: Train Model ===" -ForegroundColor Green
python train.py --config configs/my_config.yaml

Write-Host "=== Pipeline Complete! ===" -ForegroundColor Green
```

Chạy script:

```powershell
.\run_full_pipeline.ps1
```

## Các Tùy Chọn Nâng Cao

### Skip LLM Generation (Nhanh hơn)

Nếu không muốn dùng LLM descriptions, set `llm_description: null` trong config:

```yaml
data:
  llm_description: null  # Không dùng LLM
```

### Resume Training Khi Bị Gián Đoạn

```bash
python train.py --config configs/my_config.yaml --override \
  training.resume_from_checkpoint=true
```

### Giảm Memory Usage

```bash
python train.py --config configs/my_config.yaml --override \
  training.batch_size=2 \
  training.gradient_accumulation_steps=8 \
  training.use_mixed_precision=true
```

### Tăng Tốc Training

```bash
python train.py --config configs/my_config.yaml --override \
  training.use_mixed_precision=true \
  training.gradient_accumulation_steps=4 \
  training.num_workers=8
```

## Kiểm Tra Kết Quả

Sau khi training xong:

```bash
# Xem logs
cat output/logs/training.log

# Xem checkpoints
ls -lh output/checkpoints/

# Xem TensorBoard (nếu enabled)
tensorboard --logdir output/tensorboard
```

## Troubleshooting

### Lỗi Out of Memory

```bash
# Giảm batch size
python train.py --config configs/my_config.yaml --override training.batch_size=2

# Hoặc enable mixed precision
python train.py --config configs/my_config.yaml --override training.use_mixed_precision=true
```

### Lỗi LLM API

```bash
# Dùng local thay vì API
python -m src.scripts.gen_descriptions \
  --news_meta Data/processed/train/news_meta.json \
  --provider local \
  --out_dir Data/generated
```

### File Không Tìm Thấy

Kiểm tra đường dẫn trong config file và đảm bảo đã chạy đủ các bước trước đó.

## Thời Gian Ước Tính

Với MIND Small dataset trên máy có GPU:

- Download: ~5 phút
- Preprocess: ~2 phút
- LLM Generation (local): ~5 phút
- LLM Generation (OpenAI): ~30 phút (tùy rate limit)
- Training (3 epochs): ~1-2 giờ

**Tổng**: ~1.5-2.5 giờ (với local LLM)

## Kết Luận

Đây là pipeline đầy đủ để train model từ đầu. Bạn có thể:
- Chạy từng bước riêng lẻ để debug
- Dùng script tự động để chạy toàn bộ
- Tùy chỉnh config theo nhu cầu
- Skip LLM generation nếu muốn nhanh hơn
