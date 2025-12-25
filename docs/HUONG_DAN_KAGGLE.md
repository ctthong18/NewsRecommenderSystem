# Hướng Dẫn Chuẩn Bị Project Cho Kaggle

## 1. Các File/Folder CẦN THIẾT Để Training

### 1.1. Code và Config (BẮT BUỘC)
```
├── train.py                          # Script training chính
├── requirements.txt                  # Dependencies
├── configs/
│   ├── my_config.yaml               # Config tối ưu cho GPU 16GB
│   └── colab_xlarge_config.yaml     # Config cho GPU lớn hơn
├── src/                             # Toàn bộ source code
│   ├── models/                      # Model architectures
│   ├── data/                        # Dataset và dataloader
│   ├── utils/                       # Utilities
│   └── trainer/                     # Training logic
```

### 1.2. Data (BẮT BUỘC)
```
├── Data/
│   ├── raw/
│   │   ├── MINDlarge_train/         # Training data
│   │   │   ├── news.tsv
│   │   │   └── behaviors.tsv
│   │   └── MINDlarge_dev/           # Validation data
│   │       ├── news.tsv
│   │       └── behaviors.tsv
│   └── generated/
│       └── news_descriptions.json    # LLM descriptions (nếu có)
```

### 1.3. Các Folder KHÔNG CẦN (có thể xóa để giảm dung lượng)
```
❌ .git/                    # Git history (rất nặng)
❌ .cache/                  # Cache files
❌ .mypy_cache/             # Type checking cache
❌ .pytest_cache/           # Test cache
❌ output/                  # Training outputs (sẽ tạo mới)
❌ notebooks/               # Jupyter notebooks
❌ examples/                # Example scripts
❌ docs/                    # Documentation
❌ test/                    # Unit tests
❌ gpt-augmented-news-recommendation/  # Old code
❌ Data/external/           # External data
❌ Data/first/              # Old data
❌ Data/processed/          # Processed data (nếu không dùng)
❌ Data/split/              # Split data (nếu không dùng)
❌ *.ipynb                  # Jupyter notebooks
```

## 2. Script Tự Động Nén Project

Tạo file `prepare_for_kaggle.py`:

```python
import os
import shutil
import zipfile
from pathlib import Path

def prepare_kaggle_package(output_name="kaggle_training_package.zip"):
    """
    Tạo package tối ưu cho Kaggle training.
    """
    print("🚀 Chuẩn bị package cho Kaggle...")
    
    # Các folder/file CẦN THIẾT
    include_patterns = [
        "train.py",
        "requirements.txt",
        "configs/my_config.yaml",
        "configs/colab_xlarge_config.yaml",
        "configs/base_config.yaml",
        "src/**/*.py",  # Tất cả Python files trong src
        "Data/raw/MINDlarge_train/news.tsv",
        "Data/raw/MINDlarge_train/behaviors.tsv",
        "Data/raw/MINDlarge_dev/news.tsv",
        "Data/raw/MINDlarge_dev/behaviors.tsv",
        "Data/generated/news_descriptions.json",  # Nếu có
    ]
    
    # Tạo temporary folder
    temp_dir = Path("temp_kaggle_package")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Copy các file cần thiết
    print("📦 Đang copy files...")
    for pattern in include_patterns:
        if "**" in pattern:
            # Handle recursive patterns
            base_path = pattern.split("**")[0]
            for file_path in Path(".").glob(pattern):
                if file_path.is_file():
                    dest = temp_dir / file_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest)
                    print(f"  ✓ {file_path}")
        else:
            file_path = Path(pattern)
            if file_path.exists():
                dest = temp_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                if file_path.is_file():
                    shutil.copy2(file_path, dest)
                else:
                    shutil.copytree(file_path, dest)
                print(f"  ✓ {file_path}")
            else:
                print(f"  ⚠️  Không tìm thấy: {file_path}")
    
    # Tạo README cho Kaggle
    readme_content = """# News Recommendation Training Package

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start training:
```bash
# For GPU with 16GB VRAM
python train.py --config configs/my_config.yaml

# For GPU with 24GB+ VRAM
python train.py --config configs/colab_xlarge_config.yaml

# For GPU with 48GB VRAM (NEW!)
python train.py --config configs/gpu_48gb_large.yaml
```

3. Monitor training:
```bash
tensorboard --logdir output/tensorboard
```

4. Generate submission (after training):
```bash
# Automatic workflow: Training → Submission
python scripts/full_workflow.py --mode train_and_submit --config configs/gpu_48gb_large.yaml

# Or manual submission generation
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best_model.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --config configs/gpu_48gb_large.yaml
```

## Configuration

Edit config files in `configs/` to adjust:
- Batch size
- Learning rate
- Number of epochs
- Hard negative sampling
- Mixed precision training

## Output

Training outputs will be saved to:
- `output/checkpoints/` - Model checkpoints
- `output/tensorboard/` - TensorBoard logs
- `output/models/` - Final model
"""
    
    (temp_dir / "README.md").write_text(readme_content)
    print("  ✓ README.md")
    
    # Tạo ZIP file
    print(f"\n📦 Đang nén thành {output_name}...")
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    # Thống kê
    file_size = os.path.getsize(output_name) / (1024 * 1024)  # MB
    print(f"\n✅ Hoàn thành!")
    print(f"📦 File: {output_name}")
    print(f"💾 Dung lượng: {file_size:.2f} MB")
    print(f"\n🚀 Upload file này lên Kaggle Dataset và bắt đầu training!")

if __name__ == "__main__":
    prepare_kaggle_package()
```

## 3. Các Bước Thực Hiện

### Bước 1: Chạy script chuẩn bị
```bash
python prepare_for_kaggle.py
```

Script sẽ tạo file `kaggle_training_package.zip` (~100-500MB tùy data)

### Bước 2: Upload lên Kaggle

1. Vào https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload file `kaggle_training_package.zip`
4. Đặt tên dataset (ví dụ: "news-recommendation-training")
5. Click "Create"

### Bước 3: Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings → Accelerator → chọn **GPU T4** (miễn phí, 16GB VRAM)
4. Add Data → chọn dataset vừa upload

### Bước 4: Setup trong Kaggle Notebook

```python
# Cell 1: Extract và setup
!unzip -q /kaggle/input/news-recommendation-training/kaggle_training_package.zip -d /kaggle/working/
%cd /kaggle/working

# Cell 2: Install dependencies
!pip install -q -r requirements.txt

# Cell 3: Verify setup
!ls -lh Data/raw/MINDlarge_train/
!python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Cell 4: Start training
!python train.py --config configs/my_config.yaml
```

## 4. Tối Ưu Cho Kaggle

### 4.1. Config cho GPU T4 (16GB)
File `configs/my_config.yaml` đã được tối ưu:
- Batch size: 8
- Gradient accumulation: 2 (effective batch = 16)
- Mixed precision: True
- Num workers: 2

### 4.2. Config cho GPU P100 (16GB)
Tương tự `my_config.yaml`

### 4.3. Config cho GPU V100/A100 (32GB+)
File `configs/colab_xlarge_config.yaml`:
- Batch size: 16
- Gradient accumulation: 2 (effective batch = 32)
- Mixed precision: True
- Num workers: 4

## 5. Monitoring Training

### Trong Kaggle Notebook:
```python
# Xem logs realtime
!tail -f output/training.log

# Xem checkpoints
!ls -lh output/checkpoints/

# Load TensorBoard (nếu cần)
%load_ext tensorboard
%tensorboard --logdir output/tensorboard
```

## 6. Download Trained Model

Sau khi training xong:

```python
# Nén model để download
!zip -r trained_model.zip output/checkpoints/ output/models/

# Download qua Kaggle UI hoặc:
from IPython.display import FileLink
FileLink('trained_model.zip')
```

## 7. Troubleshooting

### Lỗi Out of Memory (OOM)
```bash
# Giảm batch size
python train.py --config configs/my_config.yaml --override training.batch_size=4

# Hoặc giảm history size
python train.py --config configs/my_config.yaml --override training.history_size=30
```

### Lỗi CUDA
```python
# Kiểm tra GPU
!nvidia-smi

# Force CPU (chậm hơn nhiều)
python train.py --config configs/my_config.yaml --override training.device=cpu
```

### Data không tìm thấy
```bash
# Kiểm tra đường dẫn
!ls -R Data/

# Update config nếu cần
python train.py --config configs/my_config.yaml \
  --override data.train_news=Data/raw/MINDlarge_train/news.tsv
```

## 8. Tips & Tricks

1. **Kaggle có giới hạn 30 giờ GPU/tuần** → Chạy với config tối ưu
2. **Save checkpoints thường xuyên** → Đã config sẵn mỗi epoch
3. **Enable Internet** trong Kaggle Settings để download pretrained models
4. **Commit notebook** thường xuyên để không mất progress
5. **Sử dụng Kaggle Datasets** để lưu checkpoints giữa các sessions

## 9. Ước Tính Thời Gian Training

| GPU | Batch Size | Time/Epoch | Total (3 epochs) |
|-----|-----------|------------|------------------|
| T4 (16GB) | 8 | ~4-5 giờ | ~12-15 giờ |
| P100 (16GB) | 8 | ~3-4 giờ | ~9-12 giờ |
| V100 (32GB) | 16 | ~2-3 giờ | ~6-9 giờ |

**Lưu ý**: Thời gian thực tế phụ thuộc vào:
- Kích thước dataset
- Có dùng LLM descriptions không
- Hard negative sampling có bật không
