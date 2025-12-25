# Workflow: Từ Training Đến Submission

## 📋 Tổng Quan Workflow

Đây là quy trình hoàn chỉnh từ training model đến tạo file submission cho MIND leaderboard:

```
1. Training Model → 2. Load Best Checkpoint → 3. Inference on Test Set → 4. Generate Submission
```

## 🚀 Bước 1: Training Model

### Chạy Training với Config 48GB
```bash
# Training với config tối ưu cho GPU 48GB
python train.py --config configs/gpu_48gb_large.yaml

# Hoặc training nhanh (aggressive)
python train.py --config configs/gpu_48gb_large_fast.yaml
```

### Kết Quả Training
Sau khi training xong, bạn sẽ có:
```
output/
├── checkpoints/
│   ├── best_model.pt              # Model tốt nhất (theo nDCG@10)
│   ├── checkpoint_epoch_1.pt      # Checkpoint epoch 1
│   ├── checkpoint_epoch_2.pt      # Checkpoint epoch 2
│   ├── checkpoint_epoch_3.pt      # Checkpoint epoch 3
│   └── ...
├── models/
│   └── deberta_naml_final.pt      # Model cuối cùng
└── tensorboard/                   # TensorBoard logs
```

### Chọn Model Tốt Nhất
```bash
# Xem thông tin các checkpoint
ls -lh output/checkpoints/

# Kiểm tra metrics của best model
python -c "
import torch
ckpt = torch.load('output/checkpoints/best_model.pt', map_location='cpu')
print('Epoch:', ckpt.get('epoch', 'Unknown'))
print('Metrics:', ckpt.get('metrics', {}))
"
```

## 🎯 Bước 2: Chuẩn Bị Test Data

### Download MINDlarge Test Set
```bash
# Nếu chưa có test data
python -m src.scripts.download_mind --size large --split test

# Hoặc download manual và extract vào:
# Data/raw/MINDlarge_test/news.tsv
# Data/raw/MINDlarge_test/behaviors.tsv
```

### Kiểm Tra Test Data
```bash
# Kiểm tra test data có đủ không
ls -lh Data/raw/MINDlarge_test/

# Xem số lượng test impressions
wc -l Data/raw/MINDlarge_test/behaviors.tsv
```

## 🔮 Bước 3: Generate Submission

### Cách 1: Sử Dụng Script Trực Tiếp
```bash
# Generate submission với best model
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best_model.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --config configs/gpu_48gb_large.yaml \
    --output-dir output/submission \
    --batch-size 8
```

### Cách 2: Với LLM Descriptions (Nếu Có)
```bash
# Nếu có LLM descriptions
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best_model.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --config configs/gpu_48gb_large.yaml \
    --llm-description-path Data/generated/news_descriptions.json \
    --output-dir output/submission \
    --batch-size 8
```

### Cách 3: Sử Dụng Example Script
```bash
# Chạy example script
python examples/generate_submission_example.py

# Chọn option 2 (MINDlarge)
```

## 📦 Kết Quả Submission

Sau khi chạy xong, bạn sẽ có:
```
output/submission/
├── prediction.txt              # File prediction chính (submit file này)
├── prediction_metadata.json    # Metadata về model và config
└── submission.zip             # Package đầy đủ (hoặc submit file này)
```

### Format File Prediction
```
# prediction.txt format:
impression_id_1 [ranked_news_ids_separated_by_space]
impression_id_2 [ranked_news_ids_separated_by_space]
...

# Ví dụ:
123 N54321 N12345 N67890 N11111 N22222
124 N99999 N88888 N77777 N66666 N55555
```

## 🎯 Bước 4: Submit Lên Leaderboard

### Option 1: Submit prediction.txt
1. Vào MIND leaderboard website
2. Upload file `output/submission/prediction.txt`
3. Chờ kết quả evaluation

### Option 2: Submit submission.zip
1. Vào MIND leaderboard website  
2. Upload file `output/submission/submission.zip`
3. Chờ kết quả evaluation

## 🔧 Troubleshooting

### Lỗi Out of Memory Khi Inference
```bash
# Giảm batch size
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best_model.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --batch-size 4  # Giảm từ 8 xuống 4

# Hoặc batch size = 1 (chậm nhất nhưng ít memory nhất)
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best_model.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --batch-size 1
```

### Lỗi Checkpoint Không Tìm Thấy
```bash
# Kiểm tra các checkpoint có sẵn
ls -lh output/checkpoints/

# Sử dụng checkpoint khác
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/checkpoint_epoch_3.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv
```

### Lỗi Test Data Không Tìm Thấy
```bash
# Kiểm tra đường dẫn
ls -lh Data/raw/MINDlarge_test/

# Nếu không có, download lại
python -m src.scripts.download_mind --size large --split test

# Hoặc update đường dẫn
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best_model.pt \
    --test-news path/to/your/news.tsv \
    --test-behaviors path/to/your/behaviors.tsv
```

## 📊 So Sánh Multiple Models

### Generate Submission Cho Nhiều Checkpoint
```bash
# Model từ epoch 1
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/checkpoint_epoch_1.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --output-dir output/submission_epoch1

# Model từ epoch 2  
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/checkpoint_epoch_2.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --output-dir output/submission_epoch2

# Best model
python -m src.scripts.generate_submission \
    --checkpoint output/checkpoints/best_model.pt \
    --test-news Data/raw/MINDlarge_test/news.tsv \
    --test-behaviors Data/raw/MINDlarge_test/behaviors.tsv \
    --output-dir output/submission_best
```

### So Sánh Kết Quả
```bash
# Xem metadata của từng submission
cat output/submission_epoch1/prediction_metadata.json
cat output/submission_epoch2/prediction_metadata.json  
cat output/submission_best/prediction_metadata.json
```

## ⏱️ Ước Tính Thời Gian

### Inference Time (MINDlarge test set)
| GPU | Batch Size | Inference Time |
|-----|------------|----------------|
| RTX A6000 (48GB) | 8 | ~30-45 phút |
| RTX A6000 (48GB) | 4 | ~60-90 phút |
| RTX A6000 (48GB) | 1 | ~3-4 giờ |
| A100 (80GB) | 16 | ~15-30 phút |

### File Sizes
- `prediction.txt`: ~50-100MB (tùy số impressions)
- `prediction_metadata.json`: ~5KB
- `submission.zip`: ~20-50MB (compressed)

## 🎯 Tips & Best Practices

### 1. Chọn Model Tốt Nhất
```bash
# Không nhất thiết phải dùng best_model.pt
# Có thể epoch cuối cùng tốt hơn nếu không có overfitting
# Kiểm tra validation metrics trong TensorBoard
tensorboard --logdir output/tensorboard
```

### 2. Batch Size Optimization
```bash
# Bắt đầu với batch size lớn, giảm dần nếu OOM
# GPU 48GB: thử batch_size=16 → 8 → 4 → 1
# GPU 24GB: thử batch_size=8 → 4 → 2 → 1
```

### 3. Backup Submissions
```bash
# Tạo backup với timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r output/submission output/submission_backup_$timestamp
```

### 4. Validate Submission Format
```bash
# Kiểm tra format trước khi submit
python -c "
import json
with open('output/submission/prediction_metadata.json') as f:
    meta = json.load(f)
print('Num impressions:', meta['num_impressions'])
print('Created at:', meta['created_at'])

# Kiểm tra prediction.txt
with open('output/submission/prediction.txt') as f:
    lines = f.readlines()
print('Prediction lines:', len(lines))
print('First line:', lines[0].strip())
"
```

## 🏆 Expected Results

### Baseline Performance (MINDlarge)
Với DeBERTa-xlarge và config tối ưu:
- **AUC**: 0.68-0.70
- **MRR**: 0.32-0.34  
- **nDCG@5**: 0.32-0.34
- **nDCG@10**: 0.35-0.37

### Với LLM Descriptions
Có thể cải thiện thêm 1-2%:
- **AUC**: 0.69-0.71
- **MRR**: 0.33-0.35
- **nDCG@5**: 0.33-0.35
- **nDCG@10**: 0.36-0.38

## 📝 Checklist Trước Khi Submit

- [ ] ✅ Model đã training xong và có best_model.pt
- [ ] ✅ Test data (news.tsv, behaviors.tsv) đã có sẵn
- [ ] ✅ Generate submission thành công
- [ ] ✅ File prediction.txt có format đúng
- [ ] ✅ Số lượng predictions khớp với số impressions
- [ ] ✅ Backup submission files
- [ ] ✅ Ready to submit!

**Workflow hoàn tất! Bạn đã sẵn sàng submit lên MIND leaderboard!** 🎉