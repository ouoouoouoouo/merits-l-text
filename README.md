# MERITS-L reproduction — Text branch (Stage I, IEMOCAP)

從零復現論文 *"LLM supervised Pre-training for Multimodal Emotion Recognition
in Conversations"* (Dutta & Ganapathy, ICASSP 2025) 的文字分支。

**目前先聚焦在 IEMOCAP**。MELD / CMU-MOSI / 多模態 Stage II/III 之後再加。

| Stage | Dataset | Paper 目標 (weighted F1) |
|-------|---------|-------------------------|
| I — Text | IEMOCAP | **69.84** |

> 論文這個 69.84 是用「先在 MSP-PODCAST + GPT-3.5 偽標籤上 pre-train 過的
> RoBERTa-FT」當作 init。直接從 HuggingFace `roberta-large` 起跳的話，預期約
> **64~65**（差約 5 分絕對值，對應論文 Fig.3 的 8.22% 相對降幅）。
>
> 建議路線：先跑通 Stage I (用 vanilla roberta-large) → 確認 pipeline 沒問題
> → 再回頭做 MSP-PODCAST 預訓練拉分數。

---

## 0 ‧ 環境怎麼選

| 你的情況 | 推薦做法 | 教學 |
|----------|---------|------|
| **沒 sudo / 共享 cluster** | **conda** 管理環境（不需要任何系統權限） | [docs/CONDA_zh.md](docs/CONDA_zh.md) |
| 用 RunPod 雲端 GPU | 直接用他們 PyTorch template，不用碰 Docker | [docs/RUNPOD_zh.md](docs/RUNPOD_zh.md) |
| 想學容器化 | 用本 repo 的 Dockerfile | [docs/DOCKER_zh.md](docs/DOCKER_zh.md) |
| 本機快速測試（有 admin） | venv + pip | 下面 §2.1 |

> 只想跑 IEMOCAP 純文字部分？用 `requirements-text.txt`（不含 ffmpeg/audio 依賴，沒 sudo 也能裝）。

---

## 1 ‧ 倉庫結構

```
configs/iemocap_text.yaml   # ← 主要用這個
src/
  data/iemocap.py           # 從 IEMOCAP_full_release 抓 5531 句 (已驗證)
  data/dataset.py           # PyTorch Dataset + RoBERTa tokenizer
  models/text_classifier.py # RoBERTa-large + FC head
  train.py                  # 訓練主程式 (AdamW + warmup + fp16 + early-stop)
  utils/                    # config / metrics / logger / seed
scripts/preprocess_iemocap.py
requirements.txt
Dockerfile                  # 想用 Docker 才用得到
```

其他資料集 (MELD / MOSI / MSP-PODCAST) 的腳本與 config 也在裡面，先不用管。

---

## 2 ‧ Quick start — IEMOCAP

### 2.1 本機（venv 版，有 admin）

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1               # PowerShell
# 或 source .venv/bin/activate           # bash/WSL/Linux

pip install -r requirements-text.txt     # IEMOCAP 只需要這個
# Windows PowerShell:
$env:IEMOCAP_ROOT = "C:/path/to/IEMOCAP_full_release"

python -m src.train --config configs/iemocap_text.yaml
```

### 2.1b 沒 sudo 的機器（conda 版）

```bash
# 詳見 docs/CONDA_zh.md
conda create -n merits python=3.10 -y && conda activate merits
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-text.txt

export IEMOCAP_ROOT=/path/to/IEMOCAP_full_release
python -m src.train --config configs/iemocap_text.yaml
```

第一次執行會自動：
1. 掃 IEMOCAP → 篩 5531 句 → 寫 `data/manifests/iemocap/{train,val,test}.csv`
2. 下載 `roberta-large` (~1.4GB，HuggingFace cache)
3. 訓練 50 epochs (early stopping)
4. 把最佳模型存到 `outputs/iemocap_text_stage1/best/`
5. 在 test set 印出 weighted F1 + 混淆矩陣

### 2.2 雲端 GPU

照 [docs/RUNPOD_zh.md](docs/RUNPOD_zh.md) 做。RTX A5000 / A100 上 50 epochs 大約 20~40 分鐘。

---

## 3 ‧ 你應該關心的設定

`configs/iemocap_text.yaml` 已經對齊論文 Sec. IV-B：

```yaml
model.pretrained: roberta-large       # ← 想用 RoBERTa-FT 時改成 outputs/pretrain_msp/best
train.epochs: 50
train.batch_size: 32
train.learning_rate: 1.0e-4
train.fp16: true                      # A5000/A100/3090 開著沒問題
```

要改任何參數，用 `--override`：

```bash
python -m src.train --config configs/iemocap_text.yaml \
    --override train.batch_size=16 train.epochs=20
```

---

## 4 ‧ 輸出在哪裡

```
outputs/iemocap_text_stage1/
  best/                    # HuggingFace 格式的 RoBERTa encoder + classifier.pt
  config.snapshot.yaml     # 這次跑的完整 config
  metrics.jsonl            # 每個 step 的 loss/lr/F1 (一行一個 JSON)
  tb/                      # TensorBoard 事件檔
  train.log                # 文字 log
  test_report.txt          # sklearn classification report + confusion matrix
```

開 TensorBoard 看曲線：

```bash
tensorboard --logdir outputs/
```

---

## 5 ‧ 怎麼判斷有沒有跑對

驗證點 1：data manifest 句數
- 跑完 `python -m scripts.preprocess_iemocap --iemocap-root /path/to/IEMOCAP_full_release`
- `data/manifests/iemocap/train.csv` + `val.csv` + `test.csv` 行數加起來應該是 **5531** (扣掉 header)
- train=3205 / val=1085 / test=1241 ← 已經在我本機驗過

驗證點 2：訓練曲線
- 第 1~2 epoch 的 val weighted_f1 應該已經 > 0.5
- 收斂大概落在 0.62~0.66 區間（用 vanilla `roberta-large`）
- 用 RoBERTa-FT 時應該逼近 0.70

---

## 6 ‧ 之後要加的東西（待辦）

- [ ] MSP-PODCAST 預訓練全流程 (Whisper + GPT-3.5) — 腳本已寫，需要實際資料
- [ ] MELD Stage I
- [ ] CMU-MOSI Stage I
- [ ] Stage II：Bi-GRU + self-attention 對整段對話建模
- [ ] Stage III：CARE 語音 embedding + co-attention 多模態融合
