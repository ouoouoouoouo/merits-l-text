# Conda 管理環境（沒 sudo 也能跑）

適用情境：
- **學校 HPC / 共享 GPU 集群**沒給你 sudo
- **本機 Windows** 不是 admin，或不想全機安裝
- 想要嚴格控制 Python 版本

> **conda 不需要 sudo**。它把所有東西裝在你 home 目錄下（`~/miniconda3`），完全不碰系統。

---

## 1 ‧ 安裝 Miniconda（沒 sudo 也行）

### 1.1 Linux / WSL2

```bash
# 下載安裝腳本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 互動式安裝，全程按 yes / Enter，預設裝到 ~/miniconda3
bash Miniconda3-latest-Linux-x86_64.sh

# 讓 conda 在新 shell 自動可用 (安裝程式問你 "init?" 選 yes 就會幫你寫到 .bashrc)
source ~/.bashrc

# 驗證
conda --version
# conda 24.x.x
```

裝完用 `which conda` 應該指到 `~/miniconda3/bin/conda`。**完全在你的 home 目錄**，不需要任何系統權限。

### 1.2 Windows

到 [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) 下載
**Miniconda3 Windows 64-bit installer**。安裝時：

- ☑ Install for **Just Me** （不需要 admin）
- ☐ Add to PATH（不勾，避免汙染系統 PATH；改用 Anaconda Prompt）
- ☑ Register as default Python（看你習慣）

裝完從開始選單開 **Anaconda Prompt**，所有 `conda` 指令都在這裡跑。

---

## 2 ‧ 建一個專案環境

```bash
# 建一個叫 merits 的 env，指定 Python 3.10 (跟 Dockerfile 一致)
conda create -n merits python=3.10 -y

# 啟用
conda activate merits

# 之後關掉視窗或要切回去就 `conda deactivate`
```

> ⚠️ **不要用系統預設的 `base` env**。每個專案開一個獨立 env，互不污染。

---

## 3 ‧ 裝 PyTorch — 用 conda 還是 pip？

**用 pip 裝 PyTorch（推薦）**，理由：
1. PyTorch 官方現在主推 pip wheel，conda channel 更新較慢
2. 跟 `requirements-text.txt` 一致，少一層工具

### 3.1 有 GPU 的機器

先確認 CUDA driver 版本（這個不用 sudo，是顯卡驅動，cluster 管理員早裝好了）：

```bash
nvidia-smi
# 右上角 CUDA Version: 12.x  ← 這是 driver 支援的最高 CUDA，不是已安裝的 toolkit
```

只要顯示的版本 **≥ 12.1**，就用 CUDA 12.1 的 PyTorch wheel（**包含自己的 CUDA runtime**，不用裝 toolkit）：

```bash
# 最新版 (推薦) — transformers 4.40+ 要求 torch >= 2.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ 不要釘到 torch 2.3.x — 跟新版 `transformers` 不相容會在 model load 階段報
> `AutoModel requires the PyTorch library but it was not found`。

如果 `nvidia-smi` 顯示 11.8 之類舊版本：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

驗證 GPU 抓得到：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
# 2.3.1+cu121 True NVIDIA GeForce RTX 4090
```

### 3.2 沒 GPU（純測試 pipeline）

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 4 ‧ 裝這個 repo 的其他套件

**只跑 IEMOCAP text** → 用 minimal 版：

```bash
pip install -r requirements-text.txt
```

**之後要做 MSP-PODCAST 預訓練** → 用完整版：

```bash
pip install -r requirements.txt
```

完整版有 `librosa`/`soundfile`/`openai-whisper`，這些理論上**也是 pip 純 Python wheel**，
不需要 sudo。但 `librosa` runtime 會呼叫 `ffmpeg` 來讀音檔 — 沒 sudo 裝 ffmpeg 的話：

```bash
# 用 conda 裝 ffmpeg (純 user-space，不需要 sudo)
conda install -c conda-forge ffmpeg -y
which ffmpeg
# ~/miniconda3/envs/merits/bin/ffmpeg  ← 裝在 env 裡，完全用戶層
```

這是 **conda 比 venv 強的地方**：可以裝非 Python 的二進位工具，不需要 sudo。

---

## 5 ‧ 一行起跑訓練

```bash
conda activate merits
cd /path/to/your/repo
export IEMOCAP_ROOT=/path/to/IEMOCAP_full_release
python -m src.train --config configs/iemocap_text.yaml
```

---

## 6 ‧ 在共享 cluster 上的最佳實踐

### 6.1 把 HuggingFace cache 放在你自己的目錄

預設會寫在 `~/.cache/huggingface/`，home 目錄 quota 通常很小（10~50 GB）。
改放到 **scratch / data partition**：

```bash
# 寫進 .bashrc 或在 SLURM 腳本最前面
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache/transformers
```

第一次跑前 `mkdir -p /scratch/$USER/hf_cache`。

### 6.2 SLURM 提交範例

如果你用的 cluster 是 SLURM (`sbatch` / `srun`)：

```bash
#!/bin/bash
#SBATCH --job-name=iemocap_text
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out

# 進你的 conda env (cluster 上要先 source 一下)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate merits

export IEMOCAP_ROOT=/scratch/$USER/datasets/IEMOCAP_full_release
export HF_HOME=/scratch/$USER/hf_cache

cd /path/to/your/repo
python -m src.train --config configs/iemocap_text.yaml
```

存成 `run.slurm`，然後 `sbatch run.slurm`。

### 6.3 環境檔可以匯出來分享

```bash
# 匯出 conda env 的所有套件
conda env export --no-builds > environment.yml

# 別人 / 之後的你重建環境
conda env create -f environment.yml
```

---

## 7 ‧ Conda 常用指令速查

```bash
conda env list                    # 看有哪些 env
conda activate merits             # 進 env
conda deactivate                  # 出 env
conda list                        # 看當前 env 裝了什麼
conda remove -n merits --all      # 刪掉整個 env (檔案/快取全清)
conda clean -a                    # 清快取省硬碟
```

---

## 8 ‧ 常見坑

**`conda: command not found`** — 安裝完沒 source `.bashrc`，或開了新 terminal 沒重載。
跑 `source ~/miniconda3/etc/profile.d/conda.sh` 即可。

**`pip install torch` 跑出 CPU 版而非 GPU 版** — 因為你沒指定 `--index-url`。
務必加 `--index-url https://download.pytorch.org/whl/cu121`。

**HuggingFace 抱怨 quota 滿** — home 目錄爆了，改 `HF_HOME` 到 scratch 區。

**Cluster 上 `nvidia-smi` 看得到 GPU 但 `torch.cuda.is_available()` False** — 
通常是因為你忘了在 SLURM 申請 `--gres=gpu:1`，或者你裝錯 CUDA 版本的 PyTorch。

**conda env 改放別的硬碟** — 在 `~/.condarc` 加：
```yaml
envs_dirs:
  - /scratch/$USER/conda_envs
pkgs_dirs:
  - /scratch/$USER/conda_pkgs
```
之後 `conda create` 的新 env 都會放到 scratch。

---

## 9 ‧ Conda vs venv vs Docker — 一句話結論

| 工具 | 何時用 | 需要 sudo？ |
|------|--------|-------------|
| **conda** | 沒 sudo / HPC / 需要非 Python 二進位 (ffmpeg, cuda toolkit 等) | ❌ |
| **venv** | 簡單、純 Python 專案、不要碰系統 | ❌ |
| **Docker** | 跨機器要嚴格一致、要分享環境給很多人、CI | 要（但 RunPod 已幫你裝好） |

你的情況（沒 sudo + cluster GPU）→ **用 conda**。
