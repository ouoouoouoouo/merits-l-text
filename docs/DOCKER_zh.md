# Docker 入門 — 從零到跑這個 repo

> 你已經在用 RunPod 了，**RunPod 內部就是 Docker**，你已經是受益者。
> 這份文件教你的是「你自己怎麼用 Docker」，不是用 RunPod 必備的。

---

## 1 ‧ Docker 是什麼（一分鐘版）

把 Docker 想像成「**可攜式的隔離小電腦**」。

- **Image (映像檔)**：一張快照。包含作業系統 + 安裝好的軟體 + 你的程式。
  類比：iso 安裝光碟。
- **Container (容器)**：用 image 開出來的一個執行中的小電腦。
  類比：拿光碟裝出來的一台 VM，但啟動只要 1 秒。
- **Dockerfile**：「怎麼做這張 image」的食譜。

關鍵差異 vs 虛擬機：
- 不模擬整個 OS，只共用 Linux 核心 → **啟動快、佔資源少**。
- 跨機器跑出來的環境**完全一樣** → 「我電腦能跑，你電腦不能跑」這種事消失。

實務上你會用 Docker 來：
1. 把 Python + PyTorch + 一堆套件 **打包成一張 image**。
2. 把 image 推到 Docker Hub 或自己的 registry。
3. 在任何電腦上 `docker pull` 拉下來，**立刻有一模一樣的環境**。

---

## 2 ‧ 安裝 Docker Desktop (Windows)

1. 去 [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) 下載 Docker Desktop for Windows。
2. 安裝時記得勾 **Use WSL 2 instead of Hyper-V**（預設就是）。
3. 安裝完開啟 Docker Desktop，等左下角變綠燈（Docker is running）。
4. 開 PowerShell 測試：

```powershell
docker --version
# Docker version 26.x.x, build ...

docker run --rm hello-world
# 第一次會自動 pull hello-world image，印出歡迎訊息
```

> **要不要安裝 NVIDIA Container Toolkit？**
> 你**本機沒 GPU 訓練需求**，就不用。RunPod 那邊已經幫你裝好了。
> （要在本機跑 GPU 容器才需要，且只支援 Linux/WSL2。）

---

## 3 ‧ 五個必會的指令

```bash
# 1) 拉 image 下來
docker pull python:3.10-slim

# 2) 看本機已有的 image
docker images

# 3) 開一個 container 進去
docker run -it --rm python:3.10-slim bash
# 解釋：
#   -it      互動式 terminal
#   --rm     退出後自動刪 container (不留垃圾)
#   bash     在容器內執行的指令
# 進去後跟普通 Linux 一樣，exit 退出

# 4) 看正在跑 / 過去跑過的 container
docker ps         # 正在跑
docker ps -a      # 包含已停止的

# 5) 清掉沒在用的 image / container (省硬碟)
docker system prune -a
```

---

## 4 ‧ 兩個關鍵概念：Volume 跟 Port

### 4.1 Volume — 把本機資料夾掛進 container

Container 退出後資料就消失。要讓資料/程式碼能進出，用 `-v` 掛載：

```bash
docker run -it --rm \
    -v "/path/to/this/repo":/workspace/code \
    -v "/path/to/IEMOCAP_full_release":/workspace/datasets/IEMOCAP_full_release \
    python:3.10-slim bash
```

進到容器後 `cd /workspace/code` 就會看到本機這個專案，**雙向同步**。

語法：`-v <本機絕對路徑>:<容器內路徑>`。Windows 路徑用斜線 `/` 不要用反斜線。

### 4.2 Port — 把容器內服務開到本機

要在本機瀏覽器看 container 裡跑的 TensorBoard：

```bash
docker run -it --rm \
    -p 6006:6006 \
    -v "/path/to/this/repo":/workspace/code \
    your-image \
    bash
# 進去後 tensorboard --logdir outputs --bind_all
# 本機開 http://localhost:6006
```

`-p <本機port>:<容器port>`。

---

## 5 ‧ 看懂這個 repo 的 Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```
基底 image — 已經有 CUDA 12.1 + cuDNN + Ubuntu 22.04。
> 想要 CPU only 改成 `FROM python:3.10-slim` 就好。

```dockerfile
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface
```
設環境變數。`HF_HOME` 讓 HuggingFace 把下載的模型存在 `/workspace/.cache/`，這樣
mount volume 後**模型只下載一次**。

```dockerfile
RUN apt-get update && apt-get install -y python3.10 git ffmpeg ...
```
安裝系統工具。`RUN` = 在 build image 時執行一條指令。

```dockerfile
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
```
複製 requirements 進來 + 安裝。
> 為什麼不直接 `COPY . .`？因為 image build 有快取機制：requirements 沒變的話
> 重 build 會直接跳過 `pip install`。如果一次把所有檔案 COPY 進去，改一行 .py
> 也會觸發整個 pip install 重做。

```dockerfile
CMD ["bash"]
```
container 啟動時的預設指令。`docker run image` 時會跑這個。

---

## 6 ‧ 本機完整工作流（給有 GPU 的人）

> 沒 GPU 也能 build image、跑 CPU 推論測試，只是訓練會超慢。

### 6.1 Build image

```bash
cd "/path/to/this/repo"
docker build -t merits-l-text .
# -t  幫 image 取名字 (tag)
# .   build context = 當前資料夾
```

第一次跑會花 5~10 分鐘（下載 base image、apt install、pip install）。
之後改程式碼重 build 只要幾秒 (快取生效)。

### 6.2 Run container

```bash
# Windows PowerShell 多行用反引號 `
docker run --gpus all -it --rm `
    -v "/path/to/this/repo:/workspace/code" `
    -v "/path/to/IEMOCAP_full_release:/workspace/datasets/IEMOCAP_full_release" `
    -e IEMOCAP_ROOT=/workspace/datasets/IEMOCAP_full_release `
    -p 6006:6006 `
    merits-l-text bash
```

進去後：

```bash
cd /workspace/code
python -m src.train --config configs/iemocap_text.yaml
```

訓練結果會寫進本機的 `/path/to/this/repo/outputs/`（因為 mount）。

### 6.3 退出 container

`exit` 或 Ctrl-D。因為加了 `--rm`，container 會自動刪掉，但你的 image、本機檔案
都還在。下次再 `docker run ...` 就能再開新的 container。

---

## 7 ‧ 怎麼把 image 推到 RunPod 用

如果你**真的想用自己的 Dockerfile** 在 RunPod 上跑（而不是用他們的 PyTorch
template）：

### 7.1 註冊 Docker Hub

到 [hub.docker.com](https://hub.docker.com) 註冊帳號，假設帳號是 `your-name`。

### 7.2 改 tag + push

```bash
# 重新 tag 成 hub 的格式
docker tag merits-l-text your-name/merits-l-text:latest

# 登入
docker login

# 推上去 (約 1~3GB，要花幾分鐘)
docker push your-name/merits-l-text:latest
```

### 7.3 RunPod 用你的 image

Deploy pod 時 → **Custom Template** → **Container Image** 填
`your-name/merits-l-text:latest` → Deploy。

進到 pod 後環境就是你 Dockerfile 定義的那一套。

> **大多數時候沒必要這樣做** — RunPod 預設 PyTorch template 已經夠用，
> 直接 `pip install -r requirements.txt` 就好，省下 push image 的時間。

---

## 8 ‧ 常見坑

**Build 卡在 `apt-get update`** — 多半是網路不穩，重試 `docker build -t ... .`。

**`no space left on device`** — Docker 預設把 image 存在 C 槽。
- 短期：`docker system prune -a` 清舊 image。
- 長期：Docker Desktop → Settings → Resources → Advanced，把 Disk image
  location 改到 D 槽。

**Mount 進去看不到本機檔案** — 路徑寫錯。Windows 一定要用 `/` 不能用 `\`。
進 container 後 `ls /workspace/code` 確認有東西。

**權限錯誤 (Linux container 寫不出檔)** — Windows + WSL2 mount 通常沒事，
Linux host 上要加 `--user $(id -u):$(id -g)`。

**`Error response from daemon: pull access denied`** — image 名字打錯，或
是 private image 沒登入。檢查 `docker images` 看本機有沒有。

---

## 9 ‧ 一句話總結

| 你想做 | 推薦做法 |
|--------|----------|
| 用 RunPod 跑訓練（新手） | 不用碰 Docker，照 [RUNPOD_zh.md](RUNPOD_zh.md) |
| 本機有 GPU 想快速測試 | 用 Dockerfile build 跑 |
| 想分享環境給隊友 | push image 到 Docker Hub |
| 跨機器要保證環境一致 | 用 Docker |
