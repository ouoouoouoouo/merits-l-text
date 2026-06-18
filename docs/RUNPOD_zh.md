# RunPod 上跑 IEMOCAP Stage I（新手版，不用 Docker）

RunPod 本身就是用 Docker 跑容器，但你**不用自己會 Docker** —— 他們已經把容器
包好給你，你只要 SSH 進去用就好。整個流程跟「租一台已經裝好 PyTorch 的 Linux」
完全一樣。

> 想學 Docker 怎麼運作？看 [DOCKER_zh.md](DOCKER_zh.md)。

---

## 0 ‧ 預期花費

| GPU | 訓練一次 50 epochs IEMOCAP | RunPod 計價 (約) |
|-----|----------------------------|-------------------|
| RTX A5000 (24GB) | ~30 分鐘 | $0.20/hr → 一次約 **$0.1** |
| RTX 3090 (24GB) | ~25 分鐘 | $0.30/hr → 一次約 **$0.13** |
| RTX 4090 (24GB) | ~15 分鐘 | $0.45/hr → 一次約 **$0.12** |
| A100 PCIe (40GB) | ~10 分鐘 | $1.20/hr → 一次約 **$0.2** |

第一次跑包含下載 roberta-large (~1.4GB)、安裝套件等，會多花 10~15 分鐘。
**選 RTX A5000 或 RTX 4090 性價比最好**，A100 對這個任務算 overkill。

---

## 1 ‧ 註冊 + 儲值

1. 到 [runpod.io](https://runpod.io) 註冊。
2. 在 Billing 頁面儲值 $10 美金（夠你跑幾十次實驗）。
3. （選用）在 Settings → SSH Public Keys 把你電腦的 SSH public key 貼上去，之後就能用 VSCode Remote-SSH 接進去。

---

## 2 ‧ 開一台 Pod

在 RunPod console：

1. **Pods → Deploy**。
2. **選 GPU**：點 `RTX A5000`（或 4090）。
3. **選 Template**：直接搜 `RunPod PyTorch 2.4`（或 2.3），這是官方 PyTorch image，
   已經有 Python 3.10 + CUDA 12 + PyTorch 預裝。
4. **Container Disk** 設 30GB（夠裝套件 + 下載 roberta-large）。
5. **Volume Disk** 設 20GB（資料集放這，重啟 pod 不會消失）。Volume mount path 維持預設 `/workspace`。
6. 點 **Deploy On-Demand**。

幾十秒後 pod 會變 **Running** 狀態。

---

## 3 ‧ 連進 Pod

最簡單：在 Pod 卡片上點 **Connect** → **Start Web Terminal**（瀏覽器內的 shell）。

進階一點：**Connect** → **Connect via SSH** 看到一條像這樣的指令：

```bash
ssh root@<某個IP> -p <某port> -i ~/.ssh/id_ed25519
```

如果你 step 1 設了 SSH key，這條指令在你本機跑就能進去；VSCode 也可以用
Remote-SSH 直連 (cmd palette → `Remote-SSH: Connect to Host`)。

---

## 4 ‧ 上傳 IEMOCAP 到 Pod

IEMOCAP 原始檔約 12GB（含音檔），但 **Stage I text-only 只需要 transcript +
emo evaluation 兩個資料夾**，不到 100MB。所以只傳必要的東西就好。

### 4.1 在你本機（Windows PowerShell）打包出文字部分

```powershell
cd "D:\CVdataset"
# 只壓 transcriptions/ 和 EmoEvaluation/，跳過 .wav .avi 等大檔
$src = "IEMOCAP_full_release"
$out = "IEMOCAP_text_only.zip"
Compress-Archive `
  -Path @(
    "$src\Session1\dialog\transcriptions",
    "$src\Session1\dialog\EmoEvaluation",
    "$src\Session2\dialog\transcriptions",
    "$src\Session2\dialog\EmoEvaluation",
    "$src\Session3\dialog\transcriptions",
    "$src\Session3\dialog\EmoEvaluation",
    "$src\Session4\dialog\transcriptions",
    "$src\Session4\dialog\EmoEvaluation",
    "$src\Session5\dialog\transcriptions",
    "$src\Session5\dialog\EmoEvaluation"
  ) `
  -DestinationPath $out
```

得到 `D:\CVdataset\IEMOCAP_text_only.zip`（約 5MB）。

### 4.2 上傳到 Pod

在 Pod 的 Web Terminal 或 SSH 裡：

```bash
mkdir -p /workspace/datasets
cd /workspace/datasets
```

用 RunPod 的「File Manager」標籤拖拉 `IEMOCAP_text_only.zip` 上去；或在本機跑：

```bash
# 本機 PowerShell / bash
scp -P <pod-port> -i ~/.ssh/id_ed25519 D:/CVdataset/IEMOCAP_text_only.zip root@<pod-ip>:/workspace/datasets/
```

回到 pod 上解壓 + **重建論文要求的目錄結構**：

```bash
cd /workspace/datasets
unzip IEMOCAP_text_only.zip -d IEMOCAP_text_only
# 重建期望路徑 (loader 預期 IEMOCAP_full_release/SessionN/dialog/{transcriptions,EmoEvaluation})
mkdir -p IEMOCAP_full_release
for i in 1 2 3 4 5; do
  mkdir -p IEMOCAP_full_release/Session$i/dialog
  mv IEMOCAP_text_only/IEMOCAP_full_release/Session$i/dialog/transcriptions \
     IEMOCAP_full_release/Session$i/dialog/
  mv IEMOCAP_text_only/IEMOCAP_full_release/Session$i/dialog/EmoEvaluation \
     IEMOCAP_full_release/Session$i/dialog/
done
rm -rf IEMOCAP_text_only IEMOCAP_text_only.zip
ls IEMOCAP_full_release/Session1/dialog/   # 確認看到 transcriptions/ 和 EmoEvaluation/
```

---

## 5 ‧ 把這個 repo 拉到 Pod

```bash
cd /workspace
git clone <你的 GitHub repo URL> code
cd code
pip install -r requirements.txt
```

`pip install` 大約 3~5 分鐘。

> ⚠️ requirements.txt 裡的 `torch` 已經是 RunPod template 預裝的版本，pip 會
> 自動跳過。如果它執意要重裝，加 `--no-deps torch` 或先 `pip install --upgrade pip`。

---

## 6 ‧ 跑訓練

```bash
export IEMOCAP_ROOT=/workspace/datasets/IEMOCAP_full_release
# 可選：開 WandB
# export WANDB_API_KEY=<你的 key>
# 並把 configs/iemocap_text.yaml 的 logging.use_wandb 改成 true

python -m src.train --config configs/iemocap_text.yaml
```

第一次跑：
- 會自動建 `data/manifests/iemocap/{train,val,test}.csv`
- 會下載 `roberta-large` 到 `~/.cache/huggingface/`（~1.4GB，幾分鐘）
- 訓練開始，每個 epoch 在 A5000 上約 30~40 秒，總共 20~30 分鐘左右

中途想監看：

```bash
# 開一個新 terminal
tail -f outputs/iemocap_text_stage1/train.log
# 或開 TensorBoard
tensorboard --logdir outputs/ --port 6006 --bind_all
# 然後在 Pod 卡片 Connect → HTTP Services 加 port 6006 暴露出來
```

訓練結束會印 test 結果，類似這樣：

```
TEST | acc=0.6512 weighted_f1=0.6432 macro_f1=0.6210
              precision    recall  f1-score   support
       angry     0.5871    0.6471    0.6157       170
       happy     0.6543    0.7195    0.6854       442
         sad     0.6776    0.6735    0.6755       245
     neutral     0.7034    0.5417    0.6118       384
    accuracy                         0.6512      1241
```

vanilla `roberta-large` 大概落在 **0.62~0.66**。要逼近論文 0.6984 需要先做
MSP-PODCAST 預訓練。

---

## 7 ‧ 把結果抓回本機

```bash
# 本機
scp -P <port> -i ~/.ssh/id_ed25519 -r \
    root@<ip>:/workspace/code/outputs/iemocap_text_stage1 \
    ./outputs/
```

只需要 `metrics.jsonl`、`test_report.txt`、`config.snapshot.yaml` 的話，把 `best/` 排除掉省流量：

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
    root@<ip>:/workspace/code/outputs/iemocap_text_stage1/{metrics.jsonl,test_report.txt,config.snapshot.yaml,train.log} \
    ./outputs/iemocap_text_stage1/
```

---

## 8 ‧ 跑完記得關 Pod

**RunPod 是按秒計費的**。Pod 跑完訓練不會自己停，你會一直被扣錢。

關閉方式：
- **Stop**：保留資料但不再收 GPU 錢（還是會收一點 storage 錢）。下次按 Resume 可恢復。
- **Terminate**：徹底刪掉，連 volume 一起。

跑完通常選 Stop，下次要再跑 Resume 起來，環境跟資料都還在。

---

## 9 ‧ 故障排除

**`CUDA out of memory`** — 把 batch size 調小：
```bash
python -m src.train --config configs/iemocap_text.yaml --override train.batch_size=16
```

**`OSError: We couldn't connect to 'https://huggingface.co'`** — pod 偶爾連線慢，重試。或先單獨拉模型：
```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('roberta-large')"
```

**找不到 `data/manifests/iemocap/train.csv`** — 多半是 `IEMOCAP_ROOT` 沒設或路徑不對。確認：
```bash
ls $IEMOCAP_ROOT/Session1/dialog/transcriptions/ | head
```
應該要看到 `Ses01F_impro01.txt` 之類的檔。

**訓練 loss 卡住不下降** — 學習率太大。試 `--override train.learning_rate=5e-5`。
