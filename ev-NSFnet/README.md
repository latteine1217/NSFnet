# ev-NSFnet (PINNs for Lid-Driven Cavity, Re=5000)

> 本專案以 opencode + GitHub Copilot 輔助開發 🚀

本庫為使用 Physics-Informed Neural Networks (PINNs) 進行蓋板驅動腔流（Re=5000）的研究程式。包含主網路（解 Navier–Stokes + 連續方程）與 EVM 副網路（entropy residual → artificial viscosity）。支援多階段訓練、分佈式、與 L-BFGS 段內精修。

---

## 快速開始
- 訓練（生產設定）
  - `python train.py --config configs/production.yaml`
- 僅檢視設定（不跑訓練）
  - `python train.py --config configs/production.yaml --dry-run`

---

## SGDR 學習率排程（Cosine + Warm-up）
本專案支援「SGDR = 緩啟動 LinearLR + CosineAnnealingWarmRestarts」的學習率策略，且可在多階段訓練中逐段設定。

### 1) 啟用方式
在 `training.training_stages` 的第四欄填入 `SGDR`（或 `CosineAnnealingWarmRestarts`）：

```
training:
  training_stages:
    - [0.05, 150000, 1e-3, Constant]   # Stage 1：穩定收斂
    - [0.03, 150000, 1e-3, SGDR]       # Stage 2：SGDR 開始
    - [0.01, 200000, 2e-4, SGDR]       # Stage 3
    - [0.005, 200000, 4e-5, SGDR]      # Stage 4
    - [0.002, 200000, 1e-5, SGDR]      # Stage 5
    - [0.001, 200000, 2e-6, Constant]  # Stage 6：收斂
```

> 小提示：每個 stage 的基礎學習率取自第三欄（如 `1e-3`, `2e-4`），SGDR 會在該階段內進行暖啟動與餘弦週期更新。

### 2) 參數配置（全域 SGDR）
在 `training.sgdr` 區塊設定（若缺省，會使用穩健預設）：

```
training:
  sgdr:
    warmup_epochs: 5000   # 緩啟動步數（每 stage 計）
    T_0: 30000            # 第一個餘弦週期長度（不含 warm-up）
    T_mult: 2             # 週期倍增倍率（第二個週期長度 = T_0 * T_mult）
    start_factor: 0.1     # 緩啟動起始比例（相對當前 stage 基礎 lr）
    end_factor: 1.0       # 緩啟動結束比例
    # eta_min 可省略：預設用下一個 stage 的 lr；若無下一 stage 則取 0.1×當前 lr，且不低於 1e-8
```

- `warmup_epochs`：線性由 `start_factor × lr` 漸增至 `end_factor × lr`。
- `T_0`、`T_mult`：對應 PyTorch `CosineAnnealingWarmRestarts`；每個 stage 內週期會從 `T_0` 開始，之後按 `T_mult` 倍增。
- `eta_min`：每個 stage 的餘弦波谷學習率；不設定時會自動接軌下一階段 lr，避免跨階段跳變。

### 3) 預設與建議（P100 友善）
- 預設暖啟動為該階段 5%（夾在 500～10000），`T_0` 取剩餘步數約 25%（下限 1000）、`T_mult=2`。
- 長訓（例如單階段 ≥ 150k steps）建議：`warmup_epochs=3k~10k`、`T_0=20k~40k`、`T_mult=2`。
- 若 EVM 邊訓邊凍結/解凍，或中途插入 L-BFGS 精修，SGDR 狀態會自動重建並續用，不會造成 lr 瞬斷。

### 4) 運作細節
- 本專案內部建立 `SequentialLR(LinearLR → CosineAnnealingWarmRestarts)`；每步在 `optimizer.step()` 後 `scheduler.step()`。
- 分佈式（DDP）環境與 L-BFGS 段落結束後，會保存與重建 SGDR 排程（包含暖啟動與週期參數），保持學習率連續。
- TensorBoard 會記錄 `Training/LearningRate`，Console 會輸出：
  - `🔧 SequentialLR (包含warmup/SGDR)` 或 `🔧 CosineAnnealingWarmRestarts` 設定資訊

### 5) 最小可運行範例
將 `configs/production.yaml` 中 `training_stages` 與 `training.sgdr` 設定如上，然後：

```
python train.py --config configs/production.yaml
```

如需快速檢查設定不跑訓練：
```
python train.py --config configs/production.yaml --dry-run
```

---

## 訓練命令與腳本
- CLI：
  - 訓練：`python train.py --config configs/production.yaml`
  - 測試：`python test.py`
  - P100 檢查：`python test_p100_compatibility.py`
- SLURM（伺服器 `train.sh` 以 `torchrun` 分佈式執行；分配 2× P100, 100GB RAM）

---

## 相容性與環境
- 硬體：Dell R740，Intel Xeon Gold 5118 ×2（48 threads），Nvidia Tesla P100 16GB ×2。
- CUDA Capability: 6.0（不支援 Triton）；已自動回退 eager 模式（`TORCH_COMPILE_BACKEND=eager`, `TORCHDYNAMO_DISABLE=1`）。
- PyTorch: 2.6.0+cu126；SGDR 不依賴 `torch.compile`，P100 相容 ✅。

---

## 參考指令
- 訓練：`python train.py --config configs/production.yaml`
- 只看設定：`python train.py --config configs/production.yaml --dry-run`

---

## 貢獻與授權
- 歡迎提交 Issue / PR，一起改進 🔧
- 程式授權條款詳見原始檔頭說明
