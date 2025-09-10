# ev-NSFnet: 企業級 PINNs for Lid-Driven Cavity (Re=5000) 🚀
> 本專案以 opencode + GitHub Copilot 輔助開發  
結合主 PINN（Navier-Stokes + 連續方程）與 Entropy Viscosity 副網路（EVM）以提升高雷諾數穩定性，專注「物理一致性 + 可再現 + 高效訓練」。

## 1. 概覽 (Overview)
- 問題場景：二維不可壓縮蓋板驅動腔流 (Re=5000)
- 架構組成：Main PINN（u,v,p）+ EVM Net（entropy residual → ν_art）
- 關鍵手段：Entropy residual → 人工粘滯度、PDE 距離權重、LAAF/TSA 激活、分段式訓練 stage
- 目標：在高雷諾數下減少震盪、提升收斂穩定與物理解釋性

## 2. 核心特性 (Key Features)
- 模組化：求解器 / 網路 / 最優化 / 配置解耦
- 分佈式：torchrun + DDP，P100 友善（自動回退 eager）
- 多階段訓練：alpha_EVM / lr / scheduler 三維聯動
- 自適應優化：AdamW 階段化 + L-BFGS 精修（停滯觸發）
- 可調網路：固定層數或逐層 hidden size 自定義
- 權重初始化：Xavier / Kaiming / original + gain + bias 控制
- 物理一致：導數縮放、PDE 距離權重、Entropy Viscosity
- 監控簡潔：關鍵曲線集中，避免 I/O 過載

## 3. 快速開始 (Quick Start)
安裝依賴：
```bash
pip install -r requirements.txt
```
訓練：
```bash
python train.py --config configs/production.yaml
```
乾跑檢查：
```bash
python train.py --config configs/production.yaml --dry-run
```
測試 / 推論：
```bash
python test.py
python predict.py --checkpoint path/to/checkpoint.pth --output_dir results/predict_run
```

舊版檔案轉換（.pth + .pth_evm → 統一 checkpoint）：
```bash
python pinn_modules/convert_legacy_checkpoints.py \
  --input_dir results/Re5000/4x120_Nf200k_lamB10_alpha0.05 \
  --output_dir results/converted/Re5000/4x120_Nf200k_lamB10_alpha0.05 \
  --re 5000 --alpha_evm 0.05

# 轉換後可直接用 test.py 掃描：
python test.py --run_dir results/converted/Re5000/4x120_Nf200k_lamB10_alpha0.05 \
               --output_dir results/test_results/my_run
```
精簡目錄：
```
pinn_modules/   # 管理型模組 (checkpoint / optimizer)
configs/        # YAML 配置
pinn_solver.py  # 核心求解器
net.py          # FCNet / 組裝
train.py        # 訓練入口
predict.py      # 單檔推論
tsa_activation.py / laaf.py
```

## 4. 配置指南 (Configuration)
`configs/production.yaml` 核心區塊：
- network: 結構/激活/初始化
- training: stages, weight decay, batch, N_f
- system: tensorboard, intervals, ddp 設定

Training stages 格式：
```
training_stages:
  - [alpha_evm, epochs, base_lr, scheduler]
```
支援 scheduler：`Constant | MultiStepLR | CosineAnnealingLR | SGDR (擴充封裝)`

權重初始化欄位：
```
weight_init_main / weight_init_evm: xavier_uniform|xavier_normal|kaiming_uniform|kaiming_normal|original
weight_init_gain_main / weight_init_gain_evm: 浮點縮放 (與 base gain 相乘)
bias_init_main / bias_init_evm: zeros|ones|keep
```
注意：`original` 不重置權重，只做 gain 線性縮放；初始化後才執行首/末層縮放。

## 5. 模型與方法 (Methodology)
- 控制方程：2D incompressible Navier-Stokes，Re=5000
- 邊界條件：上壁 `u = 1 - x^2`，其餘 no-slip，壓力自然條件
- Entropy Viscosity：EVM 網路估計 entropy residual → 人工粘滯 ν_art = β * R_entropy / Re（β 可配）
- PDE 距離權重：w(d) 預計算並 detach，強化邊界層學習
- 導數縮放：一/二階梯度倍率避免標準化座標下數值失衡
- 激活：`tanh | laaf | tsa`（LAAF: 層級縮放參數；TSA: PINN 收斂友善）

## 6. 優化與訓練機制 (Optimization)
- AdamW 分階段 weight decay：早期平滑 → 後期細節；最後可 0
- L-BFGS 精修：滑窗停滯觸發（指標：物理 loss 改善率下降）
- Scheduler：多階段可混用（例如前期 Cosine / 後期 Constant）
- 監控指標（推薦）：
  - Physics/Total_Loss
  - PDE/u_residual, PDE/v_residual
  - BC/total
  - EVM/alpha_current, EVM/cap_ratio
  - Training/LearningRate, Training/WeightDecay
  - LAAF/layer_k_scale (如啟用)
- 推薦檢查：新 stage、L-BFGS 結束、恢復 checkpoint 後

## 7. 分佈式與效能 (Distributed & Performance)
- P100 相容：禁用 `torch.compile`，強制 eager（環境變數已自動設定）
- I/O 降載：`tensorboard_interval`, `timing_sync_interval` 控制寫入與 GPU 同步頻率
- DDP：`broadcast_buffers=false`（無 BN 時節省同步）
- 預計算：PDE 距離權重一次生成，多 epoch 重用
- 記憶體建議：顯存 < 14GB（單卡），超出時減少 N_f 或 batch_size

## 8. 常見問題 (FAQ)
1. 初期 loss 震盪正常嗎？  
   高 Re + 多殘差項初期梯度未對齊；可降低初始 lr 或減小第一階段 α_EVM。
2. 什麼時候調整 weight_init_gain？  
   深層 tanh 梯度過早飽和 → 降 gain；收斂緩慢且梯度過小 → 微增 (≤1.2)。
3. 何時啟用 PDE 距離權重？  
   邊界層誤差持續高於內部區域；開啟後留意總 loss 平衡。
4. L-BFGS 沒觸發？  
   停滯檢測門檻未達，可延長 stage 或調低觸發閾值。
5. AdamW 為何會重建？  
   進入新 stage 或精修切換會重建動量，避免舊動量污染新 lr/α。

## 9. 開發與貢獻 (Development)
完整目錄：
```
pinn_modules/
  checkpoint_manager.py
  optimizer_manager.py
configs/
pinn_solver.py
net.py
train.py / test.py / predict.py
laaf.py / tsa_activation.py
tools.py / cavity_data.py
```
程式規範：
- Python 3.10+，PEP8 + type hints
- 模組低耦合，功能前置簡短註解
- 避免三層以上巢狀迴圈
Checkpoint：
```bash
python train.py --config configs/production.yaml --resume path/to/ckpt.pth
```

## 10. 聲明 (Acknowledgement)
- 本專案使用 opencode + GitHub Copilot 輔助開發
- License: MIT
- 若此專案對研究有助，歡迎 ⭐ 支持

---
需要更詳細數學推導或額外指標，可再行擴充專章。
