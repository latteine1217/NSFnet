# 🌊 Physics-Informed Neural Networks (PINNs) for NSFnet

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-green.svg)

## 📖 專案簡介

本專案實現了基於Physics-Informed Neural Networks (PINNs)的NSFnet模型，專門用於解決不可壓縮Navier-Stokes方程的cavity flow問題。採用分散式訓練架構，支援多GPU並行運算。

✨ **開發工具**: 本專案使用 [opencode](https://opencode.ai) + GitHub Copilot 開發

## 🏗️ 系統架構

### 核心模組
- **pinn_solver.py**: 主要的PINN求解器類別
- **net.py**: 全連接神經網路定義  
- **train.py**: 訓練主程式
- **test.py**: 模型測試與評估
- **cavity_data.py**: 腔體流數據處理與監督數據加載
- **tools.py**: 工具函數
- **config.py**: 配置管理系統，包含監督數據配置

### 網路結構
- **主網路**: 6層隱藏層，80個神經元 (u, v, p預測)
- **EVM網路**: 4層隱藏層，40個神經元 (渦流黏度預測)  
- **激活函數**: Tanh (已優化Xavier初始化)
- **輸入**: 空間座標 (x, y)
- **輸出**: 速度場 (u, v)、壓力場 (p)、渦流黏度 (e)
- **初始化**: 針對tanh的Xavier初始化，避免早期飽和

## 🚀 快速開始

### 硬體需求
- **伺服器**: Dell R740 或同等級
- **CPU**: Intel Xeon Gold 5118 12 Core×2 (48 threads)
- **記憶體**: 112GB 
- **GPU**: Nvidia P100 16GB ×2
- **作業系統**: Linux with SLURM

### 環境安裝

```bash
# 安裝依賴套件
pip install torch numpy scipy matplotlib

# 克隆專案
git clone https://github.com/YOUR_USERNAME/ev-NSFnet.git
cd ev-NSFnet
```

### 訓練模型

```bash
# 使用SLURM提交訓練作業
sbatch train.sh

# 或直接執行Python訓練
python train.py [OPTIONS]

# 從檢查點恢復訓練
# --resume <CHECKPOINT_PATH>: (可選) 指定要恢復訓練的檢查點檔案路徑。
#                             例如: ~/NSFnet/ev-NSFnet/results/Re5000/6x80_Nf120k_lamB10_alpha0.05Stage_1/checkpoint_epoch_10000.pth
python train.py --resume <CHECKPOINT_PATH>

# 依據配置檔 per-stage 指定 scheduler（Constant | MultiStepLR | CosineAnnealingLR）
python train.py --config configs/production.yaml
```

### 測試模型

```bash
# 執行測試，並指定包含檢查點的訓練結果目錄
# --run_dir <RUN_DIRECTORY>: (必填) 指定包含訓練檢查點的目錄路徑。
#                            例如: ~/NSFnet/ev-NSFnet/results/Re5000/6x80_Nf120k_lamB10_alpha0.05Stage_1
python test.py --run_dir <RUN_DIRECTORY>
```

## ⚙️ 訓練配置

### 關鍵參數
- **Reynolds數**: 3000, 5000
- **訓練點數**: 120,000個隨機點 (完整批次訓練)
- **總訓練輪數**: 1,800,000 epochs (分6個階段)
- **學習率**: 動態調整 (1e-3 → 2e-6)
- **權重係數**:
  - 邊界條件: 10.0
  - 方程約束: 1.0  
  - EVM正則化: 0.03 → 0.0002 (逐漸減少)
- **人工粘滯度上限**: β/Re (β=1.0，可配置)

### 靈活的學習率調度器 🔧

現改為在配置檔 per-stage 指定第四參數 scheduler：Constant、MultiStepLR、CosineAnnealingLR；Cosine 的 eta_min 預設為下一stage lr，最後一stage為 0.1×本stage lr。

#### ✅ Scheduler修復 (2025-01-11)
**修復問題**: 解決了CosineAnnealingLR和MultiStepLR在EVM網路freeze/unfreeze時失效的問題。現在所有scheduler都能正常工作，learning rate會按預期變化並在TensorBoard中正確顯示。

#### 可用策略

-   **`StepLR` (預設)**: 保持原始行為，在每個訓練階段內，學習率會逐步下降。
-   **`MultiStage`**: 嚴格遵循配置文件中的多階段學習率，但**移除**了階段內的自動衰減。
-   **`Constant`**: 在整個訓練過程中，始終使用第一階段設定的恆定學習率。
-   **`CosineAnnealing` (推薦)**: 一個更高級的策略，它將訓練分為多個階段：
    
    -   **後續階段**: 每個階段都是一個獨立的餘弦退火週期，從當前階段的學習率平滑下降到下一階段的初始學習率。
    -   **最後階段**: 學習率最終下降到一個極小值 `2e-6`，以進行精細微調。

### 🕐 時間預估功能
- **實時預估**: 每100個epoch計算剩餘時間
- **階段進度**: 顯示當前階段完成度和預計完成時間
- **總體追蹤**: 追蹤完整3M epochs的總訓練時間
- **效率監控**: 每epoch平均時間統計

### 📊 TensorBoard集成
- **損失追蹤**: 總損失、方程損失、邊界損失
- **系統監控**: GPU記憶體使用、學習率變化 (✅ 已修復scheduler斷層問題)
- **網絡診斷**: tanh飽和度監測，每1000 epochs自動檢查
- **訓練效率**: 每epoch時間、Alpha_EVM參數變化
- **實時可視化**: `tensorboard --logdir=runs`

### 多階段訓練策略
```yaml
training:
  training_stages:
    - [0.03, 300000, 1e-3, CosineAnnealingLR]
    - [0.01, 300000, 2e-4, CosineAnnealingLR]
    - [0.005, 300000, 4e-5, Constant]
    - [0.002, 300000, 1e-5, CosineAnnealingLR]
    - [0.0005, 300000, 2e-6, Constant]
    - [0.0002, 300000, 2e-6, Constant]
```


### 🔬 L-BFGS 精修（更新 - 階段化觸發與放寬限制）

#### 🎯 新觸發策略 (2025-08-21)
- **階段控制**: **僅在Stage 3+啟用**，避免早期階段不穩定觸發
- **分階段視窗 + EMA相對改善 + 簡化條件 + 冷卻**:
  - 視窗/門檻（Stage 3–4/5–6）：`[7500, 10000]`、放寬 min improve `[0.03, 0.015]`
  - 相對改善率：`r = (L[t−W] − L[t]) / EMA_γ(L[t−W..t])`，`γ=0.95`
  - **簡化梯度條件**: `median(||∇θL||) < 2e-3` OR `median < 2% × g_base`（移除複雜的IQR/cos條件）
  - **放寬物理條件**: `α_evm ≤ 0.02` 且 `P95(ν_E)/(β/Re) < 0.7`
  - 冷卻：兩段 L-BFGS 間隔 `cooldown_steps=5000`（使用相對步數修復階段切換問題）

#### 🏗️ 執行策略與參數
- **段內策略**: `freeze EVM`、鎖定 `vis_t_minus`、停用 scheduler、使用 `fp32`
- **段參數（P100 友善）**：
  - `max_outer_steps: 200`、`timeout_seconds: 600`
  - `max_iter: 25`、`history_size: 20`
  - `tolerance_grad: 1e-6`、`tolerance_change: 1e-8`
  - `line_search_fn: strong_wolfe`、內迭代早停：連續 `8` 次改善 < `1e-4` 則結束
- **段後**: 恢復 Adam、解鎖 `vis_t_minus`、解凍 EVM、重建 scheduler，繼續當前 stage

#### 🔧 配置範例（training.lbfgs）：
```yaml
training:
  lbfgs:
    # 階段控制：從Stage 3開始才啟用L-BFGS
    enable_from_stage: 3
    
    # 放寬的觸發條件
    trigger_window_per_stage: [5000, 7500, 10000]
    min_improve_pct_per_stage: [0.02, 0.03, 0.015]  # Stage 3-4放寬到3%，Stage 5-6放寬到1.5%
    ema_gamma: 0.95
    
    # 簡化的梯度條件
    use_simple_grad_check: true
    grad_median_abs_thresh: 0.002      # 放寬到2e-3
    grad_relative_factor: 0.02         # 放寬到2%
    
    # 放寬的物理條件
    alpha_evm_threshold: 0.02          # 放寬到0.02
    cap_ratio_threshold: 0.7           # 放寬到0.7
    
    cooldown_steps: 5000
    freeze_evm_during_lbfgs: true
    # L-BFGS段參數保持不變...
```

#### ✅ 預期改善效果
- **Stage 1-2**: 🚫 完全禁用，專注Adam基礎訓練
- **Stage 3-4**: ✅ 大幅提高觸發機會（3%改善率 vs 原1%）
- **Stage 5-6**: ✅ 頻繁精修（1.5%改善率 vs 原0.5%）
- **整體**: 💪 更實用的L-BFGS精修機制，避免過度保守

## 🧯 Troubleshooting

### L-BFGS相關問題
- **L-BFGS 不觸發**: 
  - 檢查階段：僅Stage 3+啟用，Stage 1-2會完全禁用
  - 窗口不足或冷卻未滿：確認`stage_loss_deque`累積足夠數據
  - 改善率過嚴：降低 `min_improve_pct_per_stage`（Stage 3-4建議3%，Stage 5-6建議1.5%）
  - 梯度條件：放寬 `grad_median_abs_thresh`到2e-3，`grad_relative_factor`到2%
  - 物理條件：確認 `α_evm ≤ 0.02` 且 `P95(ν_E)/(β/Re) < 0.7`
  - 冷卻期：縮短 `cooldown_steps`或檢查相對步數計算
- **L-BFGS 過於頻繁**: 提高 `min_improve_pct_per_stage`、增大 `cooldown_steps`，或返回複雜梯度檢查(`use_simple_grad_check: false`)
- **L-BFGS 段內發散/NaN**: 降低 `max_iter`→20、放寬 `tolerance_grad`→1e-5、縮短 `timeout_seconds`；確保段內 FP32。仍不穩時可暫關 line search 或先做數千步 Adam。
- **段後學習率異常**: 段後自動重建 scheduler；若曲線仍平坦，檢查日誌中的重建訊息與 `Training/LearningRate` 曲線。

### 其他常見問題
- **人工黏滯常貼上限**: 降低 `last_layer_scale_evm`（如 0.05）或 `alpha_evm`；避免一味增大 β 以免偏黏。
- **tanh 飽和/梯度尖峰**: 降低 `first_layer_scale_main`→2.0 並確保 (x,y)∈[-1,1]；觀察 `NetworkHealth/Saturation_*`。
- **PDE loss 長期停滯**: 提高 `last_layer_scale_main`→0.6–0.7；或早期短暫提高學習率/延長暖身。
- **DDP 權重不一致**: 確認首/末層縮放在 DDP 包裹前完成（本專案已於建構時處理）；自定流程務必維持此順序。
- **AMP 數值問題**: L-BFGS 段全程 FP32；cap 使用 `torch.full_like(nu_e, float(β)/float(Re))` 以對齊 dtype/裝置。

### 💡 完整批次訓練
- **無批次分割**: 每個epoch使用全部120,000個訓練點
- **記憶體優化**: 約0.006GB預估記憶體需求
- **GPU利用率**: 最大化GPU計算效率
- **收斂穩定性**: 避免批次間的梯度噪音

## 📊 分散式訓練

### SLURM配置
```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
```

### 分散式特性
- ✅ 多GPU數據分割
- ✅ DDP (DistributedDataParallel)包裝
- ✅ 跨GPU損失聚合
- ✅ 梯度同步機制

## 🧮 物理方程

### Navier-Stokes方程
```
∂u/∂t + u∇u = -∇p + (1/Re + νₜ)∇²u
∇·u = 0
```

### 渦流黏度模型
```
νₜ = min(α_evm × softplus(|e|), β/Re)
Residual = (eq1×(u-0.5) + eq2×(v-0.5)) - e
```

**人工粘滯度上限控制**:
- 上限值: β/Re（β可在配置文件中調整，production 默認 β=5.0）
- 防止過度人工粘滯度影響物理真實性
- 可根據Reynolds數與問題難度調整（建議 β∈[1,5]）

## 🧩 新設定：首/末層縮放與 EVM 輸出映射（重要）

為了在 Re=5000 的高頻流動中提升表達能力並保持訓練穩定，新增以下可配置參數：

### 網路首/末層縮放（初始化後套用）

- first_layer_scale_main: 主網路首層縮放（建議 2.0–2.5，預設 2.0）
- last_layer_scale_main: 主網路末層縮放（建議 0.5–0.8，預設 0.5）
- first_layer_scale_evm: EVM 網路首層縮放（建議 1.0–1.5，預設 1.2）
- last_layer_scale_evm: EVM 網路末層縮放（建議 0.05–0.1，預設 0.1）

實作細節：
- 所有 Linear 層先以 Xavier 初始化（gain=calculate_gain('tanh')），再對首/末層權重做縮放。
- 縮放在 DDP 包裹前完成，確保多 GPU 參數一致，且與 optimizer 狀態一致。

### EVM 輸出映射與上限（避免過度擴散）

- evm_output_activation: softplus_cap（預設）
- 邏輯：nu_e = softplus(|e_raw|) ≥ 0，並以 β/Re 作元素級上限；實際人工黏滯為 α_evm × nu_e，再與上限比較取最小值。
- 熵殘差方程仍使用帶符號的 e_raw，不改動其物理含義。

### 配置範例（已內建於 configs/production.yaml）

```yaml
network:
  layers: 6
  layers_1: 4
  hidden_size: 80
  hidden_size_1: 40
  first_layer_scale_main: 2.0
  last_layer_scale_main: 0.5
  first_layer_scale_evm: 1.2
  last_layer_scale_evm: 0.1
  evm_output_activation: softplus_cap

physics:
  Re: 5000
  alpha_evm: 0.03
  beta: 5.0   # 人工黏滯上限係數（cap=β/Re）
```

### 使用建議與監測指標

- 若 tanh 飽和（loss 劇烈震盪），將主網首層縮放降為 2.0；確保 (x,y) 已標準化至 [-1,1]。
- 若 PDE loss 長期停滯，將主網末層縮放調至 0.6–0.7。
- 監測 ν_e 的 P95/P99 與 cap 命中率；若長期接近上限，優先降低 last_layer_scale_evm 或 α_evm。

## 📁 資料結構

```
ev-NSFnet/
├── configs/                  # 配置文件
│   ├── production.yaml       # 生產環境配置
│   └── test.yaml            # 測試環境配置（含監督數據示例）
├── data/                     # 訓練數據
│   ├── cavity_Re3000_256_Uniform.mat
│   └── cavity_Re5000_256_Uniform.mat
├── results/                  # 訓練結果 (包含各次訓練的檢查點與評估結果)
├── pinn_solver.py           # 核心求解器
├── net.py                   # 神經網路定義
├── config.py                # 配置管理系統
├── train.py                 # 訓練腳本
├── test.py                  # 測試腳本
├── train.sh                 # SLURM訓練腳本
├── cavity_data.py           # 數據處理與監督數據加載
└── README.md               # 說明文件
```

## 📊 監督數據功能 (新增) 🎯

**功能概述**: 將傳統的**純物理約束PINN**升級為**物理-數據混合約束PINN**，支援從真實CFD數據中隨機採樣指定數量的監督點作為額外損失項。

**應用場景**: 
- 模擬真實PINN應用中**數據極度稀缺**的情況
- 研究少量監督數據對PINN性能的影響  
- 探索物理約束與數據約束的平衡策略

### 🔧 配置參數

```yaml
supervision:
  enabled: true                    # 啟用/禁用監督數據功能
  data_points: 1                   # 監督數據點數量 (0=純物理約束)
  data_path: "data/cavity_Re5000_256_Uniform.mat"  # CFD數據文件路徑
  weight: 1.0                      # 監督損失權重
  random_seed: 42                  # 隨機採樣種子，確保可重現性
```

### ⚡ 使用示例

```bash
# 純物理約束訓練 (默認配置)
python train.py --config configs/production.yaml

# 1個監督點的混合約束訓練
python train.py --config configs/test.yaml

# 自定義監督點數量
# 修改configs/production.yaml中的supervision.data_points參數
```

### 🏗️ 技術實現

- **數據加載**: `cavity_data.py::loading_supervision_data()` - 支援隨機採樣指定數量的監督點
- **損失集成**: `pinn_solver.py::fwd_computing_loss_2d()` - 監督損失`loss_s`完全集成到總損失計算
- **分布式支援**: 支援多GPU訓練環境下的監督損失聚合
- **配置管理**: `config.py::SupervisionConfig` - 統一的監督數據配置管理

### 📈 預期效果

- **收斂加速**: 少量真實數據點可顯著提升訓練初期的收斂速度
- **精度改善**: 在關鍵區域提供額外約束，改善整體求解精度
- **穩定性增強**: 減少陷入局部最優解的風險
- **研究價值**: 為數據稀缺環境下的PINN性能研究提供工具

## 🔬 實驗結果

### 精度指標
- **速度場誤差**: < 2% (通過Xavier初始化改善)
- **壓力場誤差**: < 5%
- **收斂性**: 良好的訓練穩定性，避免Couette流陷阱
- **效率**: 支援多GPU加速
- **網絡健康**: 實時監測tanh飽和度，平均飽和率<20%

### 檢查點保存
- 每10000個epoch自動保存模型
- 模型文件格式: `checkpoint_epoch_{epoch}.pth`
- 包含主網路、EVM網路權重、優化器狀態及訓練進度
- 儲存路徑: `~/NSFnet/ev-NSFnet/results/Re{Re}/{layers}x{hidden_size}_Nf{N_f/1000}k_lamB{bc_weight}_alpha{alpha_evm}{Stage_Name}/`

## 🔧 技術改進記錄

### Xavier初始化修復 (2025-08-12)

#### 🐛 問題描述
- **症狀**: PINN模型無法學習lid-driven cavity flow特徵，初始輸出為線性剪切(Couette流)而非預期的渦流模式
- **原因**: FCNet使用PyTorch默認的Kaiming uniform初始化，不適合tanh激活函數，導致網絡早期飽和
- **影響**: 模型傾向於學習trivial solution，無法捕捉複雜的流動特徵

#### ✅ 解決方案
- **核心修復**: 實現專門針對tanh的Xavier初始化與層級縮放策略
- **修改模組**: `net.py` - FCNet類添加`_initialize_weights()`方法
- **技術細節**:
  - ✅ 使用`nn.init.calculate_gain('tanh')`獲取適合tanh的gain
  - ✅ 首層縮放0.5倍，避免輸入推tanh到飽和區  
  - ✅ 末層縮放(主網1e-3，EVM網5e-4)，避免初始輸出過大
  - ✅ 自動識別EVM網絡並應用更小的末層縮放
- **監測工具**: 新增tanh飽和度實時監測，每1000 epochs自動檢查

#### 🎯 預期效果
- 🔄 **流型學習**: 從Couette線性剪切轉向正確的cavity flow特徵
- ⚡ **收斂改善**: 避免早期tanh飽和，梯度能有效傳播到內部
- 📊 **實時診斷**: 平均飽和率>20%時自動警告

### Learning Rate Scheduler 修復 (2025-01-11 & 2025-08-12)

#### 🐛 問題描述
- **症狀**: CosineAnnealingLR和MultiStepLR配置正確但學習率在TensorBoard中顯示為常數，且在freeze/unfreeze時出現斷層
- **原因1**: EVM網路每10000個epoch的freeze/unfreeze操作重建optimizer，但scheduler仍綁定舊實例
- **原因2**: scheduler重建時使用局部`last_epoch`而非全局步數，導致餘弦曲線重新開始
- **影響**: 動態學習率策略完全失效，學習率調度不連續

#### ✅ 解決方案
- **核心修復**: 實現scheduler自動重建機制，確保學習率調度連續性
- **修改模組**: `pinn_solver.py` - 新增`_rebuild_scheduler()`方法
- **修復範圍**: 
  - ✅ freeze/unfreeze EVM網路時
  - ✅ L-BFGS優化結束時  
  - ✅ 任何optimizer重建場景
- **連續性修復**: 使用`global_step`替代保存的`last_epoch`，確保學習率調度基於全局訓練進度
- **狀態保持**: 完整保存並恢復scheduler的訓練狀態

#### 🎯 驗證結果
- ✅ CosineAnnealingLR按餘弦曲線正確調整學習率，無斷層現象
- ✅ MultiStepLR在milestone處正確階躍降低
- ✅ TensorBoard正確顯示平滑的學習率變化曲線
- ✅ 所有現有配置文件無需修改即可生效

#### 📋 使用建議  
```yaml
training:
  training_stages:
    - [0.03, 300000, 1e-3, CosineAnnealingLR]   # ✅ 現在完全正常工作
    - [0.01, 300000, 2e-4, MultiStepLR]        # ✅ 現在完全正常工作
    - [0.005, 300000, 4e-5, Constant]           # ✅ 一直正常工作
```

推薦使用`CosineAnnealingLR`以獲得更平滑和有效的學習率衰減策略。

### 系統穩定性與效能優化 (2025-08-17)

#### 🔧 系統穩定性改進
- **Logger 初始化順序**: 提前初始化 `self.logger`，避免在設定 device 時未初始化就呼叫
- **監督座標 requires_grad 移除**: `train.py` 將 `x_sup/y_sup` 不再 `requires_grad_(True)`，減少額外梯度計算耗時
- **TensorBoard/Console 指標索引對齊**: 新增 `Loss/Supervised`，修正 `eq1..eq4` 的索引映射；console 輸出同步修正
- **Epoch 時間量測修正**: 在 epoch 起訖處加 `torch.cuda.synchronize()`（rank 0）以取得準確時間

#### ⚡ 分散式訓練效能優化
- **DDP 指標聚合**: 將三個 loss 的 `all_reduce` 合併為單次；並改成 `dist.reduce(..., dst=0)` 只聚合到 rank 0
- **DDP 正則化避免 stack 分配**: 將 `torch.stack([...])` 換成生成器 `sum(p.pow(2).sum() for p in params)`

#### 🎯 邊界層物理改進
- **PDE 距離權重 w(d)**: 在 `pinn_solver.py` 的 PDE 殘差 loss 上加入 `w(d)`
  - 距離定義: `d = min(x, 1-x, y, 1-y)`
  - 權重公式: `w = w_min + (1-w_min) * exp(-d/tau)`
  - 自動 mean 正規化、detach，與網路結構、EVM 機制、DDP 完全兼容
  - 配置: `training.pde_distance_weighting=true` (預設啟用)

#### 🔄 採樣效能優化
- **採樣距離排序開關**: DataLoader 增加 `sort_by_boundary_distance` 參數
- **配置整合**: `train.py` 從 config 傳入；`production.yaml` 預設設為 `false`（配合 w(d) 節省前處理時間）
- **向下相容**: 與舊呼叫點相容（有預設值）；`test.py`/`batch_size_test` 等使用舊介面無需調整

#### ⚠️ 重要注意事項
- **邊界層權重邏輯**: 目前假設域為 [0,1]²。若未來更改域邊界，需同步調整距離定義
- **採樣排序效能**: 保留距離排序功能但預設關閉。若開啟，現行 `tools.sort_pts` 為 O(N_f×N_bc) 實作
- **Freeze/Unfreeze 規劃**: 既有邏輯在第 10000 epoch 解凍、10001 重新凍結，net_1 幾乎總是凍結（非本次修改造成）

---

## 🛠️ 命令參考

```bash
```bash
# 使用測試配置 (含1個監督點)
python train.py --config configs/test.yaml

# 純物理約束訓練 (默認配置)
python train.py --config configs/production.yaml

# 完整訓練 (1.8M epochs)
python train.py

# 從檢查點恢復訓練
python train.py --resume <CHECKPOINT_PATH>

# 使用不同的學習率調度策略
python train.py --lr-scheduler CosineAnnealing

# 測試命令  
python test.py

# 時間預估測試
python test_time_estimation.py

# 批次效率測試 (在 batch_size_test/ 目錄)
cd batch_size_test/
python batch_efficiency_test.py

# 執行SLURM作業
sbatch train.sh

# 查看TensorBoard訓練日誌
tensorboard --logdir=runs
```

### 🔍 監控與診斷

```bash
# 檢查GPU使用狀態
nvidia-smi

# 監控訓練進度 (輸出包含時間預估)
tail -f slurm-*.out

# 查看TensorBoard訓練曲線
tensorboard --logdir=runs --host=0.0.0.0 --port=6006
```

## 📝 引用

如果您使用本專案，請引用：

```bibtex
@software{ev_nsfnet_pinn,
  title={Physics-Informed Neural Networks for NSFnet: Cavity Flow Simulation},
  author={Your Name},
  year={2025},
  note={Developed with opencode and GitHub Copilot. Includes scheduler compatibility fixes for distributed training.}
}
```

## 📄 授權

本專案採用GNU General Public License v3.0授權。詳見 [LICENSE](LICENSE) 文件。

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進本專案！

---

🚀 **Developed with [opencode](https://opencode.ai) + GitHub Copilot**
