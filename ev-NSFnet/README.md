# ev-NSFnet (Restored Baseline)

> 基於 PINN 的 2D 方腔流 (lid-driven cavity) 測試與擴展實驗框架。此版本為精簡回復版，聚焦核心物理模型、分段式訓練 (staged training) 與改良化訓練日誌。 
>
> 本專案使用 opencode + GitHub Copilot 助力開發（透明開發流程聲明）。

---

## 🧪 實驗管理功能

### 📋 實驗標記與追踪
透過配置檔案管理不同實驗版本：

```yaml
experiment_name: NSFnet_Restore
description: Restored baseline with modern style logging
```

- **`experiment_name`**: 實驗識別名稱，用於檢查點檔名與結果目錄
- **`description`**: 實驗描述，記錄於日誌與 metadata 中

### 📊 結果組織結構
```
results/
├── Re5000/
│   ├── NSFnet_Restore/          # 以實驗名稱命名
│   │   ├── model_cavity_loop*.pth
│   │   ├── training_log.txt
│   │   └── metadata.json
│   └── test_result/
└── [其他Re數值]/
```

### 🔄 實驗版本控制
建議不同實驗使用獨特的 `experiment_name`：
- `NSFnet_Baseline`: 基準實驗
- `NSFnet_HighRes`: 高解析度版本  
- `NSFnet_Supervised`: 啟用監督學習版本
- `NSFnet_Custom`: 自定義參數實驗

---
## 🎯 專案目標
- 提供可重現的 Re=O(10^3–10^4) 方腔流 PINN 訓練流程。
- 以最少依賴與簡潔程式展示等效渦黏性 (eddy viscosity) 修正策略。
- 加入多階段 alpha_evm 收斂策略以提升穩定性。
- 提供高資訊密度、節奏可控的多行訓練日誌，便於長時監控與紀錄。

---
## 🧪 物理 / 模型簡述
| 項目 | 描述 |
|------|------|
| Governing Eq. | 2D 不可壓縮 Navier–Stokes (PINN 形式) |
| Turbulence Modifier | e (evm 分支) 對等效黏性調整：vis_t = min(vis_t0, α_evm * |e|) |
| Re_eff 定義 | `Re_eff = 1 / (1/Re + mean(vis_t))` (保持現行公式) |
| 網路結構 | 主網 (u,v,p) + EVM 支援子網 (e) (全連接 Tanh) |
| Loss 組成 | 邊界條件損失 + 方程殘差 (eq1..eq4；eq4 為熵型殘差) |

### 🧠 網路架構詳細參數

本專案採用雙網路架構設計，可透過配置檔調整：

#### 🎯 主網路 (u, v, p)
- **`layers`**: 主網路層數 (預設: 6)
- **`hidden_size`**: 主網路隱藏層神經元數 (預設: 80)
- 輸出: 速度場 (u, v) 與壓力場 (p)

#### ⚡ EVM 子網路 (e) 
- **`layers_1`**: EVM 子網路層數 (預設: 4)
- **`hidden_size_1`**: EVM 子網路隱藏層神經元數 (預設: 40)
- 輸出: 渦黏性修正項 (e)

#### ⚖️ 損失函數權重
- **`bc_weight`**: 邊界條件損失權重 (預設: 10)
- **`eq_weight`**: 方程殘差損失權重 (預設: 1)

```yaml
# 網路架構配置範例
network:
  layers: 6        # 主網路深度
  layers_1: 4      # EVM 子網路深度  
  hidden_size: 80  # 主網路寬度
  hidden_size_1: 40 # EVM 子網路寬度
```

> 💡 **調整建議**: 增加 `hidden_size` 可提升精度但增加計算成本；EVM 子網路通常設為主網路的 1/2 規模。

---
## 🏗️ 訓練階段 (預設)
每個 stage 調整 `alpha_evm`、學習率及 epochs；詳細可於 `configs/production.yaml` 修改。

| Stage | alpha | epochs | lr |
|-------|-------|--------|----|
| 1 | 0.05 | 500k | 1e-3 |
| 2 | 0.03 | 500k | 2e-4 |
| 3 | 0.01 | 500k | 4e-5 |
| 4 | 0.005| 500k | 1e-5 |
| 5 | 0.002| 500k | 2e-6 |
| 6 | 0.002| 500k | 2e-6 |

> 可依需求縮減 epochs 以做快速 smoke test。

---
## 📊 新版訓練日誌 (重點)
多行輸出 (預設 `log_interval=1000`)：
- 進度條 + 百分比
- 損失分解：總損失 / 方程總 / 邊界 / eq1~eq4 (eq4 標記熵殘差)
- 時間統計：階段耗時 / 平均 epoch 時間 / interval it/s / ETA / 累積總耗時
- GPU 記憶體：allocated / total / reserved
- Throughput：單 GPU 每秒處理點數 (boundary + equation)
- 物理量：目標 Re、Re_eff、alpha_evm（放大因子暫未啟用）

示例 (節錄)：
```
[Stage 1]    1000/500000   0.20% |██████▌                        |
  損失: total=1.23e-02  方程總=8.91e-03  邊界=3.40e-03
        eq1=1.2e-03 eq2=1.1e-03 eq3=9.5e-04 eq4(熵殘差)=4.0e-03
  時間: 本階段=2m15.2s  平均/epoch=0.14s  interval_it/s=7.12  平均it/s=7.45
        剩餘預估=11h32m  累積總時長=2m15.2s
  GPU : mem=512.3MB/24564MB (res 768.0MB)  throughput=142000.0 pts/s  lr=1.00e-03
  物理: 目標Re=5000  Re_eff=4785.4  alpha_evm=0.05  放大因子=N/A
--------------------------------------------------------------------------------
```

---
## 🚀 快速開始
```bash
# 建議使用虛擬環境 / conda
python train.py --config configs/production.yaml
```
可用參數：
- `--dry-run`：僅列印配置與 stages，不執行訓練。

配置重點（節選）：
- `training.normalize_coordinates`：是否將輸入座標 `(x,y)` 線性映射至 `[-1, 1]` 再送入網路。預設 `false`。

### 🧭 座標歸一化（可選）
- 切換位置：`configs/production.yaml` → `training.normalize_coordinates: true|false`
- 作用說明：將資料域 [0,1]^2 的 `(x,y)` 線性映射為 `[-1,1]^2` 後再餵入網路；有助於網路在早期更穩定地收斂。
- 方程一致性：歸一化是線性變換，PyTorch autograd 會自動套用鏈式法則，不需額外修改 PDE 殘差中的一階/二階導數計算。
- 預設值：`false`（關閉）。如需開啟，請將其設為 `true` 並重新訓練。

### 🎯 高級訓練參數

#### 📍 採樣點策略
- **`sort_by_boundary_distance`**: 是否按邊界距離排序採樣點
  - `true`: 優先處理接近邊界的點，有助於提升邊界條件收斂
  - `false`: 隨機順序處理，保持採樣多樣性
  - 預設值: `false`

#### ⚖️ PDE 距離加權機制
控制方程殘差根據空間位置動態調整權重：

- **`pde_distance_weighting`**: 啟用/停用 PDE 距離加權
  - `true`: 根據點到邊界距離調整損失權重
  - `false`: 使用統一權重處理所有 PDE 點
  - 預設值: `false`

- **`pde_distance_w_min`**: 最小權重係數 (0.8)
  - 遠離邊界區域的最小相對權重

- **`pde_distance_tau`**: 距離衰減參數 (0.2)  
  - 控制權重隨距離變化的衰減速率

```yaml
# 距離加權公式概念
weight = w_min + (1 - w_min) * exp(-distance / tau)
```

### 🧵 分散式與單 GPU
- 本專案已加入「條件式 DDP 包裝」：只有在 `torch.distributed` 已初始化且 `WORLD_SIZE>1` 時才會以 `DistributedDataParallel` 包裹網路。
- 單 GPU / 未啟動 `torchrun` 直接呼叫 `train.py` 或 `test.py` 不會再觸發 `Default process group has not been initialized` 錯誤。

分散式訓練示例 (4 GPUs)：
```bash
torchrun --nproc_per_node=4 train.py --config configs/production.yaml
```
(或使用 `python -m torch.distributed.run --nproc_per_node=4 ...`)

### 🔍 模型評估 / 批次推論
`test.py` 會依序載入各階段 / 迭代檢查點執行 `evaluate()` 與 `test()`：
```bash
python test.py
```
- 若未啟動分散式，程式會自動設置 `RANK=0, WORLD_SIZE=1`。
- 需要先確保對應 `./results/Re5000/.../model_cavity_loop*.pth` 檔案存在。
- 產出結果 `.mat` 於 `./NSFnet/ev-NSFnet/results/Re5000/test_result/`。

### 📊 監督學習功能 (DNS 數據整合)

本專案支援 DNS (Direct Numerical Simulation) 數據監督，可增強 PINN 的準確性：

#### 🎛️ 監督設定參數
```yaml
supervision:
  enabled: false                # 啟用監督學習
  data_points: 0                # 使用的監督點數量
  data_path: data/cavity_Re5000_256_Uniform.mat  # DNS 數據源
  weight: 10.0                  # 監督損失權重
  random_seed: 42               # 採樣隨機種子
```

#### 🔧 使用方式
1. **準備 DNS 數據**: 將 `.mat` 檔案放置到指定路徑
2. **啟用監督**: 設定 `enabled: true`  
3. **調整點數**: 設定 `data_points > 0` (建議 1000-10000)
4. **權重調整**: 根據數據品質調整 `weight` 值

#### ⚡ 監督學習優勢
- **加速收斂**: 減少純物理方程訓練的不穩定性
- **提升準確性**: 直接約束解在已知點的數值
- **控制過擬合**: 透過權重平衡 DNS 約束與 PDE 殘差

> 💡 **使用建議**: 初期階段可使用較高 DNS 權重，後期降低以避免過度依賴數據點。

### ❗ 常見問題
| 問題 | 排查 | 解法 |
|------|------|------|
| 找不到 checkpoint | 路徑字串與 Stage 名稱不符 | 確認目錄層級與 `alpha_evm` 值 | 
| 單 GPU 仍報 DDP 初始化錯 | 早期舊版本 `pinn_solver.py` | 重新更新至本版 / 清除舊快取 |
| CUDA OOM | `N_f` 過大 | 降低 `N_f` 或縮小 hidden size |

---
## ⚙️ 主要程式檔案
| 檔案 | 說明 |
|------|------|
| `train.py` | 訓練入口，驅動階段式流程與資料載入 |
| `pinn_solver.py` | 核心 PINN 類別與訓練迴圈、日誌輸出 |
| `cavity_data.py` | 方腔流資料讀取與網格抽樣 |
| `net.py` | 基本全連接網路定義 |
| `config.py` | YAML 配置解析與資料類型封裝 |
| `logger.py` | 簡易格式化輸出工具 |

---
## 📐 設計原則
- 保持理論簡潔：無多餘 scheduler / 混合監督分支（可後續擴展）。
- 清晰可讀：函式職責單一，避免過度抽象。
- 控制台輸出可長期追蹤：避免大量即時刷新覆蓋。
- 可擴展：後續可加入自動收斂判據 / 自訂放大因子。

---
## 🔍 待擴充 (Roadmap)
- 放大因子公式（若需引入 Re 比值或能量尺度）
- 自動早停條件 (loss plateau / Re_eff 收斂率)
- 更細緻資料分批 (mini-batch PDE 點) 以支援更大 N_f
- 結果可視化腳本 (streamline / vorticity)

---
## ⚙️ 完整配置參數參考

### 📊 參數速查表

| 分類 | 參數 | 預設值 | 說明 |
|------|------|--------|------|
| **🧪 實驗** | `experiment_name` | `NSFnet_Restore` | 實驗標識名稱 |
| | `description` | - | 實驗描述文字 |
| **🔬 物理** | `Re` | `5000` | Reynolds 數 |
| | `alpha_evm` | `0.05` | EVM 參數 |
| | `bc_weight` | `10` | 邊界條件損失權重 |
| | `eq_weight` | `1` | 方程殘差損失權重 |
| **🧠 網路** | `layers` | `6` | 主網路層數 |
| | `layers_1` | `4` | EVM 子網路層數 |
| | `hidden_size` | `80` | 主網路隱藏層大小 |
| | `hidden_size_1` | `40` | EVM 子網路隱藏層大小 |
| **🎯 訓練** | `N_f` | `120000` | PDE 採樣點數 |
| | `sort_by_boundary_distance` | `false` | 按邊界距離排序 |
| | `normalize_coordinates` | `false` | 座標正規化 |
| | `pde_distance_weighting` | `false` | PDE 距離加權 |
| | `pde_distance_w_min` | `0.8` | 最小權重係數 |
| | `pde_distance_tau` | `0.2` | 距離衰減參數 |
| **📊 監督** | `enabled` | `false` | 啟用監督學習 |
| | `data_points` | `0` | 監督點數量 |
| | `weight` | `10.0` | 監督損失權重 |
| | `data_path` | `.mat` | DNS 數據路徑 |

### 🔧 參數調整建議

#### 🚀 性能優化
- **加速訓練**: 降低 `N_f` (60000-80000)、減少網路規模
- **提升精度**: 增加 `hidden_size` (120-160)、啟用座標正規化
- **記憶體控制**: 調整 `N_f` 與網路大小的平衡

#### ⚖️ 損失平衡  
- **邊界主導**: 提高 `bc_weight` (15-20)
- **方程主導**: 提高 `eq_weight` (2-5) 
- **監督整合**: 設置 `supervision.weight` (5-15)

#### 🎯 收斂策略
- **穩定收斂**: 啟用 `normalize_coordinates`
- **邊界優化**: 啟用 `sort_by_boundary_distance` 
- **區域平衡**: 啟用 `pde_distance_weighting`

---
## 🧾 授權
原始作者標註於檔頭（GPLv3 版權宣告保持）。本回復與增補版本遵循同授權。

---
## 🤝 開發聲明
本專案部分內容藉由 opencode 代理協助與 GitHub Copilot 補全；所有修改均經人工審閱以維持物理一致性與可解釋性。

---
## 🛰️ 引用
若本程式對研究有幫助，請於論文或報告中引用並標示來源與 GPLv3 授權性質。

---
更新時間：2025-09-11
