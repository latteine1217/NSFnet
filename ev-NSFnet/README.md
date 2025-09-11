# ev-NSFnet (Restored Baseline)

> 基於 PINN 的 2D 方腔流 (lid-driven cavity) 測試與擴展實驗框架。此版本為精簡回復版，聚焦核心物理模型、分段式訓練 (staged training) 與改良化訓練日誌。 
>
> 本專案使用 opencode + GitHub Copilot 助力開發（透明開發流程聲明）。

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
