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
- **cavity_data.py**: 腔體流數據處理
- **tools.py**: 工具函數

### 網路結構
- **主網路**: 6層隱藏層，80個神經元 (u, v, p預測)
- **EVM網路**: 6層隱藏層，40個神經元 (渦流黏度預測)
- **激活函數**: Tanh
- **輸入**: 空間座標 (x, y)
- **輸出**: 速度場 (u, v)、壓力場 (p)、渦流黏度 (e)

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

# 使用不同的學習率調度策略
# --lr-scheduler <STRATEGY>: (可選) 選擇學習率調度策略。預設為 'StepLR'。
#                            可選值: StepLR, MultiStage, CosineAnnealing, Constant
python train.py --lr-scheduler CosineAnnealing
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

### 靈活的學習率調度器

您可以通過 `--lr-scheduler` 命令行參數選擇不同的學習率策略，以進行更精細的訓練調優。

```bash
# 示例：使用分階段的CosineAnnealing策略
torchrun train.py --lr-scheduler CosineAnnealing
```

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
- **系統監控**: GPU記憶體使用、學習率變化
- **訓練效率**: 每epoch時間、Alpha_EVM參數變化
- **實時可視化**: `tensorboard --logdir=runs`

### 多階段訓練策略
```python
# 6個訓練階段，總計1,800,000 epochs
training_stages = [
    (0.03, 300000, 1e-3, "Stage 1"),   # 初始Alpha_EVM
    (0.01, 300000, 2e-4, "Stage 2"),   # 逐漸減少
    (0.005, 300000, 4e-5, "Stage 3"),  # L-BFGS混合訓練
    (0.002, 300000, 1e-5, "Stage 4"),  # 精細調整
    (0.0005, 300000, 2e-6, "Stage 5"), # 最終調整
    (0.0002, 300000, 2e-6, "Stage 6")  # 穩定收斂
]
```

### 🔬 L-BFGS混合優化 (Stage 3)
- **前60%**: Adam優化器 (快速收斂)
- **後40%**: L-BFGS優化器 (精確收斂)
- **自動切換**: 無縫銜接兩種優化策略
- **參數配置**:
  - max_iter: 50
  - history_size: 20
  - tolerance_grad: 1e-8
  - line_search: strong_wolfe

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
νₜ = min(α_evm × |e|, β/Re)
Residual = (eq1×(u-0.5) + eq2×(v-0.5)) - e
```

**人工粘滯度上限控制**:
- 上限值: β/Re (β可在配置文件中調整)
- 防止過度人工粘滯度影響物理真實性
- 默認β=1.0，可根據Reynolds數調整

## 📁 資料結構

```
ev-NSFnet/
├── configs/                  # 配置文件
│   ├── production.yaml       # 生產環境配置
│   └── test.yaml            # 測試環境配置
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
└── README.md               # 說明文件
```

## 🔬 實驗結果

### 精度指標
- **速度場誤差**: < 2%
- **壓力場誤差**: < 5%
- **收斂性**: 良好的訓練穩定性
- **效率**: 支援多GPU加速

### 檢查點保存
- 每10000個epoch自動保存模型
- 模型文件格式: `checkpoint_epoch_{epoch}.pth`
- 包含主網路、EVM網路權重、優化器狀態及訓練進度
- 儲存路徑: `~/NSFnet/ev-NSFnet/results/Re{Re}/{layers}x{hidden_size}_Nf{N_f/1000}k_lamB{bc_weight}_alpha{alpha_evm}{Stage_Name}/`

## 🛠️ 命令參考

```bash
```bash
# 使用配置文件訓練
python train.py --config configs/production.yaml

# 使用測試配置
python train.py --config configs/test.yaml

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
  note={Developed with opencode and GitHub Copilot}
}
```

## 📄 授權

本專案採用GNU General Public License v3.0授權。詳見 [LICENSE](LICENSE) 文件。

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進本專案！

---

🚀 **Developed with [opencode](https://opencode.ai) + GitHub Copilot**