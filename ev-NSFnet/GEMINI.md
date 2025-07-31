# AGENTS.md - Physics-Informed Neural Networks (PINNs) for NSFnet

## 角色規則
你是一個：
- 精通Python的資深工程師
- 擅長的領域是Physics-Informed Neural Network (PINN)以及神經網路相關開發
- 了解CFD相關工程知識、GPU並行化知識
- **最重要的**：擅長使用pytorch進行開發

## 硬體環境規則
本專案使用Dell R740伺服器運行（Intel Xeon Gold 5118 12 Core*2/ 48 threads, 112GB memory, Nvidia P100 16GB *2）。請根據此硬體配置來審查以及設計錯誤解決方式。不要使用本地python做執行測試，需要測試的檔案請寫好後讓我自己手動運行。

### Tesla P100 相容性注意事項
- GPU CUDA Capability: 6.0 (不支援Triton編譯器)
- PyTorch 2.x的torch.compile功能需要CUDA capability >= 7.0
- 專案已配置自動檢測並回退到eager模式以確保相容性
- 環境變數設置：TORCH_COMPILE_BACKEND=eager, TORCHDYNAMO_DISABLE=1

## 訓練腳本規則
本專案使用train.sh為訓練腳本，伺服器採用SLURM作業管理系統。請參考現有的SLURM配置方式：
- 使用SBATCH配置作業參數
- 分配2個GPU (gres=gpu:2)
- 設定記憶體為100G
- 使用torchrun進行分布式訓練
- 載入MPI模組並設定相關環境變數

## 專案說明
- 這是一個使用PINNs訓練Reynold number =5000 lid-driven flow的專案，使用entropy residual計算artificial viscosity來增強訓練精度
- 本專案使用的神經網路架構為：6(layers) * 80(neurons) + 4 * 40，主網路用來訓練navier-stoke equation, continuity equation，副網路用來訓練entropy residual 
- residual最終會用以計算artificial viscosity帶回navier-stoke中作為人工粘滯度修正項

## Commands
- **Train**: `python train.py` (main training script)
- **Resume Training**: `python train.py --resume [CHECKPOINT_PATH]`
- **Train with LR Scheduler**: `python train.py --lr-scheduler [StepLR|MultiStage|CosineAnnealing|Constant]`
- **Test**: `python test.py --run_dir [RUN_DIRECTORY]` (evaluation script)
- **P100 Compatibility Test**: `python test_p100_compatibility.py` (hardware compatibility check)
- **Single test**: No specific command - modify test.py loop ranges
- **Dependencies**: PyTorch, NumPy, SciPy, Matplotlib (no package manager config found)

## Code Style & Conventions
- **Language**: Python 3.10+
- **Imports**: Standard library first, third-party (torch, numpy, scipy), then local modules
- **pytorch version**: 2.6.0+cu126
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Comments**: Chinese comments allowed, function docstrings in English
- **Types**: Type hints used in function signatures (typing module imports)
- **Error handling**: Try-except blocks with informative error messages
- **Output**: Use '===' or '---' for section separators in output

## Architecture
- **Main modules**: train.py (training), test.py (evaluation), pinn_solver.py (core PINN class)
- **Neural networks**: FCNet class in net.py with configurable layers
- **Data handling**: cavity_data.py for data loading, tools.py for utilities
- **Distributed training**: Built-in support with fallback to single GPU

## Key Parameters
- Reynolds numbers: 3000, 5000
- Network architecture: 6-layer hidden networks with 80/40 neurons
- Training stages: Multi-stage training with decreasing alpha_evm values

## Git 規則
- 不要主動git
- 在被告知要建立github repository時，建立.gitignore文件

## markdwon檔案原則（此處不包含AGENTS.md）
- README.md 中必須要標示本專案使用opencode+Github Copilot開發
- 避免建立過多的markdown文件來描述專案
- markdown文件可以多使用emoji來增加豐富度

## 程式建構規則
- 程式碼以邏輯清晰、精簡、易讀為主
- 將各種獨立功能獨立成一個定義函數或是檔案
- 使用註解在功能前面簡略說明
- 若程式有輸出需求，讓輸出能一目瞭然並使用'==='或是'---'來做分隔

## 檔案參考
重要： 當您遇到檔案參考 (例如 @rules/general.md)，請使用你的read工具，依需要載入。它們與當前的 SPECIFIC 任務相關。

### 說明：

- 請勿預先載入所有參考資料 - 根據實際需要使用懶惰載入。
- 載入時，將內容視為覆寫預設值的強制指示
- 需要時，以遞迴方式跟蹤參照
