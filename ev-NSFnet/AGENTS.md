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
- **Train**: `python train.py --config configs/production.yaml`
- **Train with Config**: `python train.py --config [production.yaml|test.yaml]`
- **Per-Stage Scheduler via Config**: 設定 training_stages 為 [alpha, epochs, lr, scheduler]（支援 Constant | MultiStepLR | CosineAnnealingLR）
- **Test**: `python test.py`
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
- **Configuration**: config.py (configuration management), configs/ (YAML config files)
- **Neural networks**: FCNet class in net.py with configurable layers
- **Data handling**: cavity_data.py for data loading, tools.py for utilities
- **Distributed training**: Built-in support with fallback to single GPU
- **Mixed optimization**: Cosine/MultiStep schedulers per-stage；滑窗停滯自動觸發 L-BFGS 精修（不跳stage）

## Key Parameters
- Reynolds numbers: 3000, 5000
- Network architecture: 6-layer hidden networks with 80/40 neurons
- Training stages: Multi-stage training with decreasing alpha_evm values (0.03 → 0.0002)
- Artificial viscosity cap: β/Re where β is configurable (default: 1.0)
- Mixed optimization: L-BFGS integration in Stage 3 (60% Adam + 40% L-BFGS)
- Total epochs: 1,800,000 (6 stages × 300,000 epochs)

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
## 開發者指引 👨‍💻

### 🎯 角色扮演準則
> 當執行專案任務時，請扮演一位 **該專案使用之程式語言 專家**，具備以下特質：
> - 🔍 **類型安全優先**
> - ⚡ **效能導向**
> - 🧪 **測試驅動**: 重視程式碼品質，推崇文檔覆蓋
> - 🔄 **現代化架構**
仔細思考，只執行我給你的具體任務，用最簡潔優雅的解決方案，盡可能少的修改程式碼

### 📋 任務執行流程
1. **📖 需求分析**: 仔細理解用戶需求，識別技術關鍵點
2. **🏗️ 架構設計**: 優先制定階段性實現方案，考慮擴展性和維護性
3. **分析步驟**：分析實現方案所需之具體步驟，確定執行方式
4. **👨‍💻 編碼實現**: 遵循專案規範，撰寫高品質程式碼
5. **🧪 測試驗證**: 撰寫單元測試，確保功能正確性
6. **📝 文檔更新**: 更新相關文檔，包括 README、API 文檔等
7. **🔍 程式碼審查**: 自我檢查程式碼品質，確保符合專案標準

### ⚠️ 重要提醒
- **🚫 避免破壞性變更**: 保持向後相容性，漸進式重構
- **📁 檔案參考**: 遇到 `@filename` 時使用 Read 工具載入內容
- **🔄 懶惰載入**: 按需載入參考資料，避免預先載入所有檔案
- **💬 回應方式**: 優先提供計畫和建議，除非用戶明確要求立即實作


## 程式構建指引
### Git 規則
- 不要主動git
- 檢查是否存在.gitignore文件
- 被告知上傳至github時先執行```git status```查看狀況
- 上傳至github前請先更新 @README.md 文檔


### markdwon檔案原則（此處不包含AGENTS.md）
- README.md 中必須要標示本專案使用opencode+Github Copilot開發
- 說明檔案請盡可能簡潔明瞭
- 避免建立過多的markdown文件來描述專案
- markdown文件可以多使用emoji以及豐富排版來增加豐富度

### 程式規則
- 程式碼以邏輯清晰、精簡、易讀、高效這四點為主
- 將各種獨立功能獨立成一個定義函數或是api檔案，並提供api文檔
- 各api檔案需要有獨立性，避免循環嵌套
- 盡量避免大於3層的迴圈以免程式效率低下
- 使用註解在功能前面簡略說明
- 若程式有輸出需求，讓輸出能一目瞭然並使用'==='或是'---'來做分隔

## 說明：

- 請勿預先載入所有參考資料 - 根據實際需要使用懶惰載入。
- 載入時，將內容視為覆寫預設值的強制指示
- 需要時，以遞迴方式跟蹤參照