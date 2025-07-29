# TSA 激活函數 API 文檔 - 改進版

## 概述

本 API 實現了基於論文 "Physics-informed neural networks with trainable sinusoidal activation functions for approximating the solutions of the Navier-Stokes equations" 的可訓練正弦激活函數 (Trainable Sinusoidal Activation, TSA)。

TSA 激活函數特別適用於物理資訊神經網路 (Physics-Informed Neural Networks, PINNs)，能夠顯著提升網路在流體力學等物理問題上的表現。

## 核心特性

- 🚀 **神經元級別的獨立頻率參數**：每個神經元擁有自己的可訓練頻率
- 📈 **動態斜率恢復機制**：透過頻率正規化防止梯度消失問題
- 🔄 **正弦餘弦組合**：提供完整的函數表示能力
- 🛠️ **靈活整合**：可輕鬆整合到現有 PINN 架構中

## 安裝與導入

```python
# 將 tsa_activation.py 放入您的專案目錄
from tsa_activation import TSAActivation, SlopeRecovery, TSALinear, create_tsa_network
```

---

## API 參考

### 1. TSAActivation

**核心的可訓練正弦激活函數類**

#### 類別定義
```python
TSAActivation(num_neurons, freq_init_std=1.0, zeta1=1.0, zeta2=1.0, 
              trainable_zeta=False, freq_init_mean=0.0)
```

#### 參數說明

| 參數名稱 | 類型 | 預設值 | 說明 |
|---------|------|-------|------|
| `num_neurons` | int | - | 該層神經元數量（必填） |
| `freq_init_std` | float | 1.0 | 頻率參數初始化的標準差 |
| `zeta1` | float | 1.0 | 正弦分量的係數 |
| `zeta2` | float | 1.0 | 餘弦分量的係數 |
| `trainable_zeta` | bool | False | zeta 係數是否可訓練 |
| `freq_init_mean` | float | 0.0 | 頻率參數初始化的平均值 |

#### 數學公式
```
a_i = ζ₁ × sin(f_i × z_i) + ζ₂ × cos(f_i × z_i)
```
其中：
- `f_i` 是第 i 個神經元的可訓練頻率參數
- `z_i` 是第 i 個神經元的預激活輸出
- `ζ₁, ζ₂` 是正弦和餘弦分量的係數

#### 主要方法

##### `forward(x)`
執行前向傳播

**參數**：
- `x` (torch.Tensor): 輸入張量，形狀為 `(batch_size, num_neurons)`

**返回**：
- torch.Tensor: 激活後的張量，形狀與輸入相同

**範例**：
```python
tsa = TSAActivation(num_neurons=50)
x = torch.randn(32, 50)  # 批次大小 32，特徵維度 50
output = tsa(x)
print(output.shape)  # torch.Size([32, 50])
```

##### `get_frequencies()`
獲取當前頻率參數值

**返回**：
- torch.Tensor: 當前的頻率參數

##### `set_frequencies(new_frequencies)`
手動設定頻率參數值

##### `get_activation_stats()`
獲取激活函數的統計資訊

---

### 2. SlopeRecovery

**斜率恢復機制類 - 頻率正規化項**

SlopeRecovery 並非直接計算或恢復激活函數的斜率，而是一個**正規化項**，透過約束頻率參數 `f_i` 保持適當的值，間接維持激活函數的非零斜率，從而緩解梯度消失問題。

#### 工作原理

1. **問題背景**：當頻率參數 `f_i` 過小時，TSA 激活函數的斜率會接近零，導致梯度消失
2. **解決策略**：透過在損失函數中添加正規化項，鼓勵頻率參數保持較大的值
3. **機制**：計算所有 TSA 層頻率參數的指數平均，作為正規化約束項

#### 類別定義
```python
SlopeRecovery(tsa_layers)
```

#### 參數說明

| 參數名稱 | 類型 | 說明 |
|---------|------|------|
| `tsa_layers` | list | TSAActivation 層的列表 |

#### 數學公式

**多層情況** (L > 1)：
```
S(a) = 1 / (1/(L-1) × Σ exp(1/N_k × Σ f_k^i))
```

**單層情況** (L = 1)：
```
S(a) = 1 / exp(1/N_1 × Σ f_1^i)
```

其中：
- `L` 是 TSA 層的總數
- `N_k` 是第 k 層的神經元數量
- `f_k^i` 是第 k 層第 i 個神經元的頻率參數

> **重要說明**：當網路中只有一個 TSA 層時 (L=1)，公式會自動調整為避免除以零的情況。

#### 主要方法

##### `forward()`
計算斜率恢復項

**返回**：
- torch.Tensor: 標量張量，表示頻率正規化損失

**使用方式**：
```python
# 在損失函數中加入頻率正規化項
total_loss = data_loss + physics_loss + lambda_slope * slope_recovery()
```

**完整範例**：
```python
# 建立網路和斜率恢復模組
tsa_layers = [TSAActivation(50), TSAActivation(50)]
slope_recovery = SlopeRecovery(tsa_layers)

# 在訓練循環中使用
lambda_slope = 0.01  # 頻率正規化權重
slope_loss = slope_recovery()
total_loss = data_loss + physics_loss + lambda_slope * slope_loss
```

---

### 3. TSALinear

**整合線性層和 TSA 激活的便利類**

#### 類別定義
```python
TSALinear(in_features, out_features, bias=True, freq_init_std=1.0, 
          zeta1=1.0, zeta2=1.0, trainable_zeta=False)
```

#### 使用範例
```python
# 替代 nn.Linear + 激活函數的組合
tsa_linear = TSALinear(in_features=10, out_features=50)
x = torch.randn(32, 10)
output = tsa_linear(x)  # 自動執行線性變換和 TSA 激活
```

---

### 4. 工具函數

#### `create_tsa_network(layer_sizes, freq_init_std=1.0, ...)`

**快速建立完整的 TSA 網路**

**重要說明**：此函數會為**所有隱藏層的線性變換之後**添加 TSA 激活函數，而**不包括最終輸出層**。

**網路結構**：
```
輸入 → Linear → TSA → Linear → TSA → ... → Linear (輸出層，無激活)
```

**參數**：
- `layer_sizes` (list): 層的大小列表，如 `[input_dim, hidden1, hidden2, output_dim]`

**返回**：
- tuple: `(layers, slope_recovery)` 的元組

**範例**：
```python
# 建立 3 輸入、2 輸出、兩個隱藏層各 50 神經元的網路
# 結構：3 → Linear(50) → TSA → Linear(50) → TSA → Linear(2)
layers, slope_recovery = create_tsa_network([3, 50, 50, 2])

class MyTSAPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers, self.slope_recovery = create_tsa_network([3, 50, 50, 2])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_slope_loss(self):
        return self.slope_recovery()
```

#### `apply_glorot_initialization(model)`

**套用 Glorot 正態初始化**

根據論文建議，對線性層套用 Xavier/Glorot 初始化。

---

## 完整使用範例

### 基礎使用

```python
import torch
import torch.nn as nn
from tsa_activation import TSAActivation, SlopeRecovery

class SimpleTSAPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 隱藏層 1：線性變換 + TSA 激活
        self.layer1 = nn.Linear(3, 50)
        self.tsa1 = TSAActivation(50, freq_init_std=1.0)
        
        # 隱藏層 2：線性變換 + TSA 激活
        self.layer2 = nn.Linear(50, 50)
        self.tsa2 = TSAActivation(50, freq_init_std=1.0)
        
        # 輸出層：僅線性變換，無激活函數
        self.output_layer = nn.Linear(50, 2)
        
        # 頻率正規化模組
        self.slope_recovery = SlopeRecovery([self.tsa1, self.tsa2])
    
    def forward(self, x):
        x = self.tsa1(self.layer1(x))
        x = self.tsa2(self.layer2(x))
        x = self.output_layer(x)
        return x
    
    def get_frequency_regularization_loss(self):
        """獲取頻率正規化損失項"""
        return self.slope_recovery()

# 使用模型
model = SimpleTSAPINN()
x = torch.randn(32, 3)
output = model(x)
freq_reg_loss = model.get_frequency_regularization_loss()
```

### 訓練循環範例

```python
import torch.optim as optim

# 建立模型和優化器
model = SimpleTSAPINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 訓練參數
lambda_freq_reg = 0.01  # 頻率正規化權重

# 訓練循環
for epoch in range(1000):
    optimizer.zero_grad()
    
    # 前向傳播
    predictions = model(input_data)
    
    # 計算損失分量
    data_loss = mse_loss(predictions, target_data)
    physics_loss = compute_physics_residual(model, collocation_points)
    freq_reg_loss = model.get_frequency_regularization_loss()
    
    # 總損失
    total_loss = data_loss + physics_loss + lambda_freq_reg * freq_reg_loss
    
    # 反向傳播
    total_loss.backward()
    optimizer.step()
    
    # 監控訓練進度
    if epoch % 100 == 0:
        print(f'Epoch {epoch}:')
        print(f'  Data Loss: {data_loss.item():.6f}')
        print(f'  Physics Loss: {physics_loss.item():.6f}')
        print(f'  Freq Reg Loss: {freq_reg_loss.item():.6f}')
        print(f'  Total Loss: {total_loss.item():.6f}')
        
        # 監控 TSA 參數統計
        tsa1_stats = model.tsa1.get_activation_stats()
        print(f'  TSA1 頻率範圍: {tsa1_stats["freq_min"]:.3f} ~ {tsa1_stats["freq_max"]:.3f}')
```

---

## 參數調優建議

### 頻率初始化 (`freq_init_std`)

| 問題類型 | 建議值 | 說明 |
|---------|-------|------|
| 震盪性問題 (如圓柱繞流) | 1.0-2.0 | 適合捕獲週期性現象 |
| 穩態問題 | 0.5-1.0 | 較穩定的收斂 |
| 高 Reynolds 數湍流 | 1.0-2.0 | 需要更高頻率分量 |
| 多尺度問題 | 0.5-1.5 | 平衡不同尺度特徵 |

### 頻率正規化權重 (`lambda_freq_reg`)

| 網路深度 | 建議值 | 說明 |
|---------|-------|------|
| 淺層網路 (< 5 層) | 0.001-0.01 | 避免過度約束頻率參數 |
| 中等深度 (5-10 層) | 0.01-0.1 | 平衡穩定性和表達能力 |
| 深層網路 (> 10 層) | 0.1-1.0 | 強化頻率約束防止梯度消失 |

**調參策略**：
1. 從較小值開始 (0.001)
2. 監控頻率參數統計，如果 `freq_mean` 持續下降至接近零，增大權重
3. 如果收斂困難，可能權重過大，適當減小

### Zeta 係數建議

- **標準設定**: `zeta1=1.0, zeta2=1.0` (等權重正弦餘弦分量)
- **強調週期性**: `zeta1=1.5, zeta2=0.5` (更重視正弦分量)
- **強調平滑性**: `zeta1=0.5, zeta2=1.5` (更重視餘弦分量)
- **可訓練模式**: `trainable_zeta=True` (適用於複雜問題，讓網路自動學習最佳係數)

---

## 常見問題與解答

### Q1: 如何選擇合適的頻率初始化參數？
**A**: 
- 對於震盪性強的問題（如流體中的渦流），建議 `freq_init_std=1.0-2.0`
- 對於平滑問題，建議 `freq_init_std=0.5-1.0`
- 可以透過監控 `get_activation_stats()` 來觀察頻率分佈的演化

### Q2: 頻率正規化權重如何設定？
**A**: 
- 開始時設定較小值 (0.001-0.01)
- 監控訓練過程中的頻率統計，如果頻率參數趨向於零，增大權重
- 如果損失下降緩慢，可能權重過大，適當減小

### Q3: TSA 激活函數會顯著增加計算成本嗎？
**A**: 
- 單次前向傳播會增加約 20-30% 的計算時間
- 但由於收斂速度通常更快，總訓練時間往往更短
- 頻率參數數量與神經元數量成正比，對記憶體影響較小

### Q4: 可以只在部分層使用 TSA 激活嗎？
**A**: 
- 完全可以，您可以混合使用不同的激活函數
- 建議在處理複雜非線性的隱藏層使用 TSA
- 輸出層通常使用線性激活或根據問題特性選擇

### Q5: 單層 TSA 網路需要使用 SlopeRecovery 嗎？
**A**: 
- 對於單層 TSA 網路，SlopeRecovery 仍然有用，可以防止頻率參數過小
- 程式碼會自動處理單層情況，避免數學上的除以零問題
- 權重可以設定得相對較小 (0.001-0.01)

### Q6: 如何判斷 TSA 網路訓練是否正常？
**A**: 
- 監控頻率統計：`freq_mean` 應保持在合理範圍內 (不應趨向於零)
- 觀察損失下降：相比傳統激活函數，應有更穩定的收斂
- 檢查梯度：不應出現梯度爆炸或完全消失的情況

---

## 進階使用技巧

### 1. 動態調整頻率正規化權重

```python
class AdaptiveTSAPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 網路定義
        self.initial_freq_reg_weight = 0.01
        self.current_freq_reg_weight = self.initial_freq_reg_weight
    
    def adjust_freq_reg_weight(self, epoch):
        """根據訓練進度動態調整頻率正規化權重"""
        avg_freq = self.get_average_frequency()
        
        if avg_freq < 0.1:  # 頻率過小
            self.current_freq_reg_weight *= 1.1
        elif avg_freq > 5.0:  # 頻率過大
            self.current_freq_reg_weight *= 0.9
    
    def get_average_frequency(self):
        """計算所有 TSA 層的平均頻率"""
        total_freq = 0.0
        total_neurons = 0
        for layer in self.modules():
            if isinstance(layer, TSAActivation):
                total_freq += layer.frequencies.abs().mean().item()
                total_neurons += 1
        return total_freq / total_neurons if total_neurons > 0 else 0.0
```

### 2. 頻率參數的可視化分析

```python
import matplotlib.pyplot as plt

def visualize_frequency_evolution(model, epochs, freq_history):
    """可視化頻率參數的演化過程"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 頻率均值演化
    axes[0, 0].plot(epochs, [h['freq_mean'] for h in freq_history])
    axes[0, 0].set_title('頻率平均值演化')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('頻率平均值')
    
    # 頻率標準差演化
    axes[0, 1].plot(epochs, [h['freq_std'] for h in freq_history])
    axes[0, 1].set_title('頻率標準差演化')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('頻率標準差')
    
    # 頻率範圍演化
    axes[1, 0].plot(epochs, [h['freq_min'] for h in freq_history], label='最小值')
    axes[1, 0].plot(epochs, [h['freq_max'] for h in freq_history], label='最大值')
    axes[1, 0].set_title('頻率範圍演化')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('頻率值')
    axes[1, 0].legend()
    
    # 當前頻率分佈
    current_freqs = []
    for layer in model.modules():
        if isinstance(layer, TSAActivation):
            current_freqs.extend(layer.frequencies.detach().cpu().numpy())
    
    axes[1, 1].hist(current_freqs, bins=30, alpha=0.7)
    axes[1, 1].set_title('當前頻率分佈')
    axes[1, 1].set_xlabel('頻率值')
    axes[1, 1].set_ylabel('頻次')
    
    plt.tight_layout()
    plt.show()
```

---

## 版本資訊

- **當前版本**: 1.1.0
- **相容性**: PyTorch 1.8+
- **Python 版本**: 3.7+

## 授權

本 API 基於 TSA-PINN 論文實現，僅供學術研究使用。

---

## 更新日誌

### v1.1.0 (改進版)
- **修正**：澄清了 SlopeRecovery 的工作原理和數學公式
- **修正**：明確說明了單層 TSA 情況下的公式處理
- **改進**：詳細解釋了 `create_tsa_network` 中激活函數的放置位置
- **新增**：進階使用技巧和可視化分析方法
- **改進**：更準確的術語使用（頻率正規化 vs 斜率恢復）

### v1.0.0 (2024-12-19)
- 初始版本發布
- 實現核心 TSA 激活函數
- 添加斜率恢復機制
- 提供完整的工具函數和範例