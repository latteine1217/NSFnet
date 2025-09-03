#!/usr/bin/env python3
"""
SDF權重示意圖生成腳本

此腳本用於視覺化Physics-Informed Neural Networks (PINNs) 專案中使用的
Signed Distance Function (SDF) 權重分佈。

基於 pinn_solver.py 中的實現：
- 距離函數：d = min(x+1, 1-x, y+1, 1-y) 在 [-1,1]² 域上
- 權重函數：w(d) = w_min + (1 - w_min) * exp(-d / tau)
- 預設參數：w_min = 0.2, tau = 0.2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

def compute_sdf_distance(x, y):
    """計算到邊界的最短距離 (SDF)
    
    Args:
        x, y: 網格座標，範圍 [-1, 1]
        
    Returns:
        距離場 d = min(x+1, 1-x, y+1, 1-y)
    """
    d_x = np.minimum(x + 1.0, 1.0 - x)
    d_y = np.minimum(y + 1.0, 1.0 - y)
    return np.minimum(d_x, d_y)

def compute_weight(d, w_min=0.2, tau=0.2):
    """計算PDE殘差權重
    
    Args:
        d: 距離場
        w_min: 權重下限
        tau: 指數衰減參數
        
    Returns:
        權重 w(d) = w_min + (1 - w_min) * exp(-d / tau)
    """
    return w_min + (1.0 - w_min) * np.exp(-d / np.maximum(tau, 1e-6))

def create_visualization():
    """建立SDF權重視覺化圖表"""
    
    # 設定網格
    resolution = 200
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # 計算距離場和權重場
    distance_field = compute_sdf_distance(X, Y)
    weight_field = compute_weight(distance_field)
    
    # 建立圖表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PINN中SDF權重分佈視覺化', fontsize=16, fontweight='bold')
    
    # === 子圖1: 距離場 ===
    ax1 = axes[0, 0]
    im1 = ax1.contourf(X, Y, distance_field, levels=20, cmap='viridis')
    ax1.set_title('Distance Field: d = min(x+1, 1-x, y+1, 1-y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 添加邊界框
    boundary = patches.Rectangle((-1, -1), 2, 2, linewidth=3, 
                               edgecolor='red', facecolor='none', linestyle='--')
    ax1.add_patch(boundary)
    
    plt.colorbar(im1, ax=ax1, label='Distance to boundary')
    
    # === 子圖2: 權重場 ===
    ax2 = axes[0, 1]
    im2 = ax2.contourf(X, Y, weight_field, levels=20, cmap='plasma')
    ax2.set_title('Weight Field: w(d) = 0.2 + 0.8×exp(-d/0.2)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 添加邊界框
    boundary2 = patches.Rectangle((-1, -1), 2, 2, linewidth=3, 
                                edgecolor='red', facecolor='none', linestyle='--')
    ax2.add_patch(boundary2)
    
    plt.colorbar(im2, ax=ax2, label='PDE residual weight')
    
    # === 子圖3: 權重函數曲線 ===
    ax3 = axes[1, 0]
    d_range = np.linspace(0, 1.0, 1000)
    
    # 不同tau值的比較
    tau_values = [0.05, 0.1, 0.2, 0.3]
    colors = ['red', 'blue', 'green', 'orange']
    
    for tau, color in zip(tau_values, colors):
        w_curve = compute_weight(d_range, w_min=0.2, tau=tau)
        ax3.plot(d_range, w_curve, color=color, linewidth=2, 
                label=f'τ = {tau}')
    
    ax3.set_xlabel('Distance to boundary (d)')
    ax3.set_ylabel('Weight w(d)')
    ax3.set_title('Weight Function: w(d) = 0.2 + 0.8×exp(-d/τ)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0.15, 1.05)
    
    # === 子圖4: 權重等高線圖 ===
    ax4 = axes[1, 1]
    contour = ax4.contour(X, Y, weight_field, levels=15, colors='black', linewidths=0.5, alpha=0.6)
    ax4.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    im4 = ax4.contourf(X, Y, weight_field, levels=20, cmap='plasma', alpha=0.8)
    ax4.set_title('Weight Contours with Values')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # 添加邊界框
    boundary4 = patches.Rectangle((-1, -1), 2, 2, linewidth=3, 
                                edgecolor='red', facecolor='none', linestyle='--')
    ax4.add_patch(boundary4)
    
    plt.colorbar(im4, ax=ax4, label='Weight w(d)')
    
    # 調整佈局
    plt.tight_layout()
    
    # 儲存圖片
    plt.savefig('sdf_weight_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig('sdf_weight_visualization.pdf', bbox_inches='tight')
    
    print("=== SDF權重示意圖已生成 ===")
    print("檔案已儲存:")
    print("  - sdf_weight_visualization.png (高解析度PNG)")
    print("  - sdf_weight_visualization.pdf (向量圖)")
    print("\n圖表說明:")
    print("  左上: 距離場分佈 - 顯示到邊界的最短距離")
    print("  右上: 權重場分佈 - PDE殘差的加權係數")
    print("  左下: 權重函數曲線 - 不同tau參數的比較")
    print("  右下: 權重等高線圖 - 帶數值標籤的等高線")
    print("="*50)
    
    plt.show()

def create_cross_section_analysis():
    """建立橫截面分析圖"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SDF權重橫截面分析', fontsize=16, fontweight='bold')
    
    # === 橫截面 y=0 ===
    ax1 = axes[0, 0]
    x_line = np.linspace(-1, 1, 1000)
    y_line = np.zeros_like(x_line)
    d_line = compute_sdf_distance(x_line, y_line)
    w_line = compute_weight(d_line)
    
    ax1.plot(x_line, d_line, 'b-', linewidth=2, label='Distance d(x,0)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_line, w_line, 'r-', linewidth=2, label='Weight w(x,0)')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('Distance', color='b')
    ax1_twin.set_ylabel('Weight', color='r')
    ax1.set_title('Horizontal Cross-section (y=0)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # === 對角線橫截面 y=x ===
    ax2 = axes[0, 1]
    x_diag = np.linspace(-1, 1, 1000)
    y_diag = x_diag
    d_diag = compute_sdf_distance(x_diag, y_diag)
    w_diag = compute_weight(d_diag)
    
    ax2.plot(x_diag, d_diag, 'b-', linewidth=2, label='Distance d(x,x)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_diag, w_diag, 'r-', linewidth=2, label='Weight w(x,x)')
    
    ax2.set_xlabel('x (= y)')
    ax2.set_ylabel('Distance', color='b')
    ax2_twin.set_ylabel('Weight', color='r')
    ax2.set_title('Diagonal Cross-section (y=x)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    # === 不同w_min值的比較 ===
    ax3 = axes[1, 0]
    d_test = np.linspace(0, 0.5, 1000)
    w_min_values = [0.1, 0.2, 0.3, 0.5]
    colors = ['red', 'blue', 'green', 'orange']
    
    for w_min, color in zip(w_min_values, colors):
        w_test = compute_weight(d_test, w_min=w_min, tau=0.1)
        ax3.plot(d_test, w_test, color=color, linewidth=2, 
                label=f'w_min = {w_min}')
    
    ax3.set_xlabel('Distance to boundary (d)')
    ax3.set_ylabel('Weight w(d)')
    ax3.set_title('Effect of w_min parameter (τ=0.2)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, 0.5)
    
    # === 權重統計分析 ===
    ax4 = axes[1, 1]
    
    # 計算權重分佈統計
    resolution = 100
    x_stats = np.linspace(-1, 1, resolution)
    y_stats = np.linspace(-1, 1, resolution)
    X_stats, Y_stats = np.meshgrid(x_stats, y_stats)
    distance_stats = compute_sdf_distance(X_stats, Y_stats)
    weight_stats = compute_weight(distance_stats)
    
    # 計算權重直方圖
    weights_flat = weight_stats.flatten()
    ax4.hist(weights_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(weights_flat.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'Mean = {weights_flat.mean():.3f}')
    ax4.axvline(np.median(weights_flat), color='green', linestyle='--', linewidth=2,
               label=f'Median = {np.median(weights_flat):.3f}')
    
    ax4.set_xlabel('Weight value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Weight Distribution Histogram')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('sdf_weight_cross_sections.png', dpi=300, bbox_inches='tight')
    plt.savefig('sdf_weight_cross_sections.pdf', bbox_inches='tight')
    
    print("\n=== 橫截面分析圖已生成 ===")
    print("檔案已儲存:")
    print("  - sdf_weight_cross_sections.png")
    print("  - sdf_weight_cross_sections.pdf")
    
    plt.show()

def parameter_analysis():
    """參數敏感性分析"""
    
    print("\n=== 參數敏感性分析 ===")
    
    # 測試點：中心、邊界附近、角落
    test_points = [
        (0.0, 0.0, "中心點"),
        (0.5, 0.5, "內部點"),
        (0.9, 0.9, "近邊界點"),
        (0.05, 0.05, "極近邊界點")
    ]
    
    for x, y, desc in test_points:
        d = compute_sdf_distance(np.array([[x]]), np.array([[y]]))[0, 0]
        w = compute_weight(d)
        print(f"{desc:>10} ({x:4.1f}, {y:4.1f}): d={d:.4f}, w={w:.4f}")
    
    print("\n不同tau值的影響 (d=0.1):")
    tau_test = [0.05, 0.1, 0.2, 0.5]
    d_test = 0.1
    
    for tau in tau_test:
        w = compute_weight(d_test, tau=tau)
        print(f"  τ={tau:4.2f}: w({d_test})={w:.4f}")
    
    print(f"\n權重增強效果分析:")
    print(f"  邊界點權重 / 中心點權重 比值: {compute_weight(0.0) / compute_weight(1.0):.2f}")
    print(f"  這表示邊界附近的PDE殘差獲得 {compute_weight(0.0) / compute_weight(1.0):.1f} 倍的重視")

if __name__ == "__main__":
    # 設定matplotlib中文字體
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 建立主要視覺化圖表
    create_visualization()
    
    # 建立橫截面分析圖
    create_cross_section_analysis()
    
    # 執行參數分析
    parameter_analysis()
