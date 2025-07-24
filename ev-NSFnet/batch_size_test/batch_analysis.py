import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_batch_efficiency():
    """分析batch size效率數據"""
    # 讀取CSV數據
    df = pd.read_csv('batch_efficiency_results_20250723_052239.csv')
    
    print("=== Batch Size 效率分析 ===")
    
    # 顯示基本數據
    print("\n--- 原始數據 ---")
    print(df.to_string(index=False))
    
    # 計算效率指標
    df['time_per_sample'] = df['total_time'] / df['batch_size']
    df['memory_efficiency'] = df['throughput'] / df['total_max_gpu_memory']
    df['training_efficiency'] = 1 / (df['avg_step_time'] * df['final_loss'])
    
    print("\n--- 效率指標分析 ---")
    print(f"{'Batch Size':<12} {'Throughput':<12} {'Memory Eff':<12} {'Train Eff':<12} {'Time/Sample':<12}")
    print("-" * 60)
    
    for _, row in df.iterrows():
        print(f"{row['batch_size']:<12} {row['throughput']:<12.3f} "
              f"{row['memory_efficiency']:<12.1f} {row['training_efficiency']:<12.1f} "
              f"{row['time_per_sample']:<12.6f}")
    
    # 找出最佳配置
    best_throughput = df.loc[df['throughput'].idxmax()]
    best_memory_eff = df.loc[df['memory_efficiency'].idxmax()]
    best_training_eff = df.loc[df['training_efficiency'].idxmax()]
    
    print("\n--- 最佳配置分析 ---")
    print(f"最高吞吐量: Batch Size {int(best_throughput['batch_size'])} (throughput: {best_throughput['throughput']:.3f})")
    print(f"最佳記憶體效率: Batch Size {int(best_memory_eff['batch_size'])} (memory_eff: {best_memory_eff['memory_efficiency']:.1f})")
    print(f"最佳訓練效率: Batch Size {int(best_training_eff['batch_size'])} (train_eff: {best_training_eff['training_efficiency']:.1f})")
    
    # 綜合評分 (歸一化後加權平均)
    weights = {'throughput': 0.4, 'memory_efficiency': 0.3, 'training_efficiency': 0.3}
    
    # 歸一化指標
    df['norm_throughput'] = (df['throughput'] - df['throughput'].min()) / (df['throughput'].max() - df['throughput'].min())
    df['norm_memory_eff'] = (df['memory_efficiency'] - df['memory_efficiency'].min()) / (df['memory_efficiency'].max() - df['memory_efficiency'].min())
    df['norm_train_eff'] = (df['training_efficiency'] - df['training_efficiency'].min()) / (df['training_efficiency'].max() - df['training_efficiency'].min())
    
    # 計算綜合評分
    df['composite_score'] = (weights['throughput'] * df['norm_throughput'] + 
                           weights['memory_efficiency'] * df['norm_memory_eff'] + 
                           weights['training_efficiency'] * df['norm_train_eff'])
    
    best_overall = df.loc[df['composite_score'].idxmax()]
    
    print(f"\n--- 綜合最佳化建議 ---")
    print(f"最佳Batch Size: {int(best_overall['batch_size'])}")
    print(f"綜合評分: {best_overall['composite_score']:.3f}")
    print(f"預期吞吐量: {best_overall['throughput']:.3f} 樣本/秒")
    print(f"記憶體使用: {best_overall['total_max_gpu_memory']:.3f} GB")
    print(f"平均步驟時間: {best_overall['avg_step_time']:.3f} 秒")
    
    # 效率分析
    print(f"\n--- 效率權衡分析 ---")
    small_batch = df[df['batch_size'] == 2000].iloc[0]
    large_batch = df[df['batch_size'] == 120000].iloc[0]
    
    print(f"小批次 (2000) vs 大批次 (120000):")
    print(f"  吞吐量提升: {(small_batch['throughput']/large_batch['throughput']-1)*100:.1f}%")
    print(f"  時間節省: {(large_batch['total_time']/small_batch['total_time']-1)*100:.1f}%")
    print(f"  記憶體節省: {(large_batch['total_max_gpu_memory']/small_batch['total_max_gpu_memory']-1)*100:.1f}%")
    
    return df

if __name__ == "__main__":
    analyze_batch_efficiency()