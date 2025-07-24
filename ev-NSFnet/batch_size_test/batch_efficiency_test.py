# batch_efficiency_test.py
# 支援雙GPU分布式訓練的批次處理效率測試腳本

import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
import numpy as np
import psutil
import matplotlib.pyplot as plt
import csv
from datetime import datetime

import cavity_data as cavity
import pinn_solver as psolver
from tools import *

class DistributedBatchEfficiencyTester:
    def __init__(self):
        self.results = {}
        self.setup_distributed_environment()
        
    def setup_distributed_environment(self):
        """初始化分布式訓練環境"""
        # 檢查是否在分布式環境中
        if 'WORLD_SIZE' not in os.environ:
            # 單GPU測試模式
            self.is_distributed = False
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            os.environ['RANK'] = '0'
            os.environ['LOCAL_RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
        else:
            # 分布式訓練模式
            self.is_distributed = True
            try:
                dist.init_process_group(backend='nccl')
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.local_rank = int(os.environ['LOCAL_RANK'])
                torch.cuda.set_device(self.local_rank)
            except Exception as e:
                print(f"分布式初始化失敗: {e}")
                self.is_distributed = False
                self.rank = 0
                self.local_rank = 0
                self.world_size = 1
        
        if self.rank == 0:
            print(f"🔧 分布式設置:")
            print(f"   模式: {'分布式' if self.is_distributed else '單GPU'}")
            print(f"   World Size: {self.world_size}")
            print(f"   GPU數量: {torch.cuda.device_count()}")
        
        # 清空GPU緩存
        torch.cuda.empty_cache()
        
    def create_test_pinn(self, Re=3000):
        """創建測試用的PINN模型（支援分布式）"""
        N_neu = 80
        N_neu_1 = 40
        lam_bcs = 10
        lam_equ = 1
        alpha_evm = 0.03
        N_HLayer = 6
        N_HLayer_1 = 4
        
        # PINN已內建DDP支援，無需額外包裝
        pinn = psolver.PysicsInformedNeuralNetwork(
            Re=Re,
            layers=N_HLayer,
            layers_1=N_HLayer_1,
            hidden_size=N_neu,
            hidden_size_1=N_neu_1,
            alpha_evm=alpha_evm,
            bc_weight=lam_bcs,
            eq_weight=lam_equ,
            N_f=120000)
        
        return pinn
        
    def prepare_data(self, N_f=120000, N_b=1000):
        """準備訓練數據"""
        path = './data/'
        dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=N_b)
        
        boundary_data = dataloader.loading_boundary_data()
        training_data = dataloader.loading_training_data()
        
        return boundary_data, training_data
        
    def create_distributed_dataloader(self, x_f, y_f, x_b, y_b, u_b, v_b, batch_size):
        """創建分布式批次數據加載器"""
        # 為方程點創建dataset
        eq_dataset = TensorDataset(
            torch.tensor(x_f, dtype=torch.float32),
            torch.tensor(y_f, dtype=torch.float32)
        )
        
        # 為邊界點創建dataset  
        bc_dataset = TensorDataset(
            torch.tensor(x_b, dtype=torch.float32),
            torch.tensor(y_b, dtype=torch.float32),
            torch.tensor(u_b, dtype=torch.float32),
            torch.tensor(v_b, dtype=torch.float32)
        )
        
        if self.is_distributed:
            # 分布式採樣器
            eq_sampler = DistributedSampler(eq_dataset, shuffle=True)
            bc_sampler = DistributedSampler(bc_dataset, shuffle=True)
            
            eq_loader = DataLoader(eq_dataset, batch_size=batch_size, 
                                 sampler=eq_sampler, num_workers=4, pin_memory=True)
            bc_loader = DataLoader(bc_dataset, batch_size=min(batch_size, len(bc_dataset)), 
                                 sampler=bc_sampler, num_workers=2, pin_memory=True)
        else:
            # 單GPU模式
            eq_loader = DataLoader(eq_dataset, batch_size=batch_size, shuffle=True)
            bc_loader = DataLoader(bc_dataset, batch_size=min(batch_size, len(bc_dataset)), shuffle=True)
        
        return eq_loader, bc_loader
        
    def monitor_all_gpus_memory(self):
        """監控所有GPU的記憶體使用"""
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=i)
                allocated = torch.cuda.memory_allocated(device=i) / 1024**3
                reserved = torch.cuda.memory_reserved(device=i) / 1024**3
                gpu_info.append((allocated, reserved))
            else:
                gpu_info.append((0, 0))
        return gpu_info
        
    def monitor_cpu_memory(self):
        """監控CPU記憶體使用"""
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024**3
        return cpu_memory
        
    def test_batch_size_distributed(self, batch_size, num_steps=1000):
        """測試特定batch size的分布式性能"""
        if self.rank == 0:
            print(f"\n{'='*50}")
            print(f"測試 Batch Size: {batch_size} ({'分布式' if self.is_distributed else '單GPU'})")
            print(f"{'='*50}")
        
        # 創建模型和數據
        pinn = self.create_test_pinn()
        boundary_data, training_data = self.prepare_data()
        
        x_b, y_b, u_b, v_b = boundary_data
        x_f, y_f = training_data
        
        # 設置數據到模型
        pinn.set_boundary_data(X=boundary_data)
        pinn.set_eq_training_data(X=training_data)
        
        # 創建分布式數據加載器
        eq_loader, bc_loader = self.create_distributed_dataloader(
            x_f, y_f, x_b, y_b, u_b, v_b, batch_size
        )
        
        # 創建迭代器（提升效率）
        eq_iter = iter(eq_loader)
        bc_iter = iter(bc_loader)
        
        # 性能監控變量
        step_times = []
        all_gpu_memory = []
        cpu_memory_usage = []
        losses = []
        
        # 同步所有進程
        if self.is_distributed:
            dist.barrier()
        
        # 開始測試
        start_time = time.time()
        
        if self.rank == 0:
            print(f"開始 {num_steps} 步分布式訓練測試...")
        
        for step in range(num_steps):
            step_start = time.time()
            
            # 獲取批次數據
            try:
                eq_batch = next(eq_iter)
                bc_batch = next(bc_iter)
            except StopIteration:
                # 重新開始新的epoch
                eq_iter = iter(eq_loader)
                bc_iter = iter(bc_loader)
                eq_batch = next(eq_iter)
                bc_batch = next(bc_iter)
            
            # 設置批次數據到模型
            x_f_batch = eq_batch[0].cuda(non_blocking=True)
            y_f_batch = eq_batch[1].cuda(non_blocking=True)
            x_b_batch = bc_batch[0].cuda(non_blocking=True)
            y_b_batch = bc_batch[1].cuda(non_blocking=True)
            u_b_batch = bc_batch[2].cuda(non_blocking=True)
            v_b_batch = bc_batch[3].cuda(non_blocking=True)
            
            # 更新模型數據
            pinn.x_f = x_f_batch.requires_grad_(True)
            pinn.y_f = y_f_batch.requires_grad_(True)
            pinn.x_b = x_b_batch
            pinn.y_b = y_b_batch
            pinn.u_b = u_b_batch
            pinn.v_b = v_b_batch
            
            # 前向傳播和反向傳播
            pinn.opt.zero_grad()
            loss, loss_components = pinn.fwd_computing_loss_2d()
            loss.backward()  # DDP自動進行梯度同步
            pinn.opt.step()
            
            # 只在必要時同步
            torch.cuda.synchronize()
            
            step_end = time.time()
            step_time = step_end - step_start
            
            # 記錄性能數據（只在rank 0記錄）
            if self.rank == 0:
                step_times.append(step_time)
                losses.append(loss.item())
                
                # 記錄所有GPU記憶體使用
                gpu_memory = self.monitor_all_gpus_memory()
                all_gpu_memory.append(gpu_memory)
                
                cpu_mem = self.monitor_cpu_memory()
                cpu_memory_usage.append(cpu_mem)
                
                # 定期輸出進度
                if (step + 1) % 100 == 0:
                    avg_time = np.mean(step_times[-100:])
                    current_loss = loss.item()
                    total_gpu_memory = sum([gpu[0] for gpu in gpu_memory])
                    print(f"Step {step+1}/{num_steps} | "
                          f"Time: {avg_time:.4f}s | "
                          f"Loss: {current_loss:.6e} | "
                          f"Total GPU: {total_gpu_memory:.2f}GB")
        
        total_time = time.time() - start_time
        
        # 只在rank 0計算和返回結果
        if self.rank == 0:
            # 計算GPU記憶體統計
            max_gpu_memory_per_gpu = []
            avg_gpu_memory_per_gpu = []
            for gpu_idx in range(len(all_gpu_memory[0])):
                gpu_usage = [frame[gpu_idx][0] for frame in all_gpu_memory]
                max_gpu_memory_per_gpu.append(max(gpu_usage))
                avg_gpu_memory_per_gpu.append(np.mean(gpu_usage))
            
            result = {
                'batch_size': batch_size,
                'distributed': self.is_distributed,
                'world_size': self.world_size,
                'total_time': total_time,
                'avg_step_time': np.mean(step_times),
                'std_step_time': np.std(step_times),
                'throughput': num_steps / total_time,
                'max_gpu_memory_per_gpu': max_gpu_memory_per_gpu,
                'avg_gpu_memory_per_gpu': avg_gpu_memory_per_gpu,
                'total_max_gpu_memory': sum(max_gpu_memory_per_gpu),
                'total_avg_gpu_memory': sum(avg_gpu_memory_per_gpu),
                'avg_cpu_memory': np.mean(cpu_memory_usage),
                'final_loss': losses[-1],
                'avg_loss': np.mean(losses[-100:]),
                'step_times': step_times,
                'losses': losses,
                'gpu_memory_history': all_gpu_memory
            }
            
            self.results[f"{batch_size}_{'dist' if self.is_distributed else 'single'}"] = result
        else:
            result = None
        
        # 清理
        del pinn
        torch.cuda.empty_cache()
        
        return result
        
    def run_distributed_efficiency_tests(self):
        """運行分布式效率測試"""
        if self.rank == 0:
            print("🚀 開始分布式批次處理效率測試")
            print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"GPU配置: {torch.cuda.device_count()} × GPU")
        
        # 測試不同的batch size
        batch_sizes = [
            120000,  # 原始全批次
            24000,   # 雙GPU情況下的1/5
            12000,   # 雙GPU情況下的1/10
            8000,    # 積極設置
            4000,    # 保守設置
            2000     # 更小批次
        ]
        
        for batch_size in batch_sizes:
            try:
                result = self.test_batch_size_distributed(batch_size, num_steps=1000)
                if self.rank == 0 and result:
                    self.print_distributed_batch_summary(result)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.rank == 0:
                        print(f"❌ Batch size {batch_size} 記憶體不足，跳過測試")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            except Exception as e:
                if self.rank == 0:
                    print(f"❌ Batch size {batch_size} 測試失敗: {e}")
                continue
                
    def print_distributed_batch_summary(self, result):
        """打印分布式batch size測試結果"""
        print(f"\n📊 Batch Size {result['batch_size']} 分布式測試結果:")
        print(f"   模式: {'分布式' if result['distributed'] else '單GPU'} ({result['world_size']} GPU)")
        print(f"   總時間: {result['total_time']:.2f}s")
        print(f"   平均每步時間: {result['avg_step_time']:.4f}s")
        print(f"   吞吐量: {result['throughput']:.2f} steps/s")
        print(f"   總GPU記憶體: {result['total_max_gpu_memory']:.2f}GB")
        
        for i, gpu_mem in enumerate(result['max_gpu_memory_per_gpu']):
            print(f"   GPU {i} 最大記憶體: {gpu_mem:.2f}GB")
        
        print(f"   最終損失: {result['final_loss']:.6e}")
        
    def generate_performance_report(self):
        """生成性能分析報告"""
        if not self.results:
            print("❌ 沒有測試結果可供分析")
            return
            
        print(f"\n{'='*60}")
        print("📈 分布式批次處理效率分析報告")
        print(f"{'='*60}")
        
        # 創建比較表格
        print(f"{'Batch Size':<12} {'Mode':<12} {'Time(s)':<10} {'Steps/s':<10} {'Total GPU(GB)':<15} {'Speedup':<10}")
        print("-" * 80)
        
        baseline_time = None
        for key in sorted(self.results.keys()):
            result = self.results[key]
            
            if baseline_time is None:
                baseline_time = result['avg_step_time']
                speedup = 1.0
            else:
                speedup = baseline_time / result['avg_step_time']
                
            mode = "分布式" if result['distributed'] else "單GPU"
            print(f"{result['batch_size']:<12} {mode:<12} {result['avg_step_time']:<10.4f} "
                  f"{result['throughput']:<10.2f} {result['total_max_gpu_memory']:<15.2f} "
                  f"{speedup:<10.2f}x")
        
        # 找出最佳batch size
        best_key = min(self.results.keys(), 
                      key=lambda x: self.results[x]['avg_step_time'])
        best_result = self.results[best_key]
        
        print(f"\n🏆 最佳性能配置:")
        print(f"   Batch Size: {best_result['batch_size']}")
        print(f"   模式: {'分布式' if best_result['distributed'] else '單GPU'}")
        print(f"   加速比: {baseline_time/best_result['avg_step_time']:.2f}x")
        print(f"   總GPU記憶體使用: {best_result['total_max_gpu_memory']:.2f}GB")
        
        # 保存結果到CSV
        self.save_results_csv()
        
    def save_results_csv(self):
        """保存結果到CSV文件"""
        filename = f"batch_efficiency_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['batch_size', 'distributed', 'world_size', 'total_time', 'avg_step_time', 
                         'throughput', 'total_max_gpu_memory', 'final_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in self.results.values():
                writer.writerow({k: result[k] for k in fieldnames})
                
        print(f"\n💾 結果已保存到: {filename}")
        
    def cleanup_distributed(self):
        """清理分布式環境"""
        if self.is_distributed:
            dist.destroy_process_group()

def main():
    """主函數"""
    tester = DistributedBatchEfficiencyTester()
    
    try:
        # 運行分布式效率測試
        tester.run_distributed_efficiency_tests()
        
        # 只在rank 0生成報告
        if tester.rank == 0:
            tester.generate_performance_report()
            print(f"\n✅ 分布式測試完成！")
            print("建議根據測試結果選擇最佳的batch size進行分布式訓練。")
        
    except KeyboardInterrupt:
        if tester.rank == 0:
            print("\n⚠️  測試被用戶中斷")
    except Exception as e:
        if tester.rank == 0:
            print(f"\n❌ 測試過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    finally:
        tester.cleanup_distributed()

if __name__ == "__main__":
    main()