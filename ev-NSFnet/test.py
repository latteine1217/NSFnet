# Copyright (c) 2023 scien42.tech, Se42 Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Zhicheng Wang, Hui Xiang
# Created: 08.03.2023
import torch
from train import setup_distributed, cleanup_distributed
from tools import *
import cavity_data as cavity
import pinn_solver as psolver
import csv
import os
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description='PINN Testing Script')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the run directory containing checkpoints (e.g., ~/NSFnet/ev-NSFnet/results/Re5000/6x80_Nf120k_lamB10_alpha0.05Stage_1)')
    return parser.parse_args()

def test_run(run_dir):
    # ===================================================================
    # 1. 使用預設/基礎參數初始化PINN模型
    #    這些參數將在載入第一個checkpoint時被正確覆寫
    # ===================================================================
    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=1000,      # 臨時值
        layers=6,     # 臨時值
        hidden_size=80, # 臨時值
        # 其他參數可以使用預設值，因為它們也會被checkpoint覆蓋
    )
    
    # 為load_checkpoint準備一個臨時的優化器
    optimizer = torch.optim.Adam(
        list(PINN.get_model_parameters(PINN.net)) + list(PINN.get_model_parameters(PINN.net_1)),
        lr=0.001
    )
    PINN.set_optimizers(optimizer)

    # 找到所有checkpoint檔案
    checkpoint_files = sorted([f for f in os.listdir(run_dir) if re.match(r'checkpoint_epoch_\d+\.pth', f)])

    if not checkpoint_files:
        print(f"No checkpoint files found in {run_dir}")
        return

    # ===================================================================
    # 2. 載入第一個Checkpoint來設定模型參數和物理參數
    # ===================================================================
    first_checkpoint_path = os.path.join(run_dir, checkpoint_files[0])
    print(f"Loading initial configuration from: {first_checkpoint_path}")
    PINN.load_checkpoint(first_checkpoint_path, optimizer)
    print(f"Configuration loaded. Re={PINN.Re}, Layers={PINN.layers}, HiddenSize={PINN.hidden_size}")

    # ===================================================================
    # 3. 根據恢復的參數載入評估數據
    # ===================================================================
    path = './data/'
    dataloader = cavity.DataLoader(path=path, N_f=PINN.N_f, N_b=1000, device=PINN.device)
    filename = f'./data/cavity_Re{PINN.Re}_256_Uniform.mat'
    print(f"Loading evaluation data: {filename}")
    x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(filename)

    # ===================================================================
    # 4. 遍歷所有Checkpoint進行評估
    # ===================================================================
    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(run_dir, checkpoint_file)
        print(f"\n--- Evaluating checkpoint: {checkpoint_path} ---")
        
        # 載入當前checkpoint的權重
        PINN.load_checkpoint(checkpoint_path, optimizer)
        
        match = re.search(r'epoch_(\d+)\.pth', checkpoint_file)
        current_epoch = int(match.group(1)) if match else 0

        # 執行評估
        PINN.evaluate(x_star, y_star, u_star, v_star, p_star)
        # PINN.test(x_star, y_star, u_star, v_star, p_star, current_epoch) # test函數會儲存mat檔，可根據需要啟用


if __name__ == "__main__":
    is_distributed = setup_distributed()
    if not is_distributed:
        # fallback 或設定成單 GPU 模式
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    
    args = parse_args()
    test_run(args.run_dir)

    if is_distributed:
        cleanup_distributed()