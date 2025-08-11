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
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='PINN Testing Script')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the run directory containing checkpoints (e.g., ~/NSFnet/ev-NSFnet/results/Re5000/6x80_Nf120k_lamB10_alpha0.05Stage_1)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Unified output directory for test results. If not specified, uses timestamp-based directory.')
    return parser.parse_args()

def test_run(run_dir, output_dir=None):
    # 使用指定的輸出目錄或創建時間戳目錄
    if output_dir:
        test_results_dir = output_dir
        os.makedirs(test_results_dir, exist_ok=True)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        test_results_dir = f"results/test_results/{timestamp}"
        os.makedirs(test_results_dir, exist_ok=True)
    
    print(f"Test results will be saved to: {test_results_dir}")
    
    # 從run_dir提取stage信息
    stage_match = re.search(r'Stage_(\d+)', run_dir)
    stage_num = int(stage_match.group(1)) if stage_match else 1
    
    # 根據配置定義每個stage的epoch數（需要與train配置一致）
    epochs_per_stage = 300000  # 可以後續從配置文件讀取
    print(f"Detected Stage: {stage_num}, epochs per stage: {epochs_per_stage}")
    
    Re = 5000   # Reynolds number (This should ideally be read from checkpoint or config)
    N_neu = 80
    N_neu_1 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 200000
    alpha_evm = 0.03
    N_HLayer = 6
    N_HLayer_1 = 4

    # Initialize PINN model (weights will be loaded from checkpoint)
    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        hidden_size = N_neu,
        hidden_size_1 = N_neu_1,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
        # net_params and net_params_1 are not needed here as we load full checkpoint
    )
    
    # Dummy optimizer for loading checkpoint (state will be overwritten)
    optimizer = torch.optim.Adam(
        list(PINN.get_model_parameters(PINN.net)) + list(PINN.get_model_parameters(PINN.net_1)),
        lr=0.001, # Dummy learning rate
        weight_decay=0.0
    )
    PINN.set_optimizers(optimizer)

    path = './data/'
    dataloader = cavity.DataLoader(path=path, N_f=N_f, N_b=1000)

    filename = f'./data/cavity_Re{Re}_256_Uniform.mat'
    x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(filename)

    # Find all checkpoint files in the run_dir
    checkpoint_files = sorted([f for f in os.listdir(run_dir) if re.match(r'checkpoint_epoch_\d+\.pth', f)])

    if not checkpoint_files:
        print(f"No checkpoint files found in {run_dir}")
        return

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(run_dir, checkpoint_file)
        print(f"Evaluating checkpoint: {checkpoint_path}")
        
        # Load the checkpoint
        start_epoch = PINN.load_checkpoint(checkpoint_path, optimizer)
        
        # Extract epoch from filename for loop parameter
        match = re.search(r'epoch_(\d+)\.pth', checkpoint_file)
        current_epoch = int(match.group(1)) if match else 0
        
        # 計算全局epoch
        global_epoch = (stage_num - 1) * epochs_per_stage + current_epoch
        print(f"Stage {stage_num}, local epoch {current_epoch} → global epoch {global_epoch}")

        # Evaluating
        PINN.evaluate(x_star, y_star, u_star, v_star, p_star)
        PINN.test(x_star, y_star, u_star, v_star, p_star, global_epoch, test_results_dir)

if __name__ == "__main__":
    is_distributed = setup_distributed()
    if not is_distributed:
        # fallback 或設定成單 GPU 模式
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
    
    args = parse_args()
    test_run(args.run_dir, args.output_dir)

    if is_distributed:
        cleanup_distributed()