import os
import re
import argparse
import datetime
import torch

# 本專案模組
import pinn_solver as psolver
import cavity_data as cavity
from train import setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(description='Single-checkpoint prediction (no SLURM)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a unified checkpoint file, e.g., .../checkpoint_epoch_300000.pth')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save prediction results (MAT). If not set, auto timestamp dir is used.')
    return parser.parse_args()


def main():
    # 初始化分布式（若不可用則退化為單 GPU/CPU）
    is_distributed = setup_distributed()
    if not is_distributed:
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'

    args = parse_args()
    ckpt_path = os.path.expanduser(args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 先讀取 checkpoint 中的中繼資訊（例如 Re、alpha_evm、epoch）
    meta = torch.load(ckpt_path, map_location='cpu')
    Re = int(meta.get('Re', 5000))
    alpha_evm = float(meta.get('alpha_evm', 0.03))
    epoch_from_ckpt = int(meta.get('epoch', 0))

    # 準備輸出目錄
    if args.output_dir:
        save_dir = os.path.expanduser(args.output_dir)
        os.makedirs(save_dir, exist_ok=True)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('results', 'single_predict', ts)
        os.makedirs(save_dir, exist_ok=True)

    print('=== Prediction Setup ===')
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output Dir: {save_dir}")
    print(f"Re: {Re} | alpha_evm: {alpha_evm}")

    # 網路架構需與訓練時一致（本專案預設 6x80 / 4x40）
    N_HLayer = 6
    N_HLayer_1 = 4
    N_neu = 80
    N_neu_1 = 40
    lam_bcs = 10
    lam_equ = 1
    N_f = 200000

    # 初始化 PINN（權重稍後由 checkpoint 載入）
    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=Re,
        layers=N_HLayer,
        layers_1=N_HLayer_1,
        hidden_size=N_neu,
        hidden_size_1=N_neu_1,
        alpha_evm=alpha_evm,
        bc_weight=lam_bcs,
        eq_weight=lam_equ,
    )

    # 建立 dummy optimizer 以便載入 optimizer 狀態（若不需要可忽略）
    optimizer = torch.optim.Adam(
        list(PINN.get_model_parameters(PINN.net)) + list(PINN.get_model_parameters(PINN.net_1)),
        lr=1e-3,
        weight_decay=0.0,
    )
    PINN.set_optimizers(optimizer)

    # 載入驗證資料（與 test.py 相同路徑規則）
    data_dir = './data'
    mat_file = os.path.join(data_dir, f'cavity_Re{Re}_256_Uniform.mat')
    dataloader = cavity.DataLoader(path=data_dir, N_f=N_f, N_b=1000)
    x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(mat_file)

    # 載入 checkpoint（含主網與 EVM 權重）
    start_epoch = PINN.load_checkpoint(ckpt_path, optimizer)
    # 以 checkpoint 中的 epoch 作為輸出檔名標記事用
    loop_epoch = epoch_from_ckpt
    print(f"Loaded epoch from checkpoint: {loop_epoch}")

    # 進行評估與輸出
    PINN.evaluate(x_star, y_star, u_star, v_star, p_star)
    PINN.test(x_star, y_star, u_star, v_star, p_star, loop=loop_epoch, custom_save_dir=save_dir)

    if is_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()

