# Restored baseline training script (based on 41c5563) + modernized output
# 只保留：簡化配置 / logger / 階段列印；不引入新功能 (supervision, scheduler, AdamW 等)
import os
import argparse
import torch
import torch.distributed as dist
import time
import cavity_data as cavity
import numpy as np
import pinn_solver as psolver
from config import ConfigManager
from logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Restored NSFnet Training')
    parser.add_argument('--config', type=str, default='configs/production.yaml', help='YAML config file path')
    parser.add_argument('--dry-run', action='store_true', help='Print config & stages then exit')
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training (保持舊版語意, 增加更友好輸出)."""
    if 'WORLD_SIZE' not in os.environ or int(os.environ.get('WORLD_SIZE', '1')) <= 1:
        print('💻 單GPU模式 (未檢測到分布式環境變數)')
        return False
    required_env = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    for env_var in required_env:
        if env_var not in os.environ:
            print(f"缺少環境變數: {env_var} -> 回退單GPU")
            return False
    try:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(f"📡 分布式啟動: world={world_size}, rank={rank}, local_rank={local_rank}")
        return True
    except Exception as e:
        print(f"分布式初始化失敗: {e}; 回退單GPU")
        return False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def print_stage_table(stages, logger):
    logger.info("=== Training Stages ===")
    for i, st in enumerate(stages, 1):
        logger.info(f"{i:02d} | {st.name:<8} | alpha={st.alpha:.3g} | epochs={st.epochs:,} | lr={st.lr:.2e}")
    logger.info('='*60)


def build_pinn(cfg):
    # 將 supervision 權重傳遞到 solver（僅在啟用 supervision 時有效）
    sup_enabled = getattr(cfg, 'supervision', None) is not None and cfg.supervision.enabled
    supervised_data_weight = cfg.supervision.weight if sup_enabled else 1.0
    enable_dns_enhancement = bool(sup_enabled)

    return psolver.PysicsInformedNeuralNetwork(
        Re=cfg.physics.Re,
        layers=cfg.network.layers,
        layers_1=cfg.network.layers_1,
        hidden_size=cfg.network.hidden_size,
        hidden_size_1=cfg.network.hidden_size_1,
        N_f=cfg.training.N_f,
        alpha_evm=cfg.physics.alpha_evm,
        bc_weight=cfg.physics.bc_weight,
        eq_weight=cfg.physics.eq_weight,
        checkpoint_path='./checkpoint/',
        # coordinates normalization
        normalize_coordinates=cfg.training.normalize_coordinates,
        # supervision / DNS 增強
        enable_dns_enhancement=enable_dns_enhancement,
        dns_points_count=(cfg.supervision.data_points if sup_enabled else 0),
        supervised_data_weight=supervised_data_weight,
        # PDE 距離權重（鏡像 ev-NSFnet copy）
        pde_distance_weighting=cfg.training.pde_distance_weighting,
        pde_distance_w_min=cfg.training.pde_distance_w_min,
        pde_distance_tau=cfg.training.pde_distance_tau,
    )


def main():
    args = parse_args()

    # 讀取配置 (若檔案不存在使用預設)
    if not os.path.exists(args.config):
        print(f"⚠️ 找不到配置檔 {args.config}，使用內建預設值")
        config_manager = ConfigManager()
    else:
        config_manager = ConfigManager.from_file(args.config)
    cfg = config_manager.config

    is_distributed = setup_distributed()
    if not is_distributed:
        # 設置單 GPU 環境變數以與 solver 對齊
        os.environ['RANK'] = '0'; os.environ['LOCAL_RANK'] = '0'; os.environ['WORLD_SIZE'] = '1'

    rank = int(os.environ.get('RANK', 0))
    logger = get_logger(cfg.experiment_name, rank=rank)

    if rank == 0:
        logger.header('Experiment Configuration')
        config_manager.print_config()

    stages = cfg.training.training_stages
    if rank == 0:
        print_stage_table(stages, logger)
    if args.dry_run:
        if rank == 0:
            logger.info('Dry-run 結束 (未進行訓練)')
        return

    # 構建 PINN
    if rank == 0:
        logger.header('Construct PINN')
    PINN = build_pinn(cfg)
    # logging controls
    PINN.log_interval = cfg.training.log_interval
    PINN.progress_bar_width = 30

    # TensorBoard (only rank 0)
    tb_writer = None
    if rank == 0 and getattr(cfg.training, 'enable_tensorboard', False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            run_name = f"{cfg.experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            log_dir = os.path.join(cfg.training.tb_log_dir, run_name)
            os.makedirs(log_dir, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=log_dir)
            PINN.tb_writer = tb_writer
            logger.info(f"TensorBoard 啟用: {log_dir}")
        except Exception as e:
            logger.warning(f"TensorBoard 初始化失敗: {e}")

    # 數據載入 (保持舊版語意: datasets/)
    path = './datasets/'
    dataloader = cavity.DataLoader(
        path=path,
        N_f=cfg.training.N_f,
        N_b=1000,
        sort_by_boundary_distance=cfg.training.sort_by_boundary_distance,
    )

    # 邊界 / 方程點
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    eval_file = f'./data/cavity_Re{cfg.physics.Re}_256_Uniform.mat'
    x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(eval_file)

    # 監督數據（DNS）載入與設置
    if cfg.supervision.enabled and cfg.supervision.data_points > 0:
        if rank == 0:
            logger.info(f"載入監督數據: {cfg.supervision.data_points} 點，來源: {cfg.supervision.data_path}")
        x_sup, y_sup, u_sup, v_sup, p_sup = dataloader.loading_supervision_data(
            cfg.supervision.data_path, cfg.supervision.data_points, cfg.supervision.random_seed
        )
        if x_sup.shape[0] > 0:
            # 組裝為 set_dns_data 所需格式
            dns_points = np.hstack([x_sup, y_sup])
            dns_values = np.hstack([u_sup, v_sup, p_sup])
            # 使用預設每點權重=1，由 supervised_data_weight 控制整體權重
            PINN.set_dns_data(dns_points=dns_points, dns_values=dns_values, custom_weights=None)
            if rank == 0:
                logger.info("監督數據設置完成（DNS loss 已啟用）")

    total_epochs = sum(st.epochs for st in stages)
    if rank == 0:
        logger.info(f'🚀 開始訓練：總 epochs={total_epochs:,} (stages={len(stages)})')

    try:
        for st in stages:
            if rank == 0:
                logger.stage(st.name, st.alpha, st.epochs, st.lr)
            PINN.current_stage = st.name
            PINN.set_alpha_evm(st.alpha)
            PINN.train(num_epoch=st.epochs, lr=st.lr)
            if rank == 0:
                PINN.evaluate(x_star, y_star, u_star, v_star, p_star)
        if rank == 0:
            logger.header('Training Completed')
    except Exception as e:
        if rank == 0:
            logger.error(f'訓練失敗: {e}')
        raise
    finally:
        cleanup_distributed()
        if rank == 0:
            if 'tb_writer' in locals() and tb_writer is not None:
                try:
                    tb_writer.close()
                    logger.info('TensorBoard 已關閉')
                except Exception:
                    pass
            logger.close()


if __name__ == '__main__':
    main()
