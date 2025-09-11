# Restored baseline training script (based on 41c5563) + modernized output
# 只保留：簡化配置 / logger / 階段列印；不引入新功能 (supervision, scheduler, AdamW 等)
import os
import argparse
import torch
import torch.distributed as dist
import time
import cavity_data as cavity
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
        checkpoint_path='./checkpoint/'
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

    # 數據載入 (保持舊版語意: datasets/)
    path = './datasets/'
    dataloader = cavity.DataLoader(path=path, N_f=cfg.training.N_f, N_b=1000)

    # 邊界 / 方程點
    boundary_data = dataloader.loading_boundary_data()
    PINN.set_boundary_data(X=boundary_data)
    training_data = dataloader.loading_training_data()
    PINN.set_eq_training_data(X=training_data)

    eval_file = f'./NSFnet/ev-NSFnet/data/cavity_Re{cfg.physics.Re}_256_Uniform.mat'
    x_star, y_star, u_star, v_star, p_star = dataloader.loading_evaluate_data(eval_file)

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
            logger.close()


if __name__ == '__main__':
    main()
