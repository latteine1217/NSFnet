import os
import torch
import pinn_solver as psolver


def dummy_data(device):
    x = torch.rand(100, 1, device=device, dtype=torch.float32)
    y = torch.rand(100, 1, device=device, dtype=torch.float32)
    xb = torch.rand(10, 1, device=device, dtype=torch.float32)
    yb = torch.rand(10, 1, device=device, dtype=torch.float32)
    ub = torch.zeros(10, device=device, dtype=torch.float32)
    vb = torch.zeros(10, device=device, dtype=torch.float32)
    return (xb, yb, ub, vb), (x.requires_grad_(True), y.requires_grad_(True))


def build_pinn():
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('LOCAL_RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    PINN = psolver.PysicsInformedNeuralNetwork(
        Re=5000,
        layers=2,
        layers_1=2,
        hidden_size=8,
        hidden_size_1=8,
        N_f=100,
        batch_size=100,
        alpha_evm=0.01,
        bc_weight=1.0,
        eq_weight=1.0,
        checkpoint_freq=10,
    )
    return PINN


def test_lbfgs_segment_runs():
    PINN = build_pinn()
    device = PINN.device
    bd, eq = dummy_data(device)
    PINN.set_boundary_data(bd)
    PINN.set_eq_training_data(eq)

    opt = torch.optim.Adam(
        list(PINN.get_model(PINN.net).parameters()) + list(PINN.get_model(PINN.net_1).parameters()), lr=1e-3
    )
    PINN.set_optimizers(opt)

    PINN.current_stage = "Stage 3"

    loss_before, _ = PINN.fwd_computing_loss_2d()
    l0 = float(loss_before.detach().cpu().item())

    best = PINN.train_with_lbfgs_segment(
        max_outer_steps=5,
        lbfgs_params={
            'max_iter': 3,
            'history_size': 5,
            'tolerance_grad': 1e-6,
            'tolerance_change': 1e-9,
            'line_search_fn': 'strong_wolfe'
        },
        log_interval=1
    )

    loss_after, _ = PINN.fwd_computing_loss_2d()
    l1 = float(loss_after.detach().cpu().item())

    print("=== 測試結果 ===")
    print(f"loss_before={l0:.3e}")
    print(f"loss_after ={l1:.3e}")
    print(f"best_lbfgs ={best:.3e}")

    assert torch.isfinite(loss_before).item()
    assert torch.isfinite(loss_after).item()
    # 不強制下降，但至少確保可執行且返回數值
    assert isinstance(best, float)


if __name__ == '__main__':
    test_lbfgs_segment_runs()
