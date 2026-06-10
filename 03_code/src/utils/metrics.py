"""Error metrics and table printing shared by the examples."""

import torch
import numpy as np


def compute_errors(pred, exact, return_arrays=False):
    """RMSE/MAE/max/relative errors and R^2 between pred and exact."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(exact, torch.Tensor):
        exact = exact.detach().cpu().numpy()

    pred = np.asarray(pred).flatten()
    exact = np.asarray(exact).flatten()

    abs_error = np.abs(pred - exact)

    # floor the denominator so points where the response crosses zero
    # (sinusoidal loading) do not blow up the relative error
    max_abs_exact = float(np.max(np.abs(exact))) if exact.size > 0 else 0.0
    rel_floor = max(1e-12, 1e-6 * max_abs_exact)
    rel_denom = np.maximum(np.abs(exact), rel_floor)
    rel_error = abs_error / rel_denom

    mse = np.mean(abs_error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_error)
    max_error = np.max(abs_error)
    mean_rel_error = np.mean(rel_error)
    max_rel_error = np.max(rel_error)

    nrmse = rmse / (max_abs_exact + 1e-12)
    nmae = mae / (max_abs_exact + 1e-12)

    ss_res = np.sum((exact - pred) ** 2)
    ss_tot = np.sum((exact - exact.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    result = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'mean_rel_error': mean_rel_error,
        'max_rel_error': max_rel_error,
        'nrmse': nrmse,
        'nmae': nmae,
        'rel_floor': rel_floor,
        'max_abs_exact': max_abs_exact,
        'r2': r2,
    }

    if return_arrays:
        result['abs_error_array'] = abs_error
        result['rel_error_array'] = rel_error

    return result


def compute_l2_error(pred, exact):
    """||pred - exact||_2 / ||exact||_2."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(exact, torch.Tensor):
        exact = exact.detach().cpu().numpy()

    pred = np.asarray(pred).flatten()
    exact = np.asarray(exact).flatten()

    return np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-10)


def compute_linf_error(pred, exact):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(exact, torch.Tensor):
        exact = exact.detach().cpu().numpy()

    return np.max(np.abs(pred - exact))


def print_error_table(errors, title="Errors"):
    print(f"\n  {title}")
    print(f"  {'-'*45}")
    print(f"  {'MSE':<22} {errors['mse']:.6e}")
    print(f"  {'RMSE':<22} {errors['rmse']:.6e}")
    if 'nrmse' in errors:
        print(f"  {'NRMSE (RMSE/amp)':<22} {errors['nrmse']:.6e}")
    print(f"  {'MAE':<22} {errors['mae']:.6e}")
    if 'nmae' in errors:
        print(f"  {'NMAE (MAE/amp)':<22} {errors['nmae']:.6e}")
    print(f"  {'max abs error':<22} {errors['max_error']:.6e}")
    print(f"  {'mean rel error':<22} {errors['mean_rel_error']*100:.4f}%")
    print(f"  {'max rel error':<22} {errors['max_rel_error']*100:.4f}%")
    print(f"  {'R^2':<22} {errors['r2']:.6f}")
    print(f"  {'-'*45}")


def print_parameter_comparison(true_params, pred_params, title="Parameter inversion"):
    print(f"\n  {title}")
    print(f"  {'param':<12} {'true':<14} {'inverted':<14} {'rel err':<10}")
    print(f"  {'-'*52}")

    total_rel_error = 0
    count = 0

    for key in true_params:
        if key in pred_params:
            true_val = true_params[key]
            pred_val = pred_params[key]
            rel_error = abs(pred_val - true_val) / (abs(true_val) + 1e-10) * 100
            print(f"  {key:<12} {true_val:<14.6f} {pred_val:<14.6f} {rel_error:>6.2f}%")
            total_rel_error += rel_error
            count += 1

    if count > 0:
        print(f"  {'-'*52}")
        print(f"  {'mean':<12} {'':<14} {'':<14} {total_rel_error/count:>6.2f}%")


def format_scientific(value, precision=4):
    if abs(value) < 1e-4 or abs(value) > 1e4:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}f}"
