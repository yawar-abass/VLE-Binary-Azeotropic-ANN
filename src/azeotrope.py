import os, json
import numpy as np, matplotlib.pyplot as plt
import joblib, torch
from .model_torch import ANNBinaryVLE   # our PyTorch network

def _grid_x(n=501):
    return np.linspace(0, 1, n, dtype='float32')

def detect_azeotrope_fixedP(model_path: str, scaler_path: str, P_fixed: float,
                             Tmin: float, Tmax: float, nT: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Load PyTorch model + scaler
    model = ANNBinaryVLE()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    scaler = joblib.load(scaler_path)

    xs = _grid_x()
    Ts = np.linspace(Tmin, Tmax, nT)
    best = {'gap': 1e9, 'x': None, 'T': None, 'y': None}

    for T in Ts:
        X = np.stack([xs, np.full_like(xs, T), np.full_like(xs, P_fixed)], axis=1)
        Xs = scaler.transform(X)
        with torch.no_grad():
            y_pred = model(torch.tensor(Xs)).numpy().flatten()
        gap = np.abs(y_pred - xs)
        idx = int(np.argmin(gap))
        if gap[idx] < best['gap']:
            best = {'gap': float(gap[idx]), 'x': float(xs[idx]), 'T': float(T), 'y': float(y_pred[idx])}

    # Plot y vs x at the best T
    X = np.stack([xs, np.full_like(xs, best['T']), np.full_like(xs, P_fixed)], axis=1)
    Xs = scaler.transform(X)
    with torch.no_grad():
        y_pred = model(torch.tensor(Xs)).numpy().flatten()

    plt.figure(figsize=(6, 5))
    plt.plot(xs, y_pred, label='ANN: y₁(x₁)')
    plt.plot([0, 1], [0, 1], '--', label='y=x')
    plt.scatter([best['x']], [best['y']], color='red', label='Closest to azeotrope', s=50)
    plt.xlabel('x₁'); plt.ylabel('y₁'); plt.title(f'Best T≈{best["T"]:.2f} K at P={P_fixed:.0f} Pa')
    plt.grid(True); plt.legend()
    plt.savefig(os.path.join(outdir, 'azeotrope_scan.png'), bbox_inches='tight', dpi=140)

    with open(os.path.join(outdir, 'azeotrope.json'), 'w') as f:
        json.dump(best, f, indent=2)
