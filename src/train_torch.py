import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, joblib
from .config import TrainConfig
from .data import load_dataset, split_and_scale
from .model_torch import ANNBinaryVLE

def train(data_path: str, outdir: str, cfg: TrainConfig):
    os.makedirs(outdir, exist_ok=True)
    df = load_dataset(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = split_and_scale(
        df, cfg.val_split, cfg.test_split, cfg.seed
    )

    def to_tensor(a): return torch.tensor(a, dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(X_train), to_tensor(y_train))
    val_ds   = TensorDataset(to_tensor(X_val),   to_tensor(y_val))
    test_ds  = TensorDataset(to_tensor(X_test),  to_tensor(y_test))

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = ANNBinaryVLE(3, cfg.hidden_sizes, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss()  # MAE

    best_val = float('inf')
    patience, counter = cfg.patience, 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = sum(loss_fn(model(xv), yv).item() * len(xv) for xv, yv in val_dl) / len(val_ds)

        if vloss < best_val:
            best_val = vloss
            counter = 0
            torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
        else:
            counter += 1
            if counter > patience:
                break

    joblib.dump(scaler, os.path.join(outdir, 'x_scaler.joblib'))

    model.load_state_dict(torch.load(os.path.join(outdir, "model.pt")))
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(x) for x, _ in test_dl], 0).numpy()
    mae = float(np.mean(np.abs(y_test - preds)))
    rmse = float(np.sqrt(np.mean((y_test - preds)**2)))
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"MAE": mae, "RMSE": rmse}, f, indent=2)

    return os.path.join(outdir, "model.pt")
