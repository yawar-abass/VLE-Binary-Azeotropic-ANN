import os, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, joblib
from .model_torch import ANNBinaryVLE
from .data import load_dataset

def parity_plot(y_true, y_pred, outpath):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("y₁ data"); plt.ylabel("y₁ ANN")
    plt.title("Parity Plot"); plt.grid(True)
    plt.savefig(outpath, bbox_inches='tight', dpi=140)
    plt.close()

def evaluate(data_path, model_path, scaler_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = load_dataset(data_path)
    X = df[['x1','T','P']].values.astype('float32')
    y = df[['y1']].values.astype('float32')
    scaler = joblib.load(scaler_path)
    Xs = scaler.transform(X)
    model = ANNBinaryVLE()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(Xs)).numpy()
    mae = float(np.mean(np.abs(y - y_pred)))
    rmse = float(np.sqrt(np.mean((y - y_pred)**2)))
    with open(os.path.join(outdir,'eval_metrics.json'),'w') as f:
        json.dump({'MAE': mae, 'RMSE': rmse}, f, indent=2)
    parity_plot(y.flatten(), y_pred.flatten(), os.path.join(outdir,'parity.png'))
    df['y1_pred'] = y_pred
    df.to_csv(os.path.join(outdir,'predictions.csv'), index=False)
