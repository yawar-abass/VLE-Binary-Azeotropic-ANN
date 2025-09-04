import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def antoine_Psat(TK, A,B,C, T_unit='C', P_unit='mmHg'):
    # Convert Kelvin to required unit for Antoine (commonly Celsius)
    if T_unit.upper().startswith('C'):
        T = TK - 273.15
    else:
        T = TK
    # Antoine equation: log10(Psat) = A - B/(C+T)
    log10_Psat = A - (B/(C + T))
    Psat = 10**log10_Psat
    # convert Psat to Pa if needed
    if P_unit.lower() == 'mmhg':
        Psat_Pa = Psat * 133.322
    elif P_unit.lower() in ('bar','bars'):
        Psat_Pa = Psat * 1e5
    elif P_unit.lower() in ('kpa',):
        Psat_Pa = Psat * 1e3
    elif P_unit.lower() in ('pa',):
        Psat_Pa = Psat
    else:
        raise ValueError(f'Unsupported P_unit: {P_unit}')
    return Psat_Pa

def raoult_y1(x1, T, P, params):
    # params: dict for two components
    c1, c2 = params['component1'], params['component2']
    Psat1 = antoine_Psat(T, c1['A'], c1['B'], c1['C'], c1.get('T_unit','C'), c1.get('P_unit','mmHg'))
    Psat2 = antoine_Psat(T, c2['A'], c2['B'], c2['C'], c2.get('T_unit','C'), c2.get('P_unit','mmHg'))
    y1 = (x1 * Psat1) / (x1 * Psat1 + (1 - x1) * Psat2)
    # Ideal Raoult's at total P close to Psat1*x1 + Psat2*(1-x1); here we ignore P if provided
    return y1

def evaluate_baseline(data_csv: str, antoine_json: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(data_csv)
    with open(antoine_json,'r') as f:
        params = json.load(f)
    y_pred = df.apply(lambda r: raoult_y1(r['x1'], r['T'], r['P'], params), axis=1).values
    # metrics
    y_true = df['y1'].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    # plot
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('y₁ (data)'); plt.ylabel('y₁ (Raoult)'); plt.title('Baseline Parity')
    plt.grid(True)
    plt.savefig(os.path.join(outdir,'baseline_parity.png'), bbox_inches='tight', dpi=140)
    # save
    pd.DataFrame({'y1_true': y_true, 'y1_raoult': y_pred}).to_csv(os.path.join(outdir,'baseline_predictions.csv'), index=False)
    with open(os.path.join(outdir,'baseline_metrics.json'),'w') as f:
        json.dump({'MAE': mae, 'RMSE': rmse}, f, indent=2)
