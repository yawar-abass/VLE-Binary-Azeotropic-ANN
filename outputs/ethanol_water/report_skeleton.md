# ANN Surrogate for Binary Azeotropic VLE

## 1. Dataset
- System: <ethanol–water / acetone–chloroform / ...>
- Source: <literature / NIST / simulation tool>
- Size: <N points>, coverage near azeotrope: <describe>

## 2. Methods
### 2.1 Preprocessing
- Normalization, splits (train/val/test)
### 2.2 ANN
- Inputs: x1, T, P; Output: y1 (sigmoid)
- Architecture: <layers/units>
- Training: optimizer, loss, early stopping
### 2.3 Baseline
- Raoult's law with Antoine coefficients

## 3. Results
- Parity plots (ANN vs data, baseline vs data)
- Metrics: MAE, RMSE
- Azeotrope detection: predicted (x*, T*) vs reference

## 4. Discussion
- Error analysis near azeotrope
- Limitations, future work (physics-informed losses, multi-P/T)

## 5. References
- Data sources, models, equations
