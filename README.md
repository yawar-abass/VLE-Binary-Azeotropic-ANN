# ANN_BinaryVLE

ANN surrogate for binary VLE \( (x1, T, P) -> (y1) \)
with azeotrope detection and a Raoult's-law baseline.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (Optional) generate a synthetic dataset for a water–ethanol-like system at 1 atm
# python src/scripts/generate_synthetic_data.py --out data/gen_data.csv

# Train
python main.py train --data data/gen_data.csv --outdir outputs/ethanol_water

# Evaluate + parity plot
python main.py evaluate --data data/gen_data.csv --model outputs/ethanol_water/model.pt --outdir outputs/ethanol_water


# Detect azeotrope (grid scan at fixed P) – NOTE: use the .pt file
python main.py detect-azeotrope   --model outputs/ethanol_water/model.pt   --fixedP 101325   --Tmin 320   --Tmax 360   --nT 121   --outdir outputs/ethanol_water

# Baseline (Raoult's law) — requires Antoine parameters file (JSON), see `data/antoine_example.json`
python main.py baseline    --data data/gen_data.csv   --antoine data/antoine_example.json  --outdir outputs/ethanol_water

```

## Data format

CSV with columns: `x1,T,P,y1`. Mole fractions must be in [0,1], T in Kelvin, P in Pascals (recommended).

Example rows:

```
x1,T,P,y1
0.10,351.0,101325,0.25
...
```

## Report

The detailed project report can be downloaded here:
<a href="report.pdf" download style=>Project Report (PDF)</a>
