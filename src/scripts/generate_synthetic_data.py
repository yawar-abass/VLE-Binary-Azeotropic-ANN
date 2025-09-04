import argparse, numpy as np, pandas as pd

def toy_vle(x):
    # Non-ideal curve with slight azeotrope near x=0.95
    return 0.8 * x + 0.15 * x * (1 - x)

def main(out):
    xs = np.linspace(0,1,600)
    T = 351.0  # K (toy)
    P = 101325.0  # Pa
    ys = toy_vle(xs)
    # Add tiny noise to simulate experimental scatter
    rng = np.random.default_rng(42)
    ys = np.clip(ys + rng.normal(0, 0.01, size=ys.size), 0, 1)
    df = pd.DataFrame({'x1': xs, 'T': T, 'P': P, 'y1': ys})
    df.to_csv(out, index=False)
    print(f'Wrote {out} with {len(df)} rows.')

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    main(args.out)
