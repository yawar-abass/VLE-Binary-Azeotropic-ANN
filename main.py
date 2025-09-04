import argparse, os, json
from src.config import TrainConfig
from src.train_torch import train
from src.evaluate_torch import evaluate
from src.azeotrope import detect_azeotrope_fixedP
from src.baseline import evaluate_baseline

def cli():
    ap = argparse.ArgumentParser(description="ANN surrogate for Binary VLE (x1,T,P)->y1")
    sub = ap.add_subparsers(dest='cmd', required=True)

    ap_train = sub.add_parser('train', help='Train ANN model')
    ap_train.add_argument('--data', required=True)
    ap_train.add_argument('--outdir', required=True)

    ap_eval = sub.add_parser('evaluate', help='Evaluate trained model with parity plot')
    ap_eval.add_argument('--data', required=True)
    ap_eval.add_argument('--model', required=True)
    ap_eval.add_argument('--outdir', required=True)
    ap_eval.add_argument('--scaler', default=None)

    ap_az = sub.add_parser('detect-azeotrope', help='Grid-scan to find x~y at fixed P')
    ap_az.add_argument('--model', required=True)
    ap_az.add_argument('--fixedP', type=float, required=True)
    ap_az.add_argument('--Tmin', type=float, required=True)
    ap_az.add_argument('--Tmax', type=float, required=True)
    ap_az.add_argument('--nT', type=int, default=101)
    ap_az.add_argument('--outdir', required=True)
    ap_az.add_argument('--scaler', default=None)

    ap_bl = sub.add_parser('baseline', help='Evaluate Raoult baseline with Antoine coefficients')
    ap_bl.add_argument('--data', required=True)
    ap_bl.add_argument('--antoine', required=True)
    ap_bl.add_argument('--outdir', required=True)

    ap_rep = sub.add_parser('report-skeleton', help='Create a 2–3 page report template (Markdown)')
    ap_rep.add_argument('--out', required=True)

    args = ap.parse_args()

    if args.cmd == 'train':
        cfg = TrainConfig()
        model_path = train(args.data, args.outdir, cfg)
        print("Saved model at:", model_path)
        print("Scaler at:", os.path.join(args.outdir, 'x_scaler.joblib'))

    elif args.cmd == 'evaluate':
        scaler_path = args.scaler or os.path.join(os.path.dirname(args.model), 'x_scaler.joblib')
        evaluate(args.data, args.model, scaler_path, args.outdir)
        print("Wrote parity plot and metrics to:", args.outdir)

    elif args.cmd == 'detect-azeotrope':
        scaler_path = args.scaler or os.path.join(os.path.dirname(args.model), 'x_scaler.joblib')
        detect_azeotrope_fixedP(args.model, scaler_path, args.fixedP, args.Tmin, args.Tmax, args.nT, args.outdir)
        print("Azeotrope scan saved to:", args.outdir)

    elif args.cmd == 'baseline':
        evaluate_baseline(args.data, args.antoine, args.outdir)
        print("Baseline outputs to:", args.outdir)

    elif args.cmd == 'report-skeleton':
        md = (
            "# ANN Surrogate for Binary Azeotropic VLE\n\n"
            "## 1. Dataset\n"
            "- System: <ethanol–water / acetone–chloroform / ...>\n"
            "- Source: <literature / NIST / simulation tool>\n"
            "- Size: <N points>, coverage near azeotrope: <describe>\n\n"
            "## 2. Methods\n"
            "### 2.1 Preprocessing\n"
            "- Normalization, splits (train/val/test)\n"
            "### 2.2 ANN\n"
            "- Inputs: x1, T, P; Output: y1 (sigmoid)\n"
            "- Architecture: <layers/units>\n"
            "- Training: optimizer, loss, early stopping\n"
            "### 2.3 Baseline\n"
            "- Raoult's law with Antoine coefficients\n\n"
            "## 3. Results\n"
            "- Parity plots (ANN vs data, baseline vs data)\n"
            "- Metrics: MAE, RMSE\n"
            "- Azeotrope detection: predicted (x*, T*) vs reference\n\n"
            "## 4. Discussion\n"
            "- Error analysis near azeotrope\n"
            "- Limitations, future work (physics-informed losses, multi-P/T)\n\n"
            "## 5. References\n"
            "- Data sources, models, equations\n"
        )
        with open(args.out, 'w') as f:
            f.write(md)
        print("Report template written to:", args.out)

if __name__ == "__main__":
    cli()
