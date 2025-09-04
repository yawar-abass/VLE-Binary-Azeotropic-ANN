import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_dataset(path: str):
    df = pd.read_csv(path)
    required = {'x1','T','P','y1'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # basic sanity
    for col in ['x1','y1']:
        if not ((df[col] >= -1e-6) & (df[col] <= 1+1e-6)).all():
            raise ValueError(f"{col} values must be within [0,1].")
    return df

def split_and_scale(df: pd.DataFrame, val_split=0.15, test_split=0.15, seed=42):
    X = df[['x1','T','P']].values.astype('float32')
    y = df[['y1']].values.astype('float32')

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val_split+test_split, random_state=seed)
    relative_test = test_split/(val_split+test_split) if (val_split+test_split)>0 else 0.5
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_test, random_state=seed)

    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_val_s   = x_scaler.transform(X_val)
    X_test_s  = x_scaler.transform(X_test)

    return (X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, x_scaler)
