import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_sequences(df, feature_cols, target_cols, seq_length):
    """Creates overlapping sliding windows for a single satellite."""
    xs, ys = [], []
    # Drop rows with NaNs in required columns
    df = df.dropna(subset=feature_cols + target_cols)
    
    data_x = df[feature_cols].values
    data_y = df[target_cols].values
    
    for i in range(len(df) - seq_length):
        xs.append(data_x[i:(i + seq_length)])
        ys.append(data_y[i + seq_length])
        
    return np.array(xs), np.array(ys)

def main():
    print("Loading features.csv...")
    df = pd.read_csv("features.csv")
    
    # Sort chronologically
    df['epoch'] = pd.to_datetime(df['epoch'])
    df = df.sort_values(by=['satellite_id', 'epoch'])
    
    # NOTE: The current build_features.py doesn't seem to calculate 'orbit_error'. 
    # For the sake of completing the pipeline, if it's missing we'll mock it with 0s 
    # so the 2-target LSTM architecture won't break. 
    if 'orbit_error' not in df.columns:
        print("Warning: 'orbit_error' not found in features.csv. Mocking with zeros.")
        df['orbit_error'] = 0.0

    # Select numerical features for the model
    feature_cols = ['hour_sin', 'hour_cos', 'day_of_year'] 
    
    # Add any other numeric features that exist in the dataframe
    optional_features = ['sqrtA', 'eccentricity', 'inclination', 'raan', 'mean_anomaly']
    for col in optional_features:
        if col in df.columns:
            feature_cols.append(col)
            
    target_cols = ['clock_error', 'orbit_error']
    
    print(f"Using features: {feature_cols}")
    print(f"Using targets: {target_cols}")

    # 1. Temporal Train/Val/Test Split (70/15/15)
    # To prevent leakage, we split by TIME, not randomly.
    epochs = df['epoch'].unique()
    epochs = np.sort(epochs)
    
    train_split_idx = int(len(epochs) * 0.7)
    val_split_idx = int(len(epochs) * 0.85)
    
    train_thresh = epochs[train_split_idx]
    val_thresh = epochs[val_split_idx]
    
    df_train = df[df['epoch'] < train_thresh].copy()
    df_val = df[(df['epoch'] >= train_thresh) & (df['epoch'] < val_thresh)].copy()
    df_test = df[df['epoch'] >= val_thresh].copy()
    
    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}")

    # 2. Normalization (Fit on Train ONLY)
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit
    feature_scaler.fit(df_train[feature_cols])
    target_scaler.fit(df_train[target_cols])
    
    # Transform
    for d in [df_train, df_val, df_test]:
        d.loc[:, feature_cols] = feature_scaler.transform(d[feature_cols])
        d.loc[:, target_cols] = target_scaler.transform(d[target_cols])

    # 3. Create Sliding Windows
    seq_length = 24  # 24 steps window
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    # We must group by satellite so a sequence doesn't jump between G01 and G02
    print(f"Generating sequences of length {seq_length}...")
    for sat_id, sat_group in df_train.groupby('satellite_id'):
        x, y = create_sequences(sat_group, feature_cols, target_cols, seq_length)
        if len(x) > 0:
            X_train.append(x)
            y_train.append(y)
            
    for sat_id, sat_group in df_val.groupby('satellite_id'):
        x, y = create_sequences(sat_group, feature_cols, target_cols, seq_length)
        if len(x) > 0:
            X_val.append(x)
            y_val.append(y)
            
    for sat_id, sat_group in df_test.groupby('satellite_id'):
        x, y = create_sequences(sat_group, feature_cols, target_cols, seq_length)
        if len(x) > 0:
            X_test.append(x)
            y_test.append(y)

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_val = np.vstack(X_val)
    y_val = np.vstack(y_val)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)
    
    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")

    # 4. Save to Disk
    out_dir = "data/processed/windows"
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(f"{out_dir}/X_train.npy", X_train)
    np.save(f"{out_dir}/y_train.npy", y_train)
    np.save(f"{out_dir}/X_validation.npy", X_val)
    np.save(f"{out_dir}/y_validation.npy", y_val)
    np.save(f"{out_dir}/X_test.npy", X_test)
    np.save(f"{out_dir}/y_test.npy", y_test)
    
    with open(f"{out_dir}/feature_names.json", "w") as f:
        json.dump(feature_cols, f)
        
    with open(f"{out_dir}/window_config.json", "w") as f:
        json.dump({"sequence_length": seq_length}, f)
        
    # Save target scaler for inverse transform during evaluation
    scaler_dir = "data/processed/scalers"
    os.makedirs(scaler_dir, exist_ok=True)
    with open(f"{scaler_dir}/target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)
        
    print(f"\nAll files successfully generated and saved to {out_dir}/")

if __name__ == "__main__":
    main()
