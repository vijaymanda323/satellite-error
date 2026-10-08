import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import TensorDataset, DataLoader
from src.models.lstm import GNSSLSTM
from src.utils.checkpoint import load_checkpoint
from src.utils.metrics import calculate_metrics

def inverse_transform_targets(y, scaler_path="/data/processed/scalers/target_scaler.pkl"):
    if os.path.exists(scaler_path):
        print(f"Loading target scaler from {scaler_path}")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return scaler.inverse_transform(y)
    else:
        print("No target scaler found. Returning raw values.")
        return y

def plot_predictions(y_true, y_pred, target_name, results_dir):
    plt.figure(figsize=(15, 5))
    # Plot first 1000 points to keep it legible
    n_points = min(1000, len(y_true))
    plt.plot(y_true[:n_points], label='Actual', alpha=0.7)
    plt.plot(y_pred[:n_points], label='Predicted', alpha=0.7)
    plt.title(f'Actual vs Predicted {target_name.capitalize()} Error (Sample)')
    plt.xlabel('Time Step')
    plt.ylabel(f'{target_name.capitalize()} Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'actual_vs_predicted_{target_name}.png'))
    plt.close()
    
    # Residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=50, alpha=0.7)
    plt.title(f'Prediction Residuals for {target_name.capitalize()} Error')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f'residuals_{target_name}.png'))
    plt.close()

def main():
    data_dir = "/data/processed/windows"
    results_dir = "/data/results/lstm"
    checkpoint_path = os.path.join(results_dir, "best_lstm.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")
    
    # Load test data
    print("Loading test data...")
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Create loader
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load checkpoint to get config and rebuild model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint['config']
    
    model = GNSSLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=config["output_size"]
    ).to(device)
    
    load_checkpoint(checkpoint_path, model)
    model.eval()
    
    predictions = []
    actuals = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
            actuals.append(batch_y.numpy())
            
    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)
    
    # Inverse transform
    predictions = inverse_transform_targets(predictions)
    actuals = inverse_transform_targets(actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    # Create final report
    print("\n" + "="*50)
    print("LSTM TRAINING SUMMARY")
    print("="*50)
    print(f"Dataset:")
    print(f"Sequence length: {config['sequence_length']}")
    print(f"Number of features: {config['input_size']}")
    print(f"Number of test samples: {len(X_test)}")
    
    print(f"\nModel:")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Layers: {config['num_layers']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    print(f"\nTraining:")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_validation_loss']:.6f}")
    
    print("\nTest Results:")
    print("-" * 30)
    print("Clock Error:")
    print(f"MAE:  {metrics['clock']['MAE']:.4e}")
    print(f"RMSE: {metrics['clock']['RMSE']:.4e}")
    print(f"R²:   {metrics['clock']['R2']:.4f}")
    
    print("-" * 30)
    print("Orbit Error:")
    print(f"MAE:  {metrics['orbit']['MAE']:.4e}")
    print(f"RMSE: {metrics['orbit']['RMSE']:.4e}")
    print(f"R²:   {metrics['orbit']['R2']:.4f}")
    print("="*50)
    
    # Save results to JSON
    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plotting
    print("Generating plots...")
    plot_predictions(actuals[:, 0], predictions[:, 0], "clock", results_dir)
    plot_predictions(actuals[:, 1], predictions[:, 1], "orbit", results_dir)
    
    # Save predictions to CSV
    print("Saving predictions to CSV...")
    import pandas as pd
    df_preds = pd.DataFrame({
        'actual_clock_error': actuals[:, 0],
        'predicted_clock_error': predictions[:, 0],
        'actual_orbit_error': actuals[:, 1],
        'predicted_orbit_error': predictions[:, 1]
    })
    preds_csv_path = os.path.join(results_dir, "predictions.csv")
    df_preds.to_csv(preds_csv_path, index=False)
    
    print(f"Plots and predictions saved to {results_dir}")

if __name__ == "__main__":
    main()
