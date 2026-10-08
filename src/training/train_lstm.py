import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from src.models.lstm import GNSSLSTM
from src.utils.checkpoint import save_checkpoint

def validate_data(X, y, name, expected_features):
    print(f"Validating {name} data...")
    if len(X) != len(y):
        raise ValueError(f"{name}: X samples ({len(X)}) != y samples ({len(y)})")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError(f"{name} contains NaN values")
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError(f"{name} contains Infinite values")
    if len(X.shape) != 3:
        raise ValueError(f"{name} X shape {X.shape} is not 3D")
    if len(y.shape) != 2:
        raise ValueError(f"{name} y shape {y.shape} is not 2D")
    if y.shape[1] != 2:
        raise ValueError(f"{name} y target columns {y.shape[1]} != 2")
    if X.shape[2] != expected_features:
        raise ValueError(f"{name} features {X.shape[2]} != expected {expected_features}")
    if len(X) == 0:
        raise ValueError(f"{name} dataset is empty")
    print(f"{name} data validation passed.")

def plot_loss(train_losses, val_losses, results_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'))
    plt.close()

def main():
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    print("Random seed set to 42")

    data_dir = "/data/processed/windows"
    results_dir = "/data/results/lstm"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load config and features
    with open(os.path.join(data_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    num_features = len(feature_names)
    
    with open(os.path.join(data_dir, "window_config.json"), "r") as f:
        window_config = json.load(f)
    seq_length = window_config.get("sequence_length", 0)

    # Load data
    print("Loading datasets...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_validation.npy"))
    y_val = np.load(os.path.join(data_dir, "y_validation.npy"))
    
    # Validate
    validate_data(X_train, y_train, "Train", num_features)
    validate_data(X_val, y_val, "Validation", num_features)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Hyperparameters
    config = {
        "input_size": num_features,
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "output_size": 2,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "patience": 15,
        "max_norm": 1.0,
        "sequence_length": seq_length,
        "optimizer": "Adam",
        "loss_function": "MSELoss",
        "device": device,
        "seed": 42
    }
    
    # Save config
    with open(os.path.join(results_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # DataLoaders
    # We DO NOT shuffle to maintain temporal order. Shuffling training windows is generally safe from 
    # future leakage *if* the train/val/test split is strictly chronological (which it is here),
    # but sequential processing aligns better with RNN context and sliding windows natively.
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model
    model = GNSSLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=config["output_size"]
    ).to(device)
    
    # Print parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model parameters: {num_params}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training Loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(config["epochs"]):
        start_time = time.time()
        
        # Train
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_norm"])
            
            optimizer.step()
            epoch_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validate
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:03d}/{config['epochs']} | "
              f"Train Loss: {epoch_train_loss:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']} | "
              f"Time: {elapsed_time:.2f} sec")
              
        # Early Stopping & Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_validation_loss': best_val_loss,
                'config': config,
                'feature_names': feature_names,
                'sequence_length': seq_length
            }
            save_checkpoint(state, True, results_dir)
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")
    plot_loss(train_losses, val_losses, results_dir)

if __name__ == "__main__":
    main()
