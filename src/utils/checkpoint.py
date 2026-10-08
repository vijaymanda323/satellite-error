import torch
import os

def save_checkpoint(state, is_best, checkpoint_dir, filename="best_lstm.pt"):
    """Saves model and training parameters at checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    if is_best:
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(state, filepath)
        print(f"Saved new best model to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    """Loads model and training parameters from checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at {filepath}")
    
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint
