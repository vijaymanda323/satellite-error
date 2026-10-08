import modal
import os

# Define the environment with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "numpy", "pandas", "scikit-learn", "matplotlib")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("gnss-lstm-training")
volume = modal.Volume.from_name("gnss-data-volume", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4", # Configurable GPU type (e.g., T4, A10g, A100)
    timeout=86400, # 24 hours
    volumes={"/data": volume},
)
def train():
    import sys
    sys.path.append("/root")
    from src.training.train_lstm import main
    main()

@app.function(
    image=image,
    volumes={"/data": volume},
)
def evaluate():
    import sys
    sys.path.append("/root")
    from src.evaluation.evaluate_lstm import main
    main()
