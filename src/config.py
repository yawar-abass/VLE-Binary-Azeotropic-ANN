from dataclasses import dataclass

@dataclass
class TrainConfig:
    hidden_sizes: tuple = (64, 64, 32)
    dropout: float = 0.05
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 500
    patience: int = 30  # early stopping
    val_split: float = 0.15
    test_split: float = 0.15
    seed: int = 42
