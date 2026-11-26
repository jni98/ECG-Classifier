import torch
import torch.nn as nn


def _conv_bn_relu_block(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
    """Build Conv1d -> BatchNorm1d -> ReLU that keeps the signal length.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels (filters).
        k (int): Kernel size; padding k // 2 keeps the length for odd k.

    Returns:
        nn.Sequential: The three layers in order.
    """

    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=k, padding=k // 2
        ),
        # padding so length stays the same # out: B, out_ch, length
        nn.BatchNorm1d(out_ch),  # out: B, out_ch, length
        nn.ReLU(),
    )


class ECGClassifier(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5) -> None:
        """Small 1D-CNN for multi-label classification of 12-lead ECGs.

        Adapted from Elgazzar (2026), "Automated ECG Arrhythmia Classification
        Using Convolutional Neural Networks with Effective Class Imbalance
        Handling", Applied Sciences 16(11), 5321,
        https://doi.org/10.3390/app16115321.

        Changes to the original (single-beat, 1-lead, 360 Hz, multi-class):
        - Input: 12 leads x 1000 samples (10 s at 100 Hz).
        - Output: one logit per class for multi-label training with
          BCEWithLogitsLoss (no softmax).
        - Global average pooling instead of flatten + dense layer, so a pattern
          counts wherever it appears in the 10 s and the model has ~140k
          instead of ~400k parameters.
        - First kernel k=7 covers 70 ms at 100 Hz, close to a QRS complex
          (80-120 ms).

        Args:
            n_leads (int): Number of input channels (ECG leads).
            n_classes (int): Number of output classes.
        """

        super().__init__()

        # input shape: B, 12, 1000
        self.block_1 = nn.Sequential(
            _conv_bn_relu_block(in_ch=n_leads, out_ch=32, k=7),
            nn.MaxPool1d(2),  # out: B, 32, 500
            nn.Dropout(0.2),  # out: B, 32, 500
        )

        # input shape: B, 32, 500
        self.block_2 = nn.Sequential(
            _conv_bn_relu_block(in_ch=32, out_ch=64, k=5),
            nn.MaxPool1d(2),  # out: B, 64, 250
            nn.Dropout(0.2),  # out: B, 64, 250
        )

        # input shape: B, 64, 250
        self.block_3 = nn.Sequential(
            _conv_bn_relu_block(in_ch=64, out_ch=128, k=3),
            nn.MaxPool1d(2),  # out: B, 128, 125
            nn.Dropout(0.2),  # out: B, 128, 125
        )

        # input shape: B, 128, 125
        self.block_4 = nn.Sequential(
            _conv_bn_relu_block(in_ch=128, out_ch=256, k=3),
            nn.AdaptiveAvgPool1d(1),  # out: B, 256, 1
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=256, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of ECGs.

        Args:
            x (torch.Tensor): Signals of shape (batch, n_leads, time).

        Returns:
            torch.Tensor: Logits of shape (batch, n_classes). Apply
            torch.sigmoid to get probabilities.
        """

        out = self.block_1(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.flatten(out)
        out = self.linear(out)

        return out
