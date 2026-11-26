import torch.cuda.tunable

from ecg.model import ECGClassifier


def test_model_tensor_shape():
    input = torch.randn(8, 12, 1000)

    model = ECGClassifier()
    output = model.forward(input)

    assert output.shape == (8, 5), (
        f"Expected output to have shape (8.5) but got {output.shape} instead"
    )
