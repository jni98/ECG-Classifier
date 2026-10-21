import numpy as np

from ecg.features import extract_features


def test_extract_features_shape():

    features = extract_features(np.zeros((2, 1000, 12)))
    assert features.shape == (2, 48), (
        f"Expected feature vector shape to be (2,48) but got {features.shape} instead"
    )


def test_extract_features_mean_std():
    features = extract_features(np.ones((2, 1000, 12)))
    assert np.all(features[:, 0:12] == 1)
    assert np.all(features[:, 12:24] == 0)
