import numpy as np

from churn_prediction.preprocessing import fit_transform_preprocessor


def test_preprocessing_numeric_no_nan_consistent(sample_frame):
    preprocessor, x_train = fit_transform_preprocessor(sample_frame)
    x_again = preprocessor.transform(sample_frame)
    assert np.issubdtype(x_train.dtype, np.number)
    assert not np.isnan(x_train).any()
    assert x_train.shape[1] == x_again.shape[1]
