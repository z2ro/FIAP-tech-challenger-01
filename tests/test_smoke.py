from churn_prediction.inference import ChurnPredictor


def test_smoke_predict_probability_between_zero_and_one(test_artifacts, sample_payload):
    pred = ChurnPredictor().predict_one(sample_payload)
    assert 0 <= pred["churn_probability"] <= 1
