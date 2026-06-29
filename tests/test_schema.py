import pytest
from pydantic import ValidationError

from churn_prediction.schemas import CustomerPayload


def test_schema_accepts_valid_payload(sample_payload):
    assert CustomerPayload(**sample_payload).tenure == 12


def test_schema_rejects_invalid_values(sample_payload):
    payload = sample_payload | {"senior_citizen": 2}
    with pytest.raises(ValidationError):
        CustomerPayload(**payload)


def test_schema_rejects_unknown_fields(sample_payload):
    with pytest.raises(ValidationError):
        CustomerPayload(**(sample_payload | {"unknown": "x"}))
