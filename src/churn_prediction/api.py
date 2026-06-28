import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response

from churn_prediction.artifacts import artifacts_ready
from churn_prediction.logging_config import configure_logging
from churn_prediction.schemas import CustomerPayload, PredictionResponse

configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(title="Churn Prediction API")
_predictor = None


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.exception("request_error", extra={"route": request.url.path, "error": str(exc)})
        raise
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "request_finished",
        extra={
            "route": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


def get_predictor():
    global _predictor
    if _predictor is None:
        from churn_prediction.inference import ChurnPredictor

        _predictor = ChurnPredictor()
    return _predictor


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": artifacts_ready()}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerPayload) -> dict[str, object]:
    if not artifacts_ready():
        raise HTTPException(status_code=503, detail="Artefatos do modelo não encontrados")
    return get_predictor().predict_one(payload.model_dump())
