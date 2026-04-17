from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from datetime import datetime
import statistics
import logging
from collections import deque
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redline")

app = FastAPI(title="RedLINE Timing Service")

WINDOW_SIZE = 8   # Lowered for faster demo/testing (shows Shifting/Drift quicker)
DRIFT_THRESHOLD = 2.0   # Z-score threshold for Shifting
DRIFT_CRITICAL = 3.0    # Z-score threshold for Drift

class TimestampInput(BaseModel):
    timestamps: List[str]

class StateResponse(BaseModel):
    state: str
    drift_score: float
    baseline_interval_ms: float
    current_interval_ms: float
    message: str
    events_processed: int

events: deque[datetime] = deque(maxlen=WINDOW_SIZE)

@app.post("/analyze", response_model=StateResponse)
async def analyze(data: TimestampInput, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"API call from {client_ip} | Received {len(data.timestamps)} timestamps")

    if len(data.timestamps) < 2:
        raise HTTPException(status_code=400, detail="At least 2 timestamps required")

    try:
        parsed = []
        for ts in data.timestamps:
            if ts.endswith('Z'):
                ts = ts[:-1] + '+00:00'
            dt = datetime.fromisoformat(ts)
            parsed.append(dt)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid timestamp format: {str(e)}")

    # Calculate intervals in ms
    intervals = []
    for i in range(1, len(parsed)):
        delta = (parsed[i] - parsed[i-1]).total_seconds() * 1000
        intervals.append(delta)

    # Add new intervals to rolling window
    for interval in intervals:
        events.append(interval)  # events now stores intervals directly

    if len(events) < 2:
        return {
            "state": "Stable",
            "drift_score": 0.0,
            "baseline_interval_ms": 0.0,
            "current_interval_ms": intervals[-1] if intervals else 0.0,
            "message": "Building baseline...",
            "events_processed": len(events)
        }

    # Calculate baseline and Z-score
    rolling_list = list(events)
    baseline = np.mean(rolling_list)
    sigma = np.std(rolling_list, ddof=1) if len(rolling_list) > 1 else baseline * 0.001
    if sigma == 0:
        sigma = 1.0

    current_interval = intervals[-1]
    z_score = abs(current_interval - baseline) / sigma

    if z_score < DRIFT_THRESHOLD:
        state = "Stable"
        message = "Timing is healthy"
    elif z_score < DRIFT_CRITICAL:
        state = "Shifting"
        message = "Early timing drift forming - upstream warning"
    else:
        state = "Drift"
        message = "Cadence has drifted - intervene now"

    response = {
        "state": state,
        "drift_score": round(z_score, 3),
        "baseline_interval_ms": round(baseline, 1),
        "current_interval_ms": round(current_interval, 1),
        "message": message,
        "events_processed": len(events)
    }

    logger.info(f"Response → {client_ip} | state={state} | drift_score={response['drift_score']} | events={response['events_processed']}")
    return response
