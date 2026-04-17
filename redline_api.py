from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from datetime import datetime
import numpy as np
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redline")

app = FastAPI(title="RedLINE Timing Service")

# Configuration
WINDOW_SIZE = 20
events = deque(maxlen=WINDOW_SIZE)  # keeps only the last 20 intervals

class TimestampInput(BaseModel):
    timestamps: List[str]

@app.post("/analyze")
async def analyze(data: TimestampInput, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"API call from {client_ip} | Received {len(data.timestamps)} timestamps")

    if not data.timestamps:
        raise HTTPException(status_code=400, detail="No timestamps provided")

    # Parse timestamps
    parsed = []
    for ts_str in data.timestamps:
        try:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            parsed.append(dt)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp format: {ts_str}")

    if len(parsed) < 2:
        raise HTTPException(status_code=400, detail="At least 2 timestamps required")

    # Calculate intervals in milliseconds
    intervals = []
    for i in range(1, len(parsed)):
        delta = (parsed[i] - parsed[i-1]).total_seconds() * 1000
        intervals.append(delta)

    current_interval_ms = intervals[-1]

    # Add to rolling window
    events.append(current_interval_ms)

    if len(events) < 2:
        response = {
            "state": "Stable",
            "drift_score": 0.0,
            "baseline_interval_ms": 0.0,
            "current_interval_ms": round(current_interval_ms, 1),
            "message": "Building baseline...",
            "events_processed": len(events)
        }
        logger.info(f"Response → {client_ip} | state=Stable | building baseline")
        return response

    rolling_window = list(events)

    # Z-score logic with standard deviation
    baseline = np.mean(rolling_window)
    sigma = np.std(rolling_window, ddof=1)

    if sigma == 0:
        sigma = baseline * 0.001 if baseline > 0 else 1.0

    z_score = abs(current_interval_ms - baseline) / sigma

    if z_score < 2.0:
        state = "Stable"
        message = "Timing is healthy"
    elif z_score < 3.0:
        state = "Shifting"
        message = "Early timing drift forming — monitor closely"
    else:
        state = "Drift"
        message = "Cadence has drifted — early action recommended"

    response = {
        "state": state,
        "drift_score": round(z_score, 3),
        "baseline_interval_ms": round(baseline, 1),
        "current_interval_ms": round(current_interval_ms, 1),
        "message": message,
        "events_processed": len(events)
    }

    logger.info(f"Response → {client_ip} | state={state} | drift_score={response['drift_score']} | baseline={response['baseline_interval_ms']}ms")
    return response
# Mount the interactive docs
from fastapi.openapi.docs import get_swagger_ui_html

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url=app.openapi_url, title=app.title)
