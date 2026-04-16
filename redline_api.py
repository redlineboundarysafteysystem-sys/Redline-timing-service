from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
from datetime import datetime
import statistics
import logging

# Setup logging so we can see usage in Render logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redline")

app = FastAPI(title="RedLINE Timing Service")

# Configuration
WINDOW_SIZE = 20
DRIFT_THRESHOLD = 0.25
DRIFT_CONSECUTIVE = 3

class TimestampInput(BaseModel):
    timestamps: List[str]

class StateResponse(BaseModel):
    state: str
    drift_score: float
    baseline_interval_ms: float
    current_interval_ms: float
    message: str
    events_processed: int

events: List[datetime] = []

@app.post("/analyze", response_model=StateResponse)
async def analyze(data: TimestampInput, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"API call from {client_ip} | Received {len(data.timestamps)} timestamps")

    if not data.timestamps:
        raise HTTPException(status_code=400, detail="No timestamps provided")

    try:
        new_times = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in data.timestamps]
        events.extend(new_times)
    except Exception as e:
        logger.error(f"Invalid timestamp from {client_ip}: {e}")
        raise HTTPException(status_code=400, detail="Invalid ISO timestamp format")

    # Keep reasonable history
    if len(events) > WINDOW_SIZE * 3:
        del events[:-WINDOW_SIZE * 3]

    if len(events) < 2:
        logger.info(f"Still collecting baseline | events: {len(events)}")
        return StateResponse(
            state="Stable",
            drift_score=0.0,
            baseline_interval_ms=0.0,
            current_interval_ms=0.0,
            message="Collecting initial events...",
            events_processed=len(events)
        )

    # Calculate intervals
    intervals = [(events[i+1] - events[i]).total_seconds() * 1000 for i in range(len(events)-1)]
    
    recent = intervals[-WINDOW_SIZE:] if len(intervals) >= WINDOW_SIZE else intervals
    baseline = statistics.mean(recent)
    current = intervals[-1]

    deviation = abs(current - baseline) / baseline if baseline > 0 else 0
    drift_score = round(deviation, 3)

    recent_deviations = sum(1 for i in intervals[-DRIFT_CONSECUTIVE:] if abs(i - baseline)/baseline > DRIFT_THRESHOLD)

    if len(recent) < DRIFT_CONSECUTIVE:
        state = "Stable"
        message = "Still establishing baseline"
    elif deviation < DRIFT_THRESHOLD:
        state = "Stable"
        message = "Timing is healthy"
    elif recent_deviations >= DRIFT_CONSECUTIVE:
        state = "Drift"
        message = "Cadence has drifted — early action recommended"
    else:
        state = "Shifting"
        message = "Timing is starting to stretch — monitor closely"

    logger.info(f"Response → {client_ip} | state={state} | drift_score={drift_score} | baseline={round(baseline,1)}ms")

    return StateResponse(
        state=state,
        drift_score=drift_score,
        baseline_interval_ms=round(baseline, 1),
        current_interval_ms=round(current, 1),
        message=message,
        events_processed=len(events)
    )

@app.get("/test")
async def test():
    return {"message": "RedLINE API ready. POST to /analyze with {\"timestamps\": [\"2026-04-15T10:00:00\", ...]}"}
