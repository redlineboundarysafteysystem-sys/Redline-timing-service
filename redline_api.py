from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import statistics

app = FastAPI(title="RedLINE Timing Service")

# Configuration - feel free to tweak these
WINDOW_SIZE = 20           # events used for baseline
DRIFT_THRESHOLD = 0.25     # 25% deviation triggers Shifting
DRIFT_CONSECUTIVE = 3      # how many consecutive deviations = Drift

class TimestampInput(BaseModel):
    timestamps: List[str]   # List of ISO timestamps

class StateResponse(BaseModel):
    state: str
    drift_score: float
    baseline_interval_ms: float
    current_interval_ms: float
    message: str
    events_processed: int

events: List[datetime] = []   # stores timestamps for baseline

@app.post("/analyze", response_model=StateResponse)
async def analyze(data: TimestampInput):
    if not data.timestamps:
        raise HTTPException(status_code=400, detail="No timestamps provided")

    try:
        new_times = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in data.timestamps]
        events.extend(new_times)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ISO timestamp format")

    # Keep only recent history
    if len(events) > WINDOW_SIZE * 3:
        del events[:-WINDOW_SIZE * 3]

    if len(events) < 2:
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
    
    # Baseline from recent events
    recent = intervals[-WINDOW_SIZE:] if len(intervals) >= WINDOW_SIZE else intervals
    baseline = statistics.mean(recent)
    current = intervals[-1]
    
    deviation = abs(current - baseline) / baseline if baseline > 0 else 0
    drift_score = round(deviation, 3)

    # Determine state
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

    return StateResponse(
        state=state,
        drift_score=drift_score,
        baseline_interval_ms=round(baseline, 1),
        current_interval_ms=round(current, 1),
        message=message,
        events_processed=len(events)
    )

# Simple test endpoint
@app.get("/test")
async def test():
    return {"message": "RedLINE API ready. POST to /analyze with {\"timestamps\": [\"2026-04-15T10:00:00\", ...]}"}
