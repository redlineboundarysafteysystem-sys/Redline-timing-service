from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import statistics

app = FastAPI(title="RedLINE Timing Service")

# Configuration
WINDOW_SIZE = 20          # number of events for baseline
DRIFT_THRESHOLD = 0.25    # 25% deviation = Shifting
DRIFT_CONSECUTIVE = 3     # how many consecutive deviations = Drift

class TimestampEvent(BaseModel):
    timestamp: str   # ISO format, e.g. "2026-04-15T10:02:00"

class StateResponse(BaseModel):
    state: str                    # "Stable", "Shifting", or "Drift"
    drift_score: float            # 0.0 to 1.0+ 
    baseline_interval_ms: float
    current_interval_ms: float
    message: str

# In-memory store for demo (last events)
events: List[datetime] = []

@app.post("/analyze", response_model=StateResponse)
async def analyze(event: TimestampEvent):
    try:
        ts = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ISO timestamp")

    events.append(ts)
    if len(events) > WINDOW_SIZE * 2:   # keep reasonable history
        events.pop(0)

    if len(events) < 2:
        return StateResponse(
            state="Stable",
            drift_score=0.0,
            baseline_interval_ms=0.0,
            current_interval_ms=0.0,
            message="Collecting initial events..."
        )

    # Calculate intervals
    intervals = [(events[i+1] - events[i]).total_seconds() * 1000 for i in range(len(events)-1)]
    
    # Baseline = average of last WINDOW_SIZE intervals
    recent_intervals = intervals[-WINDOW_SIZE:] if len(intervals) >= WINDOW_SIZE else intervals
    baseline = statistics.mean(recent_intervals)
    
    # Current interval (last one)
    current = intervals[-1]
    
    # Drift score (simple normalized deviation)
    deviation = abs(current - baseline) / baseline if baseline > 0 else 0
    drift_score = round(deviation, 3)

    # State logic
    if len(recent_intervals) < DRIFT_CONSECUTIVE:
        state = "Stable"
        message = "Still establishing baseline"
    elif deviation < DRIFT_THRESHOLD:
        state = "Stable"
        message = "Timing is healthy"
    elif sum(1 for i in intervals[-DRIFT_CONSECUTIVE:] if abs(i - baseline)/baseline > DRIFT_THRESHOLD) >= DRIFT_CONSECUTIVE:
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
        message=message
    )

# Keep a simple test endpoint for quick demo
@app.get("/test")
async def test():
    return {"message": "RedLINE API is running. Send POST to /analyze with {\"timestamp\": \"ISO string\"}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
