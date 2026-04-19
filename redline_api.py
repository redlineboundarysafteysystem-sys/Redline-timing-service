from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime
import numpy as np
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redline")

app = FastAPI(title="RedLINE Timing Service")

WINDOW_SIZE = 8
SCORE_HISTORY_SIZE = 10

class TimestampInput(BaseModel):
    timestamps: List[str]

class StateResponse(BaseModel):
    human_summary: str
    state: str
    drift_score: float
    baseline_interval_ms: float
    current_interval_ms: float
    message: str
    events_processed: int
    trend: str
    trend_velocity: float

# In-memory storage
events: deque[datetime] = deque(maxlen=WINDOW_SIZE)
score_history: deque[float] = deque(maxlen=SCORE_HISTORY_SIZE)

@app.post("/analyze", response_model=StateResponse)
async def analyze(data: TimestampInput, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    
    # Sort timestamps chronologically BEFORE processing (fixes out-of-order inputs)
    try:
        sorted_timestamps = sorted(data.timestamps)
        logger.info(f"Received {len(sorted_timestamps)} timestamps from {client_ip} (sorted)")
    except Exception as e:
        logger.error(f"Failed to sort timestamps from {client_ip}: {e}")
        raise HTTPException(status_code=422, detail="Invalid timestamps format")

    events.clear()
    
    for ts_str in sorted_timestamps:
        try:
            # Handle both Z and timezone formats
            clean_ts = ts_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
            events.append(dt)
        except Exception as e:
            logger.warning(f"Failed to parse timestamp '{ts_str}' from {client_ip}: {e}")
            continue  # skip bad timestamps instead of crashing

    if len(events) < 2:
        return StateResponse(
            human_summary="Rhythm looks healthy.",
            state="Stable",
            drift_score=0.0,
            baseline_interval_ms=0.0,
            current_interval_ms=0.0,
            message="Need at least 2 timestamps",
            events_processed=len(events),
            trend="Steady",
            trend_velocity=0.0
        )

    # Calculate intervals in ms
    intervals = []
    for i in range(1, len(events)):
        delta = (events[i] - events[i-1]).total_seconds() * 1000
        intervals.append(delta)

    # Add to rolling window (we store intervals now)
    for interval in intervals:
        events.append(interval)   # reuse events deque for intervals

    if len(events) < 2:
        return StateResponse(
            human_summary="Still building baseline...",
            state="Stable",
            drift_score=0.0,
            baseline_interval_ms=0.0,
            current_interval_ms=0.0,
            message="Building baseline...",
            events_processed=len(events),
            trend="Steady",
            trend_velocity=0.0
        )

    # Z-score calculation
    rolling_list = list(events)
    baseline = np.mean(rolling_list)
    sigma = np.std(rolling_list, ddof=1) if len(rolling_list) > 1 else max(baseline * 0.001, 1.0)

    current = intervals[-1]
    z_score = abs(current - baseline) / sigma

    # State + human_summary
    if z_score < 1.8:
        state = "Stable"
        message = "Timing is healthy"
        human_summary = "Rhythm looks healthy."
    elif z_score < 3.0:
        state = "Shifting"
        message = "Early timing drift forming - upstream warning"
        human_summary = "Nothing looked wrong yet... but timing already changed. Early upstream shift detected."
    else:
        state = "Drift"
        message = "Cadence has drifted - intervene now"
        human_summary = "Timing has clearly shifted. Time to act."

    # Trend calculation
    score_history.append(z_score)
    trend = "Steady"
    trend_velocity = 0.0
    if len(score_history) >= 2:
        recent = list(score_history)[-3:]
        prev_avg = sum(recent) / len(recent)
        velocity = z_score - prev_avg
        buffer = 0.15
        if velocity > buffer:
            trend = "Increasing"
            trend_velocity = round(velocity, 3)
            if state != "Stable":
                human_summary += " — and it’s accelerating."
        elif velocity < -buffer:
            trend = "Decreasing"
            trend_velocity = round(velocity, 3)
            human_summary += " — but slowing down."

    response = {
        "human_summary": human_summary,
        "state": state,
        "drift_score": round(z_score, 3),
        "baseline_interval_ms": round(baseline, 1),
        "current_interval_ms": round(current, 1),
        "message": message,
        "events_processed": len(events),
        "trend": trend,
        "trend_velocity": trend_velocity
    }

    logger.info(f"Response: {state} | drift={response['drift_score']} | trend={trend} | events={len(events)}")
    return response
