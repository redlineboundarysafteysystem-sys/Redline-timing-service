from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
from datetime import datetime
import numpy as np
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redline")

app = FastAPI(title="RedLINE Timing Service")

WINDOW_SIZE = 8                    # Responsive for demos
SCORE_HISTORY_SIZE = 10            # For trend

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

# In-memory rolling storage
events: deque[float] = deque(maxlen=WINDOW_SIZE)           # intervals in ms
score_history: deque[float] = deque(maxlen=SCORE_HISTORY_SIZE)

@app.post("/analyze", response_model=StateResponse)
async def analyze(data: TimestampInput, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Received {len(data.timestamps)} timestamps from {client_ip}")

    if len(data.timestamps) < 2:
        return {
            "human_summary": "Waiting for more data...",
            "state": "Stable",
            "drift_score": 0.0,
            "baseline_interval_ms": 0.0,
            "current_interval_ms": 0.0,
            "message": "Need at least 2 timestamps",
            "events_processed": 0,
            "trend": "Steady",
            "trend_velocity": 0.0
        }

    # Parse timestamps
    try:
        parsed_times = []
        for ts in data.timestamps:
            if ts.endswith('Z'):
                ts = ts[:-1] + '+00:00'
            dt = datetime.fromisoformat(ts)
            parsed_times.append(dt)
    except Exception as e:
        logger.error(f"Timestamp parse error: {e}")
        raise

    # Calculate intervals in ms
    intervals = []
    for i in range(1, len(parsed_times)):
        delta_ms = (parsed_times[i] - parsed_times[i-1]).total_seconds() * 1000
        intervals.append(delta_ms)

    # Add to rolling window
    for interval in intervals:
        events.append(interval)

    if len(events) < 2:
        return {
            "human_summary": "Still building baseline...",
            "state": "Stable",
            "drift_score": 0.0,
            "baseline_interval_ms": 0.0,
            "current_interval_ms": intervals[-1] if intervals else 0.0,
            "message": "Building baseline...",
            "events_processed": len(events),
            "trend": "Steady",
            "trend_velocity": 0.0
        }

    # Z-score calculation
    rolling_list = list(events)
    baseline = np.mean(rolling_list)
    sigma = np.std(rolling_list, ddof=1) if len(rolling_list) > 1 else max(baseline * 0.001, 1.0)

    current = intervals[-1]
    z_score = abs(current - baseline) / sigma

    # State determination
    if z_score < 1.8:                     # Slightly lowered for better demo sensitivity
        state = "Stable"
        message = "Timing is healthy"
        human_summary = "Rhythm looks healthy."
    elif z_score < 3.0:
        state = "Shifting"
        message = "Early timing drift forming - upstream warning"
        human_summary = "Nothing looked wrong yet… but timing already changed. Early upstream shift detected."
    else:
        state = "Drift"
        message = "Cadence has drifted - intervene now"
        human_summary = "Timing has clearly shifted. Time to act."

    # Trend calculation
    score_history.append(z_score)
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
            if state != "Stable":
                human_summary += " — but slowing down."
        else:
            trend = "Steady"
            trend_velocity = 0.0
    else:
        trend = "Steady"
        trend_velocity = 0.0

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

    logger.info(f"Response: {state} | drift={response['drift_score']} | trend={trend} | events={response['events_processed']}")
    return response
