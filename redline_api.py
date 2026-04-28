from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from collections import deque
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RedLINE Timing Service")

# Constants
WINDOW_SIZE = 8
SCORE_HISTORY_SIZE = 10

interval_window = deque(maxlen=WINDOW_SIZE)
score_history = deque(maxlen=SCORE_HISTORY_SIZE)

class TimestampInput(BaseModel):
    timestamps: List[Union[str, float, int]] = None

@app.post("/analyze")
async def analyze(input_data: Union[TimestampInput, List, dict] = None):
    # Extract the list of values
    if isinstance(input_data, dict) and "timestamps" in input_data:
        raw = input_data["timestamps"]
    elif isinstance(input_data, list):
        raw = input_data
    elif isinstance(input_data, TimestampInput):
        raw = input_data.timestamps
    else:
        raw = []

    if len(raw) < 2:
        return {"error": "Need at least 2 values. Send timestamps or intervals like [72, 78, 75, ...]"}

    # If all numbers → treat as intervals in seconds
    if all(isinstance(x, (int, float)) for x in raw):
        base_time = datetime(2026, 4, 28, 14, 0, 0)
        timestamps = []
        current = base_time
        timestamps.append(current.isoformat())
        for interval in raw[:-1]:
            current += timedelta(seconds=float(interval))
            timestamps.append(current.isoformat())
    else:
        timestamps = [str(t) for t in raw]

    # Parse timestamps
    parsed_times = []
    for ts in timestamps:
        try:
            dt_str = str(ts).replace("Z", "+00:00")
            dt = datetime.fromisoformat(dt_str)
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            parsed_times.append(dt)
        except:
            continue

    if len(parsed_times) < 2:
        return {"error": "Could not parse enough valid timestamps"}

    # Calculate intervals
    intervals = []
    for i in range(1, len(parsed_times)):
        delta = (parsed_times[i] - parsed_times[i-1]).total_seconds() * 1000
        intervals.append(delta)

    if not intervals:
        return {"error": "No intervals calculated"}

    # Rolling baseline + z-score
    interval_window.extend(intervals)
    baseline = np.mean(interval_window)
    sigma = np.std(interval_window, ddof=1) if np.std(interval_window, ddof=1) > 0 else (baseline * 0.001 or 1)
    current_interval = intervals[-1]
    z_score = abs(current_interval - baseline) / sigma

    # Trend
    score_history.append(z_score)
    if len(score_history) >= 2:
        recent = list(score_history)[-3:]
        prev_avg = np.mean(recent[:-1])
        velocity = z_score - prev_avg
        trend = "Increasing" if velocity > 0.15 else "Decreasing" if velocity < -0.15 else "Steady"
        trend_velocity = round(velocity, 3)
    else:
        trend = "Steady"
        trend_velocity = 0.0

    # State
    if z_score < 1.5:
        state = "Stable"
        human_summary = "Rhythm looks healthy."
        message = "Timing is healthy"
    elif z_score < 2.0:
        state = "Shifting"
        human_summary = "Nothing looked wrong yet... but timing already changed. Early upstream shift detected."
        message = "Early timing drift forming - upstream warning"
    else:
        state = "Drift"
        human_summary = "Cadence has moved sharply off baseline. Severe compression or expansion detected."
        message = "Critical — upstream timing collapse detected"

    response = {
        "human_summary": human_summary,
        "state": state,
        "drift_score": round(z_score, 3),
        "baseline_interval_ms": int(round(baseline)),
        "current_interval_ms": int(round(current_interval)),
        "message": message,
        "events_processed": len(parsed_times),
        "trend": trend,
        "trend_velocity": trend_velocity
    }

    logger.info(f"Processed {len(parsed_times)} events → State: {state}, Drift: {response['drift_score']}")
    return response
