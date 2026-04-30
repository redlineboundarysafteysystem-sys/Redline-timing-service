from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, Dict, Any
import numpy as np
from collections import deque
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RedLINE Timing Service")

# Constants
WINDOW_SIZE = 8
SCORE_HISTORY_SIZE = 10

# In-memory storage
interval_window = deque(maxlen=WINDOW_SIZE)
score_history = deque(maxlen=SCORE_HISTORY_SIZE)

class TimestampInput(BaseModel):
    timestamps: List[str]

class StateResponse(BaseModel):
    human_summary: str
    state: str
    drift_score: float
    baseline_interval_ms: int
    current_interval_ms: int
    message: str
    events_processed: int
    trend: str
    trend_velocity: float

@app.post("/analyze", response_model=StateResponse)
async def analyze(input_data: Union[Dict[str, Any], List, TimestampInput, None] = None):
    if input_data is None:
        input_data = {}

    # === FLEXIBLE INPUT HANDLING ===
    if isinstance(input_data, list):
        ts_list = input_data
    elif isinstance(input_data, dict):
        ts_list = input_data.get("timestamps") or input_data.get("sessions") or []
    elif isinstance(input_data, TimestampInput):
        ts_list = input_data.timestamps
    else:
        ts_list = []

    if len(ts_list) < 2:
        return StateResponse(
            human_summary="Need at least 2 values to analyze rhythm.",
            state="Error",
            drift_score=0.0,
            baseline_interval_ms=0,
            current_interval_ms=0,
            message="Insufficient data",
            events_processed=len(ts_list),
            trend="Steady",
            trend_velocity=0.0
        )

    # Handle raw numbers (sessions) OR timestamps
    if all(isinstance(x, (int, float)) for x in ts_list):
        # Raw number list (e.g. {"sessions": [72, 78, ...]})
        intervals = [float(x) for x in ts_list]
    else:
        # Timestamp parsing (your original logic)
        parsed_times = []
        for ts_str in ts_list:
            try:
                ts_clean = str(ts_str).strip().replace("Z", "+00:00")
                dt = datetime.fromisoformat(ts_clean)
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                parsed_times.append(dt)
            except Exception as e:
                logger.warning(f"Failed to parse timestamp: {ts_str} - {e}")
                continue

        if len(parsed_times) < 2:
            return StateResponse(
                human_summary="Could not parse enough valid timestamps.",
                state="Error",
                drift_score=0.0,
                baseline_interval_ms=0,
                current_interval_ms=0,
                message="Parsing failed",
                events_processed=len(parsed_times),
                trend="Steady",
                trend_velocity=0.0
            )

        intervals = []
        for i in range(1, len(parsed_times)):
            delta = parsed_times[i] - parsed_times[i-1]
            intervals.append(delta.total_seconds() * 1000)

    # Core analysis logic
    for interval in intervals:
        interval_window.append(interval)

    current_interval = intervals[-1]

    if len(interval_window) >= 2:
        baseline = np.mean(interval_window)
        sigma = np.std(interval_window, ddof=1)
        if sigma == 0:
            sigma = max(baseline * 0.001, 0.001)
        z_score = abs(current_interval - baseline) / sigma
    else:
        baseline = current_interval
        z_score = 0.0

    score_history.append(z_score)

    if len(score_history) >= 2:
        recent = list(score_history)[-3:]
        prev_avg = sum(recent[:-1]) / len(recent[:-1])
        velocity = z_score - prev_avg
        buffer = 0.15
        if velocity > buffer:
            trend = "Increasing"
        elif velocity < -buffer:
            trend = "Decreasing"
        else:
            trend = "Steady"
        trend_velocity = round(velocity, 3)
    else:
        trend = "Steady"
        trend_velocity = 0.0

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

    response = StateResponse(
        human_summary=human_summary,
        state=state,
        drift_score=round(z_score, 3),
        baseline_interval_ms=int(round(baseline)),
        current_interval_ms=int(round(current_interval)),
        message=message,
        events_processed=len(ts_list),
        trend=trend,
        trend_velocity=trend_velocity
    )

    logger.info(f"Processed {len(ts_list)} events → State: {state}, Drift: {response.drift_score}")
    return response
