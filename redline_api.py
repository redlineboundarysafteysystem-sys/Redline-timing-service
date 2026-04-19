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

WINDOW_SIZE = 6
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

intervals_window: deque[float] = deque(maxlen=WINDOW_SIZE)
score_history: deque[float] = deque(maxlen=SCORE_HISTORY_SIZE)

@app.post("/analyze", response_model=StateResponse)
async def analyze(data: TimestampInput, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"redline:Received {len(data.timestamps)} timestamps from {client_ip}")

    sorted_timestamps = sorted(data.timestamps)
    parsed_times = []
    for ts_str in sorted_timestamps:
        try:
            clean_ts = ts_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_ts)
            parsed_times.append(dt)
        except Exception as e:
            logger.warning(f"Failed to parse timestamp {ts_str}: {e}")
            continue

    if len(parsed_times) < 2:
        return StateResponse(
            human_summary="Rhythm looks healthy.",
            state="Stable",
            drift_score=0.0,
            baseline_interval_ms=0.0,
            current_interval_ms=0.0,
            message="Need at least 2 timestamps",
            events_processed=len(parsed_times),
            trend="Steady",
            trend_velocity=0.0
        )

    new_intervals = []
    for i in range(1, len(parsed_times)):
        delta = (parsed_times[i] - parsed_times[i-1]).total_seconds() * 1000
        new_intervals.append(delta)

    for interval in new_intervals:
        intervals_window.append(interval)

    if len(intervals_window) < 2:
        return StateResponse(
            human_summary="Rhythm looks healthy.",
            state="Stable",
            drift_score=0.0,
            baseline_interval_ms=0.0,
            current_interval_ms=0.0,
            message="Need at least 2 valid intervals",
            events_processed=len(parsed_times),
            trend="Steady",
            trend_velocity=0.0
        )

    baseline = np.mean(intervals_window)
    sigma = np.std(intervals_window, ddof=1) if len(intervals_window) > 1 else baseline * 0.001
    if sigma == 0:
        sigma = baseline * 0.001

    current_interval = intervals_window[-1]
    z_score = abs(current_interval - baseline) / sigma

    score_history.append(z_score)
    if len(score_history) >= 2:
        recent_avg = np.mean(list(score_history)[-3:])
        velocity = z_score - recent_avg
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

    # Final thresholds for real trust and clear state alignment
    if z_score < 1.5:
        state = "Stable"
        if trend == "Increasing":
            human_summary = "Rhythm looks healthy — but spacing is tightening slightly. Early compression forming."
            message = "Timing is healthy, but tightening slightly"
        elif trend == "Decreasing":
            human_summary = "Rhythm looks healthy — but spacing is widening slightly."
            message = "Timing is healthy, but widening slightly"
        else:
            human_summary = "Rhythm looks healthy."
            message = "Timing is healthy"
    elif z_score < 2.0:
        state = "Shifting"
        if trend == "Increasing":
            human_summary = "Nothing looked wrong yet… but timing already changed. Early upstream shift detected — and it’s accelerating."
            message = "Early timing drift forming - upstream warning"
        else:
            human_summary = "Nothing looked wrong yet… but timing already changed. Early upstream shift detected — and the drift is slowing."
            message = "Early timing drift forming - upstream warning"
    else:
        state = "Drift"
        if current_interval < baseline * 0.3:
            human_summary = "Cadence has moved sharply off baseline. Severe compression detected."
        else:
            human_summary = "Cadence has moved off baseline. Drift detected."
        if trend == "Increasing":
            human_summary += " — and it’s accelerating."
        message = "Critical – upstream shift detected"

    response = StateResponse(
        human_summary=human_summary,
        state=state,
        drift_score=round(z_score, 3),
        baseline_interval_ms=round(baseline, 0),
        current_interval_ms=round(current_interval, 0),
        message=message,
        events_processed=len(parsed_times),
        trend=trend,
        trend_velocity=trend_velocity
    )

    logger.info(f"Response: {state} | drift={response.drift_score} | trend={trend}")
    return response
