from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Union
from datetime import datetime
from dateutil import parser
import numpy as np
import logging
from collections import deque

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
async def analyze(data: Union[TimestampInput, List[str]], request: Request):
    client_ip = request.client.host if request.client else "unknown"
    
    # Handle both {"timestamps": [...]} and raw list ["2026-..."]
    if isinstance(data, TimestampInput):
        ts_list = data.timestamps
    else:
        ts_list = data

    logger.info(f"redline:Received {len(ts_list)} timestamps from {client_ip}")

    if len(ts_list) < 2:
        return StateResponse(
            human_summary="Rhythm looks healthy.",
            state="Stable",
            drift_score=0.0,
            baseline_interval_ms=0.0,
            current_interval_ms=0.0,
            message="Need at least 2 timestamps",
            events_processed=len(ts_list),
            trend="Steady",
            trend_velocity=0.0
        )

    try:
        # Parse any reasonable timestamp format using dateutil (very flexible)
        parsed_times = []
        for ts_str in ts_list:
            try:
                dt = parser.parse(ts_str)
                parsed_times.append(dt)
            except Exception as e:
                logger.warning(f"Failed to parse timestamp {ts_str}: {e}")
                continue

        if len(parsed_times) < 2:
            raise ValueError("Not enough valid timestamps after parsing")

        # Convert to unix timestamps (seconds)
        unix_times = [dt.timestamp() for dt in parsed_times]

        # Calculate intervals in milliseconds
        new_intervals = []
        for i in range(1, len(unix_times)):
            delta = (unix_times[i] - unix_times[i-1]) * 1000
            new_intervals.append(delta)

        # Add to rolling window
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

        # Determine state
        if z_score < 1.5:
            state = "Stable"
        elif z_score < 3.0:
            state = "Shifting"
        else:
            state = "Drift"

        # Simple trend
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
        else:
            trend = "Steady"

        score_history.append(z_score)

        # Human-friendly messages
        if state == "Stable":
            human_summary = "Rhythm looks healthy."
            message = "Timing is healthy"
        elif state == "Shifting":
            human_summary = "Nothing looked wrong yet... but timing already changed. Early upstream shift detected."
            message = "Early timing drift forming - upstream warning"
        else:
            human_summary = "Cadence has moved sharply off baseline. Severe compression detected."
            message = "Critical - upstream shift detected"

        response = StateResponse(
            human_summary=human_summary,
            state=state,
            drift_score=round(z_score, 3),
            baseline_interval_ms=round(baseline, 0),
            current_interval_ms=round(current_interval, 0),
            message=message,
            events_processed=len(parsed_times),
            trend=trend,
            trend_velocity=round(velocity, 3) if 'velocity' in locals() else 0.0
        )

        logger.info(f"Response: {state} | drift={response.drift_score} | trend={trend}")
        return response

    except Exception as e:
        logger.error(f"Error processing timestamps: {e}")
        raise HTTPException(status_code=400, detail=f"Could not parse timestamps: {str(e)}")
