from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RedLINE Timing Service")

# Per-request window size for internal calculations
WINDOW_SIZE = 8

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
    jitter_pct: float
    early_warning_score: float

@app.post("/analyze", response_model=StateResponse)
async def analyze(timestamps: Union[TimestampInput, List[str]]):
    # Support both { "timestamps": [...] } and bare [ ... ]
    if isinstance(timestamps, TimestampInput):
        ts_list = timestamps.timestamps
    else:
        ts_list = timestamps

    if len(ts_list) < 2:
        return StateResponse(
            human_summary="Need at least 2 timestamps to analyze rhythm.",
            state="Error",
            drift_score=0.0,
            baseline_interval_ms=0,
            current_interval_ms=0,
            message="Insufficient data",
            events_processed=len(ts_list),
            trend="Steady",
            trend_velocity=0.0,
            jitter_pct=0.0,
            early_warning_score=0.0
        )

    # Parse timestamps and make them naive (remove timezone)
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
            trend_velocity=0.0,
            jitter_pct=0.0,
            early_warning_score=0.0
        )

    # Calculate intervals in milliseconds
    intervals = []
    for i in range(1, len(parsed_times)):
        delta = parsed_times[i] - parsed_times[i - 1]
        intervals.append(delta.total_seconds() * 1000)

    if not intervals:
        return StateResponse(
            human_summary="No valid intervals could be calculated.",
            state="Error",
            drift_score=0.0,
            baseline_interval_ms=0,
            current_interval_ms=0,
            message="No intervals",
            events_processed=len(parsed_times),
            trend="Steady",
            trend_velocity=0.0,
            jitter_pct=0.0,
            early_warning_score=0.0
        )

    # Use only the last WINDOW_SIZE intervals from THIS request
    if len(intervals) > WINDOW_SIZE:
        window = intervals[-WINDOW_SIZE:]
    else:
        window = intervals[:]

    current_interval = window[-1]

    # Baseline + sigma from this request only
    if len(window) >= 2:
        baseline = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        if sigma == 0:
            sigma = max(baseline * 0.001, 0.001)
    else:
        baseline = float(current_interval)
        sigma = max(baseline * 0.001, 0.001)

    # Core drift metric (Z-score) for current interval
    z_score = abs(current_interval - baseline) / sigma

    # Trend / velocity: compare last z to previous z-average in this request
    if len(window) >= 3:
        z_scores = []
        for iv in window:
            z = abs(iv - baseline) / sigma if sigma > 0 else 0.0
            z_scores.append(z)

        recent = z_scores[-3:]
        if len(recent) >= 2:
            prev_avg = sum(recent[:-1]) / len(recent[:-1])
            velocity = z_scores[-1] - prev_avg
        else:
            velocity = 0.0

        buffer = 0.15
        if velocity > buffer:
            trend = "Increasing"
        elif velocity < -buffer:
            trend = "Decreasing"
        else:
            trend = "Steady"
        trend_velocity = round(float(velocity), 3)
    else:
        trend = "Steady"
        trend_velocity = 0.0

    # Jitter: how far this interval is from baseline (percentage)
    if baseline > 0:
        jitter_pct = abs(current_interval - baseline) / baseline
    else:
        jitter_pct = 0.0
    jitter_pct = round(float(jitter_pct), 4)

    # Early warning score
    early_warning_score = (
        0.5 * float(z_score) +
        0.3 * abs(jitter_pct) +
        0.2 * abs(trend_velocity)
    )
    early_warning_score = round(float(early_warning_score), 3)

    # --- State classification with collapse override ---

    # Detect severe compression (micro-collapse)
    is_compression = current_interval < baseline
    is_hard_jitter = jitter_pct >= 0.80  # 80%+ deviation
    is_collapse = is_compression and is_hard_jitter

    # Override: treat severe compression as Drift even if z < 1.5
    if is_collapse and z_score >= 1.0:
        state = "Drift"
        human_summary = "Cadence has moved sharply off baseline. Severe compression or expansion detected."
        message = "Critical — upstream timing collapse detected"

    elif z_score < 1.5:
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

    # Build response
    response = StateResponse(
        human_summary=human_summary,
        state=state,
        drift_score=round(float(z_score), 3),
        baseline_interval_ms=int(round(baseline)),
        current_interval_ms=int(round(current_interval)),
        message=message,
        events_processed=len(parsed_times),
        trend=trend,
        trend_velocity=trend_velocity,
        jitter_pct=jitter_pct,
        early_warning_score=early_warning_score
    )

    logger.info(
        f"Processed {len(parsed_times)} events → "
        f"State: {state}, Drift: {response.drift_score}, "
        f"Jitter: {response.jitter_pct}, EWS: {response.early_warning_score}"
    )
    return response
