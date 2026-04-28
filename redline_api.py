from fastapi import FastAPI, HTTPException
from typing import List, Union, Dict, Any
import numpy as np
from collections import deque

app = FastAPI(title="RedLINE Timing Service")

# Rolling window for baseline
WINDOW_SIZE = 8
interval_window = deque(maxlen=WINDOW_SIZE)

@app.post("/analyze")
async def analyze(input_data: Union[Dict[str, Any], List, None] = None):
    if input_data is None:
        raise HTTPException(status_code=400, detail="No data received")

    # Accept simple list OR {"sessions": [...]}
    if isinstance(input_data, list):
        sessions = input_data
    elif isinstance(input_data, dict):
        sessions = input_data.get("sessions") or input_data.get("timestamps") or []
    else:
        sessions = []

    if len(sessions) < 2:
        raise HTTPException(status_code=400, 
            detail="Send at least 2 numbers like [72, 78, 75, ...] or {\"sessions\": [72, 78, ...]}")

    # Convert to numbers
    try:
        sessions = [float(x) for x in sessions]
    except:
        raise HTTPException(status_code=400, detail="All values must be numbers")

    # Calculate current interval (average for demo)
    current = sum(sessions) / len(sessions)

    interval_window.append(current)

    if len(interval_window) < 2:
        return {
            "human_summary": "Building baseline...",
            "state": "Stable",
            "drift_score": 0.0,
            "baseline_interval": round(current, 1),
            "current_interval": round(current, 1),
            "events_processed": len(sessions)
        }

    baseline = np.mean(list(interval_window))
    drift_score = abs(current - baseline)

    if drift_score < 10:
        state = "Stable"
        summary = "Rhythm looks healthy."
    elif drift_score < 25:
        state = "Shifting"
        summary = "Nothing looked wrong yet... but timing already changed."
    else:
        state = "Drift"
        summary = "Cadence has moved sharply off baseline."

    return {
        "human_summary": summary,
        "state": state,
        "drift_score": round(drift_score, 2),
        "baseline_interval": round(baseline, 1),
        "current_interval": round(current, 1),
        "events_processed": len(sessions)
    }
