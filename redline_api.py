from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict
import numpy as np

app = FastAPI(title="RedLINE Timing Service API")

class TimestampInput(BaseModel):
    timestamps: List[str]   # Example: ["2026-04-09T10:00:00", ...]

@app.post("/analyze")
async def analyze_timestamps(data: TimestampInput) -> Dict:
    """
    RedLINE Timing Service
    Send timestamps → get Stable / Shifting / Drift
    """
    try:
        times = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in data.timestamps]
        
        if len(times) < 3:
            return {"error": "Need at least 3 timestamps"}

        intervals = [(times[i+1] - times[i]).total_seconds() / 60 for i in range(len(times)-1)]
        
        baseline = np.mean(intervals[:4]) if len(intervals) > 4 else np.mean(intervals)
        
        shift_threshold = 1.25
        drift_threshold = 1.60
        
        states = []
        for gap in intervals:
            if gap <= baseline * shift_threshold:
                states.append("Stable")
            elif gap <= baseline * drift_threshold:
                states.append("Shifting")
            else:
                states.append("Drift")
        
        return {
            "states": states,
            "baseline_minutes": round(float(baseline), 2),
            "intervals_minutes": [round(float(g), 2) for g in intervals],
            "message": "RedLINE upstream signal generated"
        }
        
    except Exception as e:
        return {"error": f"Invalid timestamp format: {str(e)}"}
