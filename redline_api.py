from fastapi import FastAPI, HTTPException
from typing import List, Union, Dict

app = FastAPI(title="RedLINE Timing Service")

@app.post("/analyze")
async def analyze(data: Union[Dict, List, None] = None):
    if data is None:
        data = {}

    # Accept simple list or {"sessions": [...]}
    if isinstance(data, list):
        sessions = data
    else:
        sessions = data.get("sessions") or data.get("timestamps") or []

    if len(sessions) < 2:
        raise HTTPException(status_code=400, detail="Send at least 2 numbers: [72, 78, 75, ...]")

    try:
        sessions = [float(x) for x in sessions]
    except:
        raise HTTPException(status_code=400, detail="All values must be numbers")

    avg = sum(sessions) / len(sessions)

    return {
        "human_summary": "Rhythm looks healthy." if avg > 60 else "Timing is shifting.",
        "state": "Stable" if avg > 60 else "Shifting",
        "average_interval": round(avg, 1),
        "events_processed": len(sessions)
    }
