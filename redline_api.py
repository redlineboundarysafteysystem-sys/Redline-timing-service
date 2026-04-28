from fastapi import FastAPI

app = FastAPI(title="RedLINE Timing Service")

@app.post("/analyze")
async def analyze(data: dict = None):
    if data is None:
        data = {}
    
    sessions = data.get("sessions") or data.get("timestamps") or data if isinstance(data, list) else []
    
    if len(sessions) < 2:
        return {"human_summary": "Need more data", "state": "Error"}
    
    try:
        numbers = [float(x) for x in sessions]
        avg = round(sum(numbers) / len(numbers), 1)
    except:
        return {"human_summary": "Numbers only please", "state": "Error"}
    
    return {
        "human_summary": "Rhythm looks healthy." if avg > 60 else "Timing is shifting.",
        "state": "Stable" if avg > 60 else "Shifting",
        "average_interval": avg,
        "events_processed": len(numbers)
    }
