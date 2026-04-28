@app.post("/analyze")
async def analyze(body: dict = None):
    if body is None:
        body = {}

    # Accept different ways people send data
    if isinstance(body, list):
        sessions = body
    else:
        sessions = body.get("sessions") or body.get("timestamps") or body.get("input_data") or []

    if not isinstance(sessions, list) or len(sessions) < 2:
        raise HTTPException(status_code=400, 
            detail="Send a list like [72, 78, 75, ...] or {\"sessions\": [72, 78, ...]}")

    try:
        sessions = [float(x) for x in sessions]
    except:
        raise HTTPException(status_code=400, detail="All values must be numbers")

    current = sum(sessions) / len(sessions)

    interval_window.append(current)

    if len(interval_window) < 2:
        return {"human_summary": "Building baseline...", "state": "Stable", "drift_score": 0.0}

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
