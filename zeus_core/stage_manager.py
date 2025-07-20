#!/usr/bin/env python3
"""
Very-small FastAPI service that:
  • reads/updates zeus_stage in Redis + Postgres
  • exposes /health  /stage  /telemetry
"""
import os, json, redis, psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ---------- connections helpers ---------------------------------------------
def _r():
    return redis.Redis(host="localhost", port=6379, decode_responses=True)

def _pg():
    return psycopg2.connect(os.getenv("DATABASE_URL"))

# ---------- API --------------------------------------------------------------
app = FastAPI()

class Telemetry(BaseModel):
    progress:   float
    confidence: float
    risk:       float
    emergence:  bool

@app.get("/health")
def health():             return {"status": "ok"}

@app.get("/stage")
def get_stage():          return {"stage": int(_r().get("zeus_stage") or 1)}

@app.post("/telemetry")
def ingest(t: Telemetry):
    stage = int(_r().get("zeus_stage") or 1)
    # naïve promotion rule
    if stage <= 3 and t.progress >= .25 and t.confidence >= .95 and t.risk <= 3:
        stage = 4
    elif stage <= 6 and (t.progress >= .60 or t.confidence >= .97):
        stage = 7
    else:
        return {"stage_updated": None, "reason": "threshold_not_met"}

    # persist both stores
    r = _r()
    r.set("zeus_stage", stage)
    pg = _pg(); c = pg.cursor()
    c.execute("UPDATE zeus.agents_state SET stage=%s WHERE agent_id='ZEUS'", (stage,))
    pg.commit(); pg.close()
    print("🛰  stage_changed →", stage, "::", json.dumps(t.dict()))
    return {"stage_updated": stage, "reason": "promotion_rule_triggered"}

# allow  `python -m zeus_core.stage_manager --port 7000`
if __name__ == "__main__":
    import sys
    port = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--port" else 7000
    uvicorn.run("zeus_core.stage_manager:app", host="0.0.0.0", port=port, reload=False)