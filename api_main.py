
from fastapi import FastAPI, Query
from typing import Optional
import pandas as pd
import os
from matcher_model import AIMatchModel, RuleBasedMatcher

DATA_PATH = os.getenv("PROFILES_CSV", "roommate_dataset.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "compat_model.joblib")
META_PATH = os.getenv("META_PATH", "compat_meta.json")

app = FastAPI(title="ApanaGhr Roommate Matcher API", version="1.0.0")

def _load_profiles(path: str):
    df = pd.read_csv(path)
    profiles = {int(r["userId"]): r.to_dict() for _, r in df.iterrows()}
    # normalize key names that our feature code expects
    for k, v in profiles.items():
        if "cleanliness" in v and "cleanlinessLevel" not in v:
            v["cleanlinessLevel"] = v["cleanliness"]
        if "budget" in v and "budgetRange" not in v:
            v["budgetRange"] = v["budget"]
        if "location" in v and "preferredLocation" not in v:
            v["preferredLocation"] = v["location"]
    return profiles

profiles = _load_profiles(DATA_PATH)

# Try to load AI model; if missing, we will fallback to rule-based
ai_model = None
if os.path.exists(MODEL_PATH):
    try:
        ai_model = AIMatchModel(MODEL_PATH, META_PATH, profiles)
    except Exception as e:
        ai_model = None

rule_model = RuleBasedMatcher(profiles)

@app.get("/health")
def health():
    return {"status": "ok", "profiles": len(profiles), "ai_model_loaded": bool(ai_model)}

@app.get("/matches/{user_id}")
def get_matches(user_id: int, N: int = Query(10, ge=1, le=50), algo: Optional[str] = Query("ai", description="ai or rule")):
    if algo == "ai" and ai_model is not None:
        out = ai_model.get_matches(user_id, N)
        return {"algo": "ai", "matches": out}
    else:
        out = rule_model.get_matches(user_id, N)
        return {"algo": "rule", "matches": out}
