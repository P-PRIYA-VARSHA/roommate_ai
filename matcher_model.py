
import joblib, os, json, numpy as np
from typing import Dict, List
from feature_utils import compute_pair_features, rule_based_score

class AIMatchModel:
    def __init__(self, model_path: str, meta_path: str = None, profiles: Dict[int, dict] = None):
        self.model = joblib.load(model_path)
        self.meta = {}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        self.profiles = profiles or {}

    def set_profiles(self, profiles: Dict[int, dict]):
        self.profiles = profiles

    def predict_pair(self, u1: dict, u2: dict) -> float:
        X = np.array([compute_pair_features(u1,u2)], dtype=float)
        y = float(self.model.predict(X)[0])
        # Clip to [0,100]
        return max(0.0, min(100.0, y))

    def get_matches(self, userId: int, N: int = 10) -> List[dict]:
        if not self.profiles:
            raise RuntimeError("Profiles not set. Call set_profiles(profiles_dict).")
        if userId not in self.profiles:
            raise KeyError(f"userId {userId} not in profiles.")

        target = self.profiles[userId]
        results = []
        for uid, prof in self.profiles.items():
            if uid == userId:
                continue
            score = self.predict_pair(target, prof)
            results.append({"userId": int(uid), "score": round(score, 2)})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:N]

class RuleBasedMatcher:
    def __init__(self, profiles: Dict[int, dict]):
        self.profiles = profiles

    def get_matches(self, userId: int, N: int = 10) -> List[dict]:
        if userId not in self.profiles:
            raise KeyError(f"userId {userId} not in profiles.")
        target = self.profiles[userId]
        res = []
        for uid, prof in self.profiles.items():
            if uid == userId: 
                continue
            score = rule_based_score(target, prof)
            res.append({"userId": int(uid), "score": score})
        res.sort(key=lambda x: x["score"], reverse=True)
        return res[:N]
